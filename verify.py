"""Run a blinded evidence verifier on a pinned local checkpoint."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from evaluate import SPECIALISTS, audit_evidence, binary, load_calls, object_payload, read_jsonl


JUDGMENTS = {"supports", "contradicts", "insufficient"}
INSTRUCTION = (
    "한국어 뉴스 제목의 핵심 주장을 아래 기사 본문과 대조한다. 본문 자체의 진위는 판단하지 않는다. "
    "후보 근거는 참고용이고 항상 관련된 문맥 전체를 확인한다. 수치, 확정 여부, 적용 조건, 발언 주체를 "
    "하나라도 실질적으로 다르게 적었고 본문으로 그 차이를 확인할 수 있으면 contradicts, "
    "주요 주장을 모두 뒷받침하면 supports, 본문만으로 결정할 수 없으면 insufficient다. "
    '설명 없이 JSON 객체 하나만 반환한다. 예시: {"judgment":"insufficient"}.'
)


def prompts_for(cases, prior, order):
    for uid in sorted(prior):
        case = cases[uid]
        sentences = case["sentences"]
        votes = Counter(binary(prior[uid][role]) for role in SPECIALISTS)
        verdict = next((label for label in ("supported", "misleading") if votes[label] >= 3), None)
        candidate_ids = set()
        for role in SPECIALISTS:
            record = prior[uid][role]
            if verdict is not None and binary(record) == verdict:
                citation = audit_evidence(record, sentences, order)
                if citation.exact_copy:
                    candidate_ids.add(citation.usable_id)
        indices = list(range(len(sentences), 0, -1)) if order == "reversed" else list(range(1, len(sentences) + 1))
        displayed = [str(position) for position, index in enumerate(indices, 1) if index in candidate_ids]
        candidate = ", ".join(displayed) or "없음"
        text = "\n".join(f"{position}. {sentences[index - 1]}" for position, index in enumerate(indices, 1))
        user = f"제목\n{case['headline']}\n\n후보 근거 문장 번호\n{candidate}\n\n본문\n{text}"
        yield uid, [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": user}]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--order", default="original", choices=("original", "reversed"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and token limit must be positive.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = {row["id"]: row for row in read_jsonl(args.cases)}
    jobs = list(prompts_for(cases, load_calls(args.inference), args.order))
    manifest_path = args.out.with_suffix(".manifest.json")
    timing_path = args.out.with_suffix(".timing.json")
    if any(path.exists() for path in (args.out, manifest_path, timing_path)):
        parser.error("Output paths already exist. Use a new name for a separate run.")
    torch.manual_seed(20261126)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    torch.cuda.synchronize()
    model_load = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    device = torch.cuda.get_device_properties(0)
    revisions = dict(
        checkpoint_revision=args.checkpoint.name, input_sha256=sha(args.inference), cases_sha256=sha(args.cases),
        code_sha256=sha(Path(__file__)), audit_sha256=sha(Path(__file__).with_name("evaluate.py")),
        prompt_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(), order=args.order,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, cases=len(jobs),
        torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, gpu=device.name,
    )
    manifest_path.write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")
    batches = []
    batch_peak = torch.cuda.max_memory_allocated
    with args.out.open("x", encoding="utf-8", newline="\n") as output, torch.inference_mode():
        for offset in range(0, len(jobs), args.batch_size):
            batch = jobs[offset:offset + args.batch_size]
            wall_begin = time.perf_counter()
            formatted = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False) for _, messages in batch]
            encoded = tokenizer(formatted, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            generated_secs = time.perf_counter() - start
            raw_ids = generated[:, encoded.input_ids.shape[1]:]
            for (uid, _), raw in zip(batch, tokenizer.batch_decode(raw_ids, skip_special_tokens=True), strict=True):
                judgment = object_payload(raw).get("judgment")
                if not isinstance(judgment, str) or judgment not in JUDGMENTS:
                    judgment = "invalid"
                record = {"id": uid, "judgment": judgment, "output": raw}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            batches.append({"titles": len(batch), "generate_seconds": generated_secs,
                            "wall_seconds": time.perf_counter() - wall_begin, "peak_allocated_bytes": batch_peak()})
    timing_path.write_text(json.dumps({"model_load_seconds": model_load, "batches": batches}, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
