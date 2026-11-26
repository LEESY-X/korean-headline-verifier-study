"""Run a pinned local Transformers checkpoint on the fictional headlines."""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path


ROLES = {
    "general": "제목의 핵심 주장을 편집 데스크에서 일반 검토한다.",
    "quantity": "수치, 단위와 그 수치가 가리키는 대상을 우선 대조한다.",
    "certainty": "검토, 확정, 취소, 시행 상태가 섞이지 않았는지 우선 대조한다.",
    "scope": "적용 대상과 조건, 시간 범위가 바뀌었는지 우선 대조한다.",
    "attribution": "발언이나 결정 또는 조치를 한 주체를 우선 대조한다.",
}
JUDGMENTS = {"supported", "misleading", "uncertain"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def make_prompt(row, role, source_order):
    evidence = "\n".join(
        f"{position}. {row['sentences'][source_id - 1]}"
        for position, source_id in enumerate(source_order, 1)
    )
    system = (
        "너는 발행 전 뉴스 제목을 검토하는 한국어 편집자다. " + ROLES[role]
        + " 역할과 상관없이 제목의 핵심 주장을 기사 본문 전체와 대조한다. "
        "제목이 원문에 의해 뒷받침되면 supported, 수치·상태·적용 범위·행위 주체 등이 "
        "원문과 실질적으로 다르면 misleading, 판단에 필요한 정보가 본문에 없으면 uncertain을 선택한다. "
        "원문의 진위는 판단하지 않는다. 판단에 가장 직접적인 문장 번호를 evidence_id에, "
        "그 문장을 토씨까지 그대로 quote에 적어라. 정보 부족일 때는 evidence_id=0과 빈 quote를 사용한다. "
        '다른 내용 없이 JSON 객체 하나만 답한다. 키는 "judgment", "evidence_id", "quote"다.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"제목\n{row['headline']}\n\n본문\n{evidence}"},
    ]


def parse(raw, sentences, source_order):
    start = raw.find("{")
    if start < 0:
        return {"status": "no_json"}
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (json.JSONDecodeError, RecursionError):
        return {"status": "invalid_json"}
    judgment = value.get("judgment")
    number = value.get("evidence_id")
    quote = value.get("quote")
    if not isinstance(judgment, str) or judgment not in JUDGMENTS:
        return {"status": "invalid_judgment"}
    source_id = source_order[number - 1] if type(number) is int and 1 <= number <= len(source_order) else None
    citation_match = bool(
        source_id is not None and isinstance(quote, str)
        and " ".join(quote.split()) == " ".join(sentences[source_id - 1].split())
    )
    return {
        "status": "ok",
        "judgment": judgment,
        "evidence_id": source_id,
        "citation_match": citation_match,
    }


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("practice", "test"))
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--order", choices=("original", "reversed"), default="original")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, output length and optional limit must be positive.")
    return args


def main():
    args = read_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("This runner requires a CUDA GPU.")
    cases = [row for row in read_jsonl(args.cases) if args.split is None or row["split"] == args.split]
    random.Random(args.seed).shuffle(cases)
    cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No matching cases.")
    jobs = [(row, role) for row in cases for role in ROLES]
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    device = torch.cuda.get_device_properties(0)
    metadata = {
        "checkpoint_revision": args.checkpoint.name,
        "data_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "code_sha256": code_hash,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        "case_ids": [row["id"] for row in cases],
        "order": args.order,
        "attention": args.attn,
        "precision": "bfloat16",
        "gpu": device.name,
        "gpu_memory_bytes": device.total_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.out.with_suffix(".manifest.json")
    timings = args.out.with_suffix(".timing.json")
    if args.out.exists():
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise SystemExit("Existing result uses a different experiment configuration.")
        previous = read_jsonl(args.out)
    else:
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        previous = []
    done = {(row["id"], row["role"]) for row in previous}
    jobs = [(row, role) for row, role in jobs if (row["id"], role) not in done]
    if not jobs:
        print("All requested calls already exist.")
        return

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=args.attn,
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"Loaded {device.name} in {time.perf_counter() - load_start:.2f}s. {len(jobs)} calls pending.", flush=True)
    torch.cuda.reset_peak_memory_stats()
    batch_records = json.loads(timings.read_text(encoding="utf-8"))["batches"] if timings.exists() else []

    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    stop_ids.add(tokenizer.pad_token_id)
    with args.out.open("a", encoding="utf-8", newline="\n") as output, torch.inference_mode():
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            orders = [list(range(1, len(row["sentences"]) + 1)) for row, _ in batch]
            if args.order == "reversed":
                orders = [order[::-1] for order in orders]
            prompts = [
                tokenizer.apply_chat_template(make_prompt(row, role, order), tokenize=False, add_generation_prompt=True)
                for (row, role), order in zip(batch, orders, strict=True)
            ]
            encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
            input_counts = encoded.attention_mask.sum(dim=1).tolist()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            new_tokens = generated[:, encoded.input_ids.size(1):].tolist()
            counts = []
            for (row, role), order, tokens, inputs in zip(batch, orders, new_tokens, input_counts, strict=True):
                length = next((i + 1 for i, token in enumerate(tokens) if token in stop_ids), len(tokens))
                raw = tokenizer.decode(tokens[:length], skip_special_tokens=True)
                decision = {"id": row["id"], "role": role, "input_tokens": inputs, "output_tokens": length,
                            "output": raw, **parse(raw, row["sentences"], order)}
                counts.append(length)
                output.write(json.dumps(decision, ensure_ascii=False) + "\n")
            output.flush()
            batch_records.append({"calls": len(batch), "input_tokens": sum(input_counts),
                                  "output_tokens": sum(counts), "seconds": seconds,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            timings.write_text(json.dumps({"batches": batch_records}, indent=2), encoding="utf-8")
            if len(batch_records) == 1 or len(batch_records) % 5 == 0 or start + len(batch) >= len(jobs):
                print(f"{start + len(batch)}/{len(jobs)} calls. Last batch {seconds:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
