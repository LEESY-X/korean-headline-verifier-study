"""Measure verifier throughput and compare SDPA kernel paths on an isolated GPU."""

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import time

from evaluate import load_calls, object_payload, read_jsonl
from verify import prompts_for


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def runtime(args, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cases = {row["id"]: row for row in read_jsonl(args.cases)}
    prompts = list(prompts_for(cases, load_calls(args.inference), "original"))
    random.Random(20261126).shuffle(prompts)
    prompts = prompts[:args.titles]
    jobs = [(uid, tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=False), add_special_tokens=False))
            for uid, messages in prompts]
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    results = []
    reference = {}

    def generate(batch):
        encoded = tokenizer.pad([features for _, features in batch], return_tensors="pt").to("cuda")
        output = model.generate(**encoded, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return output[:, encoded.input_ids.shape[1]:], encoded.attention_mask.numel()

    with torch.inference_mode():
        for batch_size in args.batches:
            for scheduling in ("original", "length_sorted"):
                ordered = jobs if scheduling == "original" else sorted(jobs, key=lambda job: len(job[1]["input_ids"]))
                generate(ordered[:batch_size])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                durations, trials, padding, decisions = [], [], 0, {}
                for _ in range(args.repetitions):
                    trial_begin = time.perf_counter()
                    for index in range(0, len(ordered), batch_size):
                        batch = ordered[index:index + batch_size]
                        begin = time.perf_counter()
                        output, slots = generate(batch)
                        torch.cuda.synchronize()
                        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                        durations.append(time.perf_counter() - begin)
                        padding += slots - sum(len(features["input_ids"]) for _, features in batch)
                        decisions.update((uid, object_payload(raw).get("judgment"))
                                         for (uid, _), raw in zip(batch, decoded, strict=True))
                    trials.append(time.perf_counter() - trial_begin)
                if not reference:
                    reference = decisions
                results.append(dict(
                    batch_size=batch_size, scheduling=scheduling, titles=len(jobs), repetitions=args.repetitions,
                    title_per_sec=median(len(jobs) / trial for trial in trials), trial_seconds=trials,
                    batch_ms_p50=1000 * median(durations), batch_ms_p95=1000 * percentile(durations, 0.95),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    padding_tokens=padding, judgment_changes=sum(reference[uid] != result for uid, result in decisions.items()),
                ))
        profile = []
        profile_available = None
        if args.profile:
            from torch.profiler import ProfilerActivity, profile as record

            try:
                with record(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as measurement:
                    generate(jobs[:8])
                    torch.cuda.synchronize()
                top = sorted(measurement.key_averages(), key=lambda event: event.self_device_time_total, reverse=True)
                profile = [{"operation": item.key, "self_cuda_us": item.self_device_time_total}
                           for item in top[:20] if item.self_device_time_total > 0]
            except RuntimeError:
                profile = []
            profile_available = bool(profile)
    return {"mode": "runtime", "model_revision": args.checkpoint.name, "results": results,
            "profile_available": profile_available, "profile": profile}


def attention(args, torch):
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    dim = config.get("head_dim", config["hidden_size"] // heads)
    implementations = [("auto", None), ("flash", SDPBackend.FLASH_ATTENTION),
                       ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)]
    results = []
    torch.manual_seed(20261126)
    with torch.inference_mode():
        for batch in (1, 8, 32):
            for context in (256, 1024, 2048):
                q = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.bfloat16)
                key, value = (torch.randn(batch, kv_heads, context, dim, device="cuda", dtype=torch.bfloat16)
                              for _ in range(2))
                for layout in ("uniform", "masked_mixed") if batch > 1 else ("uniform",):
                    mask = None
                    if layout == "masked_mixed":
                        lengths = torch.arange(batch, device="cuda").remainder(4).add(1) * context // 4
                        mask = (torch.arange(context, device="cuda")[None, :] < lengths[:, None])[:, None, None, :]
                    with sdpa_kernel(SDPBackend.MATH):
                        reference = functional.scaled_dot_product_attention(q.float(), key.float(), value.float(),
                                                                            attn_mask=mask, enable_gqa=heads != kv_heads)
                    for label, backend in implementations:
                        row = dict(batch=batch, context=context, layout=layout, implementation=label)
                        try:
                            with (sdpa_kernel(backend) if backend else nullcontext()):
                                for _ in range(7):
                                    output = functional.scaled_dot_product_attention(q, key, value, attn_mask=mask,
                                                                                   enable_gqa=heads != kv_heads)
                                before = torch.cuda.Event(enable_timing=True)
                                after = torch.cuda.Event(enable_timing=True)
                                intervals = []
                                for _ in range(9):
                                    before.record()
                                    for _ in range(25):
                                        output = functional.scaled_dot_product_attention(
                                            q, key, value, attn_mask=mask, enable_gqa=heads != kv_heads)
                                    after.record()
                                    after.synchronize()
                                    intervals.append(1000 * before.elapsed_time(after) / 25)
                            error = (output.float() - reference).abs()
                            row.update(median_us=median(intervals), p95_us=percentile(intervals, 0.95),
                                       max_abs_error=float(error.max()), mean_abs_error=float(error.mean()))
                        except RuntimeError:
                            row.update(unavailable=True)
                        results.append(row)
    return {"mode": "attention", "heads": heads, "kv_heads": kv_heads, "head_dim": dim, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("runtime", "attention"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--titles", type=int, default=48)
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 4, 8, 16, 32))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.mode == "runtime" and not (args.cases and args.inference):
        parser.error("Runtime measurements require case data and initial inference.")
    if args.out.exists() or min(args.titles, args.repetitions, *args.batches) < 1:
        parser.error("Use a fresh output path and positive benchmark sizes.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    results = runtime(args, torch) if args.mode == "runtime" else attention(args, torch)
    results["gpu"] = torch.cuda.get_device_name(0)
    results["torch"] = torch.__version__
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(results, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
