# Evidence verification under a fixed review budget

This study measures whether an extra verifier changes which misleading Korean news headlines pass without human review. A.X specialist votes provide a shared base decision. The comparison adds one evidence call from A.X or from Phi-4-mini. Each policy sends the same 25% of titles to review.

The repository contains original fictional articles, experiment code and aggregate results. The synthetic pairs use controlled changes to a number, certainty, eligibility or speaker. Authors of the paired generator know the labels by construction. The repeated templates limit what the test says about real headlines.

## Reproduce the fictional evaluation

Use Python 3.12 and a CUDA GPU with enough memory for each model. The measured environment used PyTorch 2.7.1 with CUDA 12.8 and Transformers 4.57.1. Download the public [A.X 4.0 Light](https://huggingface.co/skt/A.X-4.0-Light) and [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) checkpoints under their published terms. Point the commands at local snapshot directories. Check `results/model-revisions.json` for the exact pinned revisions.

```bash
python -B -m unittest -v test_evaluate
python build_fictional.py --per-risk 30 --out fictional-120.jsonl
mkdir -p runs/fictional
bash run_evaluation.sh runs/fictional "$(command -v python)" \
  "$AX_CHECKPOINT" "$PHI_CHECKPOINT" fictional-120.jsonl
```

To measure the separate verifier on the same cases, pass the stored original baseline outputs to the benchmark. Batch latency is the time to process a batch. Model loading and upstream calls are excluded from runtime throughput.

```bash
mkdir -p runs/benchmark
python bench.py attention --checkpoint "$AX_CHECKPOINT" --out runs/benchmark/attention.json
python bench.py runtime --checkpoint "$AX_CHECKPOINT" --cases fictional-120.jsonl \
  --inference runs/fictional/base-original.jsonl --out runs/benchmark/runtime-same.json --profile
python bench.py runtime --checkpoint "$PHI_CHECKPOINT" --cases fictional-120.jsonl \
  --inference runs/fictional/base-original.jsonl --out runs/benchmark/runtime-other.json
```

## How to read the results

- The false-pass rate is the number of misleading titles accepted as supported outside the review queue divided by all misleading titles.
- The paired interval resamples article families, retains both titles from every selected family and reapplies the 25% review rule on every draw.
- Citation format, normalized quotation match and the gold evidence span are separate counts. Semantic verifier labels are scored through routing. The same output can contain a verbatim quote and a mistaken decision.
- `results/` contains aggregates from fictional inputs and synthetic GPU workloads. Exact per-title results can be regenerated with the commands above.

See [results.md](results.md) for the measured findings and [protocol.md](protocol.md) for the analysis design.
