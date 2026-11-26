#!/usr/bin/env bash
set -euo pipefail

job=$1
python=$2
base_model=$3
other_model=$4
cases=$5
src=$(cd -- "$(dirname -- "$0")" && pwd)
trap 'rc=$?; printf "%s\n" "$rc" > "$job/exit_code"' EXIT
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

for order in original reversed; do
  echo "STAGE baseline $order"
  "$python" -u "$src/infer.py" --cases "$cases" --order "$order" --checkpoint "$base_model" \
    --batch-size 8 --out "$job/base-$order.jsonl"
  for verifier in same other; do
    echo "STAGE verifier $order $verifier"
    checkpoint=$base_model
    if test "$verifier" = other; then checkpoint=$other_model; fi
    "$python" -u "$src/verify.py" --cases "$cases" --inference "$job/base-$order.jsonl" --order "$order" \
      --checkpoint "$checkpoint" --batch-size 8 --out "$job/$verifier-$order.jsonl"
  done
  echo "STAGE scoring $order"
  "$python" -B "$src/evaluate.py" --cases "$cases" --order "$order" --inference "$job/base-$order.jsonl" \
    --same-model "$job/same-$order.jsonl" --other-model "$job/other-$order.jsonl" > "$job/summary-$order.json"
done
echo 'STAGE done'
