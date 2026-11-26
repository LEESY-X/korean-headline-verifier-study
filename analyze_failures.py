"""Summarize unreviewed errors by risk and evidence type without emitting article text."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from evaluate import POLICIES, SPECIALISTS, audit_evidence, derive, load_calls, read_jsonl, route, verifier_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--same-model", type=Path, required=True)
    parser.add_argument("--other-model", type=Path, required=True)
    parser.add_argument("--order", choices=("original", "reversed"), default="original")
    args = parser.parse_args()
    cases = {row["id"]: row for row in read_jsonl(args.cases)}
    calls = load_calls(args.inference)
    rows, _ = derive(cases, calls, args.order)
    verifiers = {"same_model": verifier_calls(args.same_model, set(calls)),
                 "other_model": verifier_calls(args.other_model, set(calls))}
    result = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        case = cases[row["id"]]
        risk = case.get("risk", "unspecified")
        result["reference"][risk][case["gold"]] += 1
        if row["votes"] == 4 and row["decision"] != case["gold"]:
            result["all_wrong_unanimous"][risk]["headlines"] += 1
            if any(audit_evidence(calls[row["id"]][role], case["sentences"], args.order).exact_copy
                   for role in SPECIALISTS):
                result["all_wrong_unanimous"][risk]["with_at_least_one_exact_quote"] += 1
        if case["gold"] == "misleading":
            for role in SPECIALISTS:
                evidence = audit_evidence(calls[row["id"]][role], case["sentences"], args.order)
                if evidence.exact_copy and evidence.usable_id == case.get("gold_evidence_id"):
                    result["exact_decisive_counterevidence"][risk]["specialist_calls"] += 1
                    if calls[row["id"]][role].get("judgment") == "supported":
                        result["exact_decisive_counterevidence"][risk]["wrongly_said_supported"] += 1
    for policy in POLICIES:
        ranked = [(*route(row, policy, verifiers), row) for row in rows]
        for position, (_, decision, row) in enumerate(sorted(ranked, key=lambda item: item[0])):
            kind = cases[row["id"]].get("risk", "unspecified")
            stats = result[policy][kind]
            if position < len(rows) // 4:
                stats["reviewed"] += 1
            elif decision == "supported" and row["gold"] == "misleading":
                stats["false_pass"] += 1
            elif decision == "misleading" and row["gold"] == "supported":
                stats["false_flag"] += 1
            elif decision is None:
                stats["unresolved_over_budget"] += 1
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
