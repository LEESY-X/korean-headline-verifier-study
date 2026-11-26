"""Audit evidence and score paired routing policies under a fixed review budget."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


SPECIALISTS = ("quantity", "certainty", "scope", "attribution")
LABELS = {"supported", "misleading"}
POLICIES = ("single", "vote", "citation", "same_model", "other_model")
SEED = 20261126


@dataclass(frozen=True)
class Evidence:
    usable_id: int | None = None
    exact_copy: bool = False
    number_status: str = "unavailable"


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def object_payload(raw):
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
    except (ValueError, RecursionError, AttributeError):
        return {}
    return payload


def audit_evidence(record, sentences, order):
    payload = object_payload(record["output"])
    if not payload:
        return Evidence()
    number = payload.get("evidence_id")
    if type(number) is int:
        status = "integer"
    elif isinstance(number, str) and number.isascii() and number.isdecimal() and len(number) < 10:
        number, status = int(number), "string_number"
    else:
        return Evidence(number_status="invalid_type")
    quote = payload.get("quote")
    if number == 0 and payload.get("judgment") == "uncertain" and quote == "":
        label = "abstention" if status == "integer" else "string_abstention"
        return Evidence(number_status=label)
    if not 1 <= number <= len(sentences):
        return Evidence(number_status="out_of_range")
    source_id = number if order == "original" else len(sentences) + 1 - number
    exact = isinstance(quote, str) and quote.split() == sentences[source_id - 1].split()
    return Evidence(source_id, exact, status)


def load_calls(path):
    records = defaultdict(dict)
    for row in read_jsonl(path):
        if row["role"] in records[row["id"]]:
            raise ValueError(f"Duplicate model output for {row['id']} and {row['role']}")
        records[row["id"]][row["role"]] = row
    expected = {"general", *SPECIALISTS}
    if not records or any(set(item) != expected for item in records.values()):
        raise ValueError("Each included title needs one general and four specialist outputs.")
    manifest = path.with_suffix(".manifest.json")
    if manifest.exists() and set(json.loads(manifest.read_text(encoding="utf-8"))["case_ids"]) != set(records):
        raise ValueError("Model outputs do not match the inference manifest.")
    return records


def verifier_calls(path, expected_ids):
    calls = read_jsonl(path)
    predictions = {row["id"]: row["judgment"] for row in calls}
    if set(predictions) != expected_ids or len(predictions) != len(calls):
        raise ValueError("Verifier outputs must match the title IDs exactly once.")
    if not set(predictions.values()) <= {"supports", "contradicts", "insufficient", "invalid"}:
        raise ValueError("Verifier judgment outside the registered schema.")
    return predictions


def binary(record):
    label = record.get("judgment")
    return label if record.get("status") == "ok" and label in LABELS else None


def derive(cases, records, order):
    audits = Counter()
    rows = []
    for uid, calls in records.items():
        case = cases[uid]
        evidence = {role: audit_evidence(call, case["sentences"], order) for role, call in calls.items()}
        for role, call in calls.items():
            item = evidence[role]
            audits[f"evidence_id_{item.number_status}"] += 1
            audits["model_parse_errors"] += call.get("status") != "ok"
            audits["literal_quote_matches"] += item.exact_copy
            audits["annotated_span_matches"] += item.exact_copy and item.usable_id == case.get("gold_evidence_id")
        votes = Counter(binary(calls[role]) for role in SPECIALISTS)
        verdict = next((label for label in sorted(LABELS) if votes[label] >= 3), None)
        cited = [evidence[role].usable_id for role in SPECIALISTS
                 if verdict and binary(calls[role]) == verdict and evidence[role].exact_copy]
        rows.append(dict(
            id=uid, family=case["group"], gold=case["gold"], single=binary(calls["general"]),
            decision=verdict, votes=votes[verdict] if verdict else 0,
            citations=len(cited), same_span=max(Counter(cited).values(), default=0),
            single_cite=evidence["general"].exact_copy,
            tie=(-int(hashlib.sha256(case["headline"].encode()).hexdigest(), 16), uid),
        ))
    return rows, audits


def route(row, policy, verifiers):
    single = policy == "single"
    decision = row["single"] if single else row["decision"]
    if policy in verifiers:
        expected = {"supported": "supports", "misleading": "contradicts"}.get(decision)
        mismatch = verifiers[policy][row["id"]] != expected
    else:
        mismatch = False
    use_citation = policy != "vote"
    citations = int(row["single_cite"]) if single else row["citations"]
    return (decision is not None, not mismatch, row["votes"] if not single else 0,
            citations if use_citation else 0, row["same_span"] if use_citation and not single else 0,
            row["tie"]), decision


def rates(items):
    budget = len(items) // 4
    ordered = sorted(items, key=lambda item: item[0])
    accepted = ordered[budget:]
    misleading = sum(gold == "misleading" for _, _, gold, _ in ordered)
    if misleading == 0:
        raise ValueError("Need at least one misleading title in every scored sample.")
    errors = Counter()
    for _, decision, gold, unanimous in accepted:
        if decision is None:
            errors["unresolved_over_budget"] += 1
        elif decision != gold:
            misses = gold == "misleading"
            errors["false_pass" if misses else "false_flag"] += 1
            errors["unanimous_false_pass"] += misses and unanimous
    return dict(
        titles=len(items), reviewed=budget, misleading=misleading,
        false_pass=errors["false_pass"], false_pass_rate=errors["false_pass"] / misleading,
        false_flag=errors["false_flag"], unresolved_over_budget=errors["unresolved_over_budget"],
        unanimous_false_pass=errors["unanimous_false_pass"],
    )


def bootstrap(grouped, repetitions):
    rng = random.Random(SEED)
    groups = list(grouped.values())
    differences = []
    for _ in range(repetitions):
        selected = rng.choices(groups, k=len(groups))
        same = rates([x for group in selected for x in group[0]])
        other = rates([x for group in selected for x in group[1]])
        if not (same["unresolved_over_budget"] or other["unresolved_over_budget"]):
            differences.append(100 * (other["false_pass_rate"] - same["false_pass_rate"]))
    differences.sort()
    retained = len(differences)
    interval = [differences[round((retained - 1) * p)] for p in (0.025, 0.975)] if retained else None
    return {"runs": repetitions, "feasible_runs": retained, "ci95_pp_when_feasible": interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--order", default="original", choices=("original", "reversed"))
    parser.add_argument("--same-model", type=Path)
    parser.add_argument("--other-model", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if bool(args.same_model) != bool(args.other_model) or args.bootstrap_draws < 1:
        parser.error("Provide both verifier output files together and use positive draw counts.")

    case_rows = read_jsonl(args.cases)
    cases = {row["id"]: row for row in case_rows}
    if len(cases) != len(case_rows) or any(row["gold"] not in LABELS for row in case_rows):
        parser.error("Duplicate case IDs or invalid gold labels.")
    records = load_calls(args.inference)
    rows, audits = derive(cases, records, args.order)
    families = defaultdict(set)
    for row in rows:
        families[row["family"]].add(row["gold"])
    if any(labels != LABELS for labels in families.values()):
        parser.error("Every article family must include both headline labels.")
    verifiers = {}
    if args.same_model:
        verifiers = {key: verifier_calls(path, set(records)) for key, path in
                     (("same_model", args.same_model), ("other_model", args.other_model))}
    tables = {}
    grouped = defaultdict(lambda: ([], []))
    for policy in (POLICIES if verifiers else POLICIES[:3]):
        ranked = []
        for row in rows:
            rank, decision = route(row, policy, verifiers)
            item = (rank, decision, row["gold"], row["votes"] == 4)
            ranked.append(item)
            if policy in verifiers:
                grouped[row["family"]][policy == "other_model"].append(item)
        tables[policy] = rates(ranked)
    report = dict(
        order=args.order, families=len(families), role_calls=5 * len(rows),
        evidence_audit=dict(audits), policies=tables,
    )
    if verifiers:
        same, other = tables["same_model"], tables["other_model"]
        feasible = not (same["unresolved_over_budget"] or other["unresolved_over_budget"])
        effect = 100 * (other["false_pass_rate"] - same["false_pass_rate"])
        report["primary"] = dict(
            feasible=bool(feasible), difference_pp=effect if feasible else None,
            bootstrap=bootstrap(grouped, args.bootstrap_draws) if feasible else None,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
