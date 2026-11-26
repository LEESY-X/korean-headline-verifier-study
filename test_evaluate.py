"""Regression checks for evidence formatting and mandatory review accounting."""

import unittest

from evaluate import audit_evidence, rates


class EvaluationTests(unittest.TestCase):
    def test_numeric_string_does_not_hide_a_verbatim_quote(self):
        payload = {"output": '{"judgment":"supported","evidence_id":"2","quote":"원문 문장"}'}
        evidence = audit_evidence(payload, ["앞 문장", "원문 문장"], "original")
        self.assertEqual((evidence.usable_id, evidence.number_status, evidence.exact_copy), (2, "string_number", True))

    def test_reverse_order_points_back_to_the_original_sentence(self):
        payload = {"output": '{"judgment":"misleading","evidence_id":2,"quote":"첫 문장"}'}
        evidence = audit_evidence(payload, ["첫 문장", "둘째 문장"], "reversed")
        self.assertEqual((evidence.usable_id, evidence.exact_copy), (1, True))

    def test_false_pass_denominator_includes_reviewed_risky_items(self):
        items = [((0,), None, "misleading", False), ((1,), "supported", "misleading", True),
                 ((2,), "supported", "supported", True), ((3,), "supported", "supported", False)]
        result = rates(items)
        self.assertEqual((result["reviewed"], result["false_pass"], result["misleading"]), (1, 1, 2))
        self.assertEqual(result["false_pass_rate"], 0.5)

    def test_invalid_decisions_over_budget_remain_unresolved(self):
        items = [((0,), None, "misleading", False), ((1,), None, "supported", False),
                 ((2,), "misleading", "misleading", True), ((3,), "supported", "supported", True)]
        self.assertEqual(rates(items)["unresolved_over_budget"], 1)


if __name__ == "__main__":
    unittest.main()
