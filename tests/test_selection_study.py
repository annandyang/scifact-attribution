from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


spec = spec_from_file_location(
    "selection_study", Path(__file__).resolve().parents[1] / "scripts" / "14_selection_study.py"
)
study = module_from_spec(spec)
spec.loader.exec_module(study)


class SelectionStudyTests(unittest.TestCase):
    def test_neighbor_expansion_respects_document_bounds_and_keeps_anchors(self):
        candidates = [
            {"doc_id": doc, "sent_idx": pos, "text": f"{doc}-{pos}"}
            for doc in (10, 20) for pos in (0, 1, 2)
        ]
        by_id = {study.sentence_id(row): row for row in candidates}
        expanded = study.with_neighbors(["10#0", "20#1", "10#1"], by_id, limit=5)
        self.assertEqual(list(map(study.sentence_id, expanded)),
                         ["10#0", "20#1", "10#1", "20#0", "20#2"])
        self.assertEqual(study.with_neighbors([], by_id), [])

    def test_coverage_counts_a_complete_alternative_without_requiring_the_union(self):
        claim = {"gold_evidence_sets": [["1#0", "1#1"], ["2#0"]]}
        self.assertEqual(study.evidence_coverage([claim], [["2#0"]]), (1, 1, 1))
        self.assertEqual(study.evidence_coverage([claim], [["1#0"]]), (1, 1, 0))
        self.assertEqual(study.evidence_coverage([{"gold_evidence_sets": []}], [[]]), (0, 0, 0))

    def test_fixed_retrieval_must_match_claim_and_comparison(self):
        claim = {"claim_id": 10, "claim": "text", "label": "REFUTE"}
        prediction = {"claim_id": 10, "claim": "text", "gold_label": "REFUTE",
                      "seed": 0, "mode": "experiment", "status": "ok",
                      "condition": "c1", "retrieved_ids": [2, 3]}
        self.assertEqual(study.load_predictions([prediction], [claim], "c1", 0)[10], prediction)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            study.load_predictions([prediction, prediction], [claim], "c1", 0)
        with self.assertRaisesRegex(ValueError, "Retrieval differs"):
            study.require_same_retrieval(10, prediction, {**prediction, "retrieved_ids": [3, 2]})

    def test_clarification_scores_the_same_evidence_with_a_different_prompt(self):
        claim = {"claim_id": 1, "claim": "A causes B.", "label": "NEI"}
        old = {1: {"retrieved_ids": [10], "selected_ids": ["10#0"], "pred_label": "SUPPORT"}}
        docs = {10: [{"doc_id": 10, "sent_idx": 0, "text": "A was observed."}]}

        class Reader:
            def score_labels(self, prompt):
                if "same topic" in prompt:
                    return {"NOT_ENOUGH_INFO": 1.0}
                return {"SUPPORT": 1.0}

        results = study.run_part([claim], ("original", "clarify"), old, docs, None,
                                 Reader(), study.load_condition("c3"), "calibration")
        original, clarified = results["original"][0], results["clarify"][0]
        self.assertEqual(original["selected_ids"], clarified["selected_ids"])
        self.assertEqual((original["pred_label"], clarified["pred_label"]), ("SUPPORT", "NEI"))
