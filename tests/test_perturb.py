import importlib
import random
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class PerturbationTests(unittest.TestCase):
    def test_perturbations_use_c4_and_label_first_scores_without_note(self):
        e4 = importlib.import_module("09_evidence_perturb")

        class Reader:
            def __init__(self):
                self.events = []

            def score_labels(self, prompt):
                has_note = "Evidence assessment" in prompt
                self.events.append(("score", has_note))
                return {"SUPPORT": 0.1 if has_note else 0.8,
                        "REFUTE": 0.8 if has_note else 0.1, "NOT_ENOUGH_INFO": 0.1}

            def generate(self, prompts, *, batch_size, max_new_tokens, use_adapter):
                self.events.append(("generate", len(prompts), batch_size, use_adapter))
                return ["E1: The sentence addresses the claim."] * len(prompts)

        row = {"claim_id": "2", "seed": 0, "gold_label": "REFUTE", "pred_label": "REFUTE",
               "status": "ok", "selected_ids": ["10#0", "10#1"], "cited_ids": ["10#0"],
               "retrieved_ids": [10]}
        claim = {"claim": "An example claim.", "gold_doc_ids": [10]}
        sentences = {sid: {"doc_id": doc, "sent_idx": idx, "text": text}
                     for sid, doc, idx, text in [("10#0", 10, 0, "Cited."),
                                                 ("10#1", 10, 1, "Uncited."),
                                                 ("20#0", 20, 0, "Distractor.")]}
        cfg = e4.load_condition("c4")
        gen = {**cfg["reader"]["gen"], "batch_size": 12}
        reader = Reader()
        result, jobs = e4.prepare_case(row, claim, sentences, cfg, "replace")
        e4.run_jobs(reader, jobs, gen, seed=0)

        self.assertEqual(reader.events, [("score", False), ("generate", 3, 12, False),
                                         ("score", True), ("score", True)])
        self.assertEqual(result["label_first"], "SUPPORT")
        self.assertEqual(result["cited_label"], "REFUTE")
        self.assertEqual(result["cited_assessment_ids"], ["20#0"])
        self.assertEqual(result["label_first_assessment_ids"], ["10#0"])
        summary = e4.summarise([result], 0, "replace")
        self.assertEqual(summary["label_first_accuracy_delta"], -1.0)
        self.assertEqual(summary["label_first_assessment_citation_rate"], 1.0)

    def test_replacement_uses_equal_counts_and_same_distractor(self):
        e4 = importlib.import_module("09_evidence_perturb")
        selected = ["10#0", "10#1", "10#2"]
        pair = e4.matched_pair(selected, ["10#1"], ["20#0"], random.Random(4))
        cited, control = pair
        self.assertEqual(cited, ["10#0", "20#0", "10#2"])
        self.assertEqual(len(control), len(selected))
        self.assertEqual(control[1], "10#1")
        self.assertEqual(control.count("20#0"), 1)
        self.assertIsNone(e4.matched_pair(selected, selected[:2], ["20#0", "20#1"],
                                         random.Random(4)))


if __name__ == "__main__":
    unittest.main()
