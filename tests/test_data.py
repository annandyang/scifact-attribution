import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.data import load_claims, load_corpus


class TestOfficialSciFact(unittest.TestCase):
    def load_rows(self, rows, loader, *args):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            return loader(path, *args)

    def test_official_sentence_indices_are_preserved(self):
        sentences = ["Methods:", "We measured pH."] + [f"Sentence {i}." for i in range(12)]
        corpus = self.load_rows(
            [{"doc_id": 9, "title": "Example", "abstract": sentences}], load_corpus,
        )
        self.assertEqual(corpus[0]["sentences"], sentences)

    def test_official_rationales_preserve_documents_and_alternatives(self):
        claims = self.load_rows([{
            "id": 1,
            "claim": "Claim one.",
            "evidence": {
                "10": [{"label": "CONTRADICT", "sentences": [0, 4]},
                       {"label": "CONTRADICT", "sentences": [7]}],
                "11": [{"label": "CONTRADICT", "sentences": [4]}],
            },
            "cited_doc_ids": [10, 11, 20],
        }], load_claims, "dev")
        self.assertEqual(claims[0]["label"], "REFUTE")
        self.assertEqual(claims[0]["gold_doc_ids"], [10, 11])
        self.assertEqual(claims[0]["gold_sentence_ids"], ["10#0", "10#4", "10#7", "11#4"])
        self.assertEqual(claims[0]["gold_evidence_sets"], [["10#0", "10#4"], ["10#7"], ["11#4"]])

    def test_empty_evidence_and_absent_annotations_are_distinct(self):
        labeled = self.load_rows([{"id": 1, "claim": "Claim.", "evidence": {}}], load_claims, "train")
        unlabeled = self.load_rows([{"id": 2, "claim": "Claim."}], load_claims, "test")
        self.assertEqual(labeled[0]["label"], "NEI")
        self.assertIsNone(unlabeled[0]["label"])


if __name__ == "__main__":
    unittest.main()
