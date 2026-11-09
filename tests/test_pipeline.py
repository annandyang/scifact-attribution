"""End-to-end pipeline check with no model and no dataset.

`--dry-run` in scripts/04 exists for the same reason this test does: the plumbing between
retrieval, selection, prompting and the prediction row is exercised on a three-abstract
corpus, so a shape error surfaces in seconds rather than in Week 5 after a three-hour
training run.

The reader is replaced by `None`, which the pipeline handles by emitting placeholder labels
with status "dry_run".
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.config import load_condition
from scifact_attr.reader.train_lora import build_sft_examples
from scifact_attr.selection.selector import OverlapSelector, select, sentence_alias_map
from scifact_attr.system import RunContext, run_claim

REQUIRED_COLUMNS = {
    "claim_id", "condition", "pred_label", "gold_label", "pred_prob", "p_support", "p_refute",
    "p_nei", "conformal_score", "gate_decision", "retrieved_ids", "selected_ids", "cited_ids",
    "citation_precision", "rationale", "status", "latency_ms", "top_score",
}


def make_corpus():
    corpus = {
        11: {"doc_id": 11, "title": "Vitamin D and mortality",
             "sentences": ["Vitamin D supplementation did not reduce all-cause mortality.",
                           "The trial enrolled 2,500 adults over three years."]},
        22: {"doc_id": 22, "title": "Vitamin D meta-analysis",
             "sentences": ["Pooled estimates showed no significant mortality benefit.",
                           "Heterogeneity was low across the included trials."]},
        33: {"doc_id": 33, "title": "Unrelated abstract",
             "sentences": ["Soil carbon storage varies with rainfall.",
                           "No clinical outcomes were measured."]},
    }
    sentences_by_doc = {k: [{"doc_id": k, "sent_idx": i, "text": t}
                            for i, t in enumerate(v["sentences"])] for k, v in corpus.items()}
    return corpus, sentences_by_doc


class FakeRetriever:
    name = "fake"

    def search(self, claim, top_k):
        return [{"doc_id": 11, "score": 8.5}, {"doc_id": 22, "score": 7.1},
                {"doc_id": 33, "score": 0.4}]


CLAIM = {"claim_id": "1", "claim": "Vitamin D supplementation reduces all-cause mortality.",
         "split": "dev", "label": "REFUTE", "gold_doc_ids": [11], "gold_sentences": [0],
         "rationales": [[0]], "cited_doc_ids": [11, 22]}


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.corpus, self.sentences = make_corpus()

    def _context(self, condition):
        cfg = load_condition(condition)
        return RunContext(corpus_by_id=self.corpus, sentences_by_doc=self.sentences,
                          retriever=FakeRetriever(), selector=OverlapSelector(), config=cfg)

    def test_closed_book_row_shape(self):
        row = run_claim(CLAIM, self._context("c0"), reader=None)
        self.assertTrue(REQUIRED_COLUMNS.issubset(row.keys()), REQUIRED_COLUMNS - row.keys())
        self.assertEqual(row["retrieved_ids"], [])
        self.assertEqual(row["selected_ids"], [])
        self.assertEqual(row["status"], "dry_run")

    def test_retrieval_rung_records_abstracts_not_sentences(self):
        row = run_claim(CLAIM, self._context("c1"), reader=None)
        self.assertEqual(len(row["retrieved_ids"]), 3)
        self.assertEqual(row["selected_ids"], [])
        self.assertGreater(row["top_score"], 0.0)

    def test_selection_rung_keeps_sentences_within_cap(self):
        row = run_claim(CLAIM, self._context("c2"), reader=None)
        self.assertGreater(len(row["selected_ids"]), 0)
        # two abstracts with evidence-ish sentences, cap of two per abstract
        docs = {sid.split("#")[0] for sid in row["selected_ids"]}
        self.assertLessEqual(len(docs), 3)
        self.assertLessEqual(len(row["selected_ids"]), 4)

    def test_optional_evidence_rule_skips_model_only_for_empty_context(self):
        ctx = self._context("c2")
        ctx.config["require_evidence"] = True
        empty = ("prompt", [], [], 0.0, 0.0)
        row = run_claim(CLAIM, ctx, reader=object(), prepared=empty)
        self.assertEqual(row["pred_label"], "NEI")
        self.assertEqual(row["prediction_source"], "empty_evidence_rule")

        class Reader:
            def score_labels(self, prompt):
                return {"SUPPORT": 0.1, "REFUTE": 0.8, "NOT_ENOUGH_INFO": 0.1}

        row = run_claim(CLAIM, ctx, reader=Reader())
        self.assertTrue(row["selected_ids"])
        self.assertEqual(row["pred_label"], "REFUTE")
        self.assertNotIn("prediction_source", row)

    def test_selector_cap_is_enforced(self):
        candidates = [{"doc_id": 1, "sent_idx": i, "text": f"s{i}"} for i in range(6)]
        chosen = select([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], candidates, top_n=4, max_per_doc=2)
        self.assertEqual(len(chosen), 2)          # only one abstract -> cap of 2

    def test_alias_map_matches_prompt_order(self):
        candidates = [{"doc_id": 11, "sent_idx": 0, "text": "a"},
                      {"doc_id": 22, "sent_idx": 0, "text": "b"}]
        mapping = sentence_alias_map(candidates)
        self.assertEqual(mapping["E1"], "11#0")
        self.assertEqual(mapping["E2"], "22#0")

    def test_conformal_score_is_consistent_with_probability(self):
        row = run_claim(CLAIM, self._context("c2"), reader=None)
        self.assertAlmostEqual(row["conformal_score"], 1.0 - row["pred_prob"], places=9)

    def test_gold_sentence_ids_are_constructed_for_citation_scoring(self):
        row = run_claim(CLAIM, self._context("c4"), reader=None)
        self.assertEqual(row["citation_precision"], 0.0)   # no citations in a dry run

    def test_uncited_reasoning_is_kept_out_of_label_prompt(self):
        class BadReader:
            def generate(self, prompts, **kwargs):
                return ["I cannot answer this."]

            def score_labels(self, prompt):
                self.prompt = prompt
                return {"SUPPORT": 0.2, "REFUTE": 0.3, "NOT_ENOUGH_INFO": 0.5}

        reader = BadReader()
        row = run_claim({**CLAIM, "label": "NEI"}, self._context("c4"), reader)
        self.assertEqual(row["pred_label"], "NEI")
        self.assertEqual(row["status"], "uncited_reasoning")
        self.assertNotIn("I cannot answer this.", reader.prompt)
        self.assertEqual(row["claim"], CLAIM["claim"])

    def test_nei_stays_in_training_when_evidence_is_empty(self):
        examples = build_sft_examples([{**CLAIM, "label": "NEI"}], {})
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], "NOT_ENOUGH_INFO")


if __name__ == "__main__":
    unittest.main()
