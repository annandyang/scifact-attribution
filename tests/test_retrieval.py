"""Retrieval arithmetic: RRF, BM25, recall@k and nDCG@k."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.retrieval.bm25 import BM25Index, FrozenBM25, tokenize
from scifact_attr.retrieval.evaluate import ndcg_at_k, recall_at_k
from scifact_attr.retrieval.fuse import rrf


class TestFuse(unittest.TestCase):
    def test_hand_computed_order(self):
        fused = rrf([["a", "b", "c"], ["b"]], k=60)
        scores = dict(fused)
        self.assertAlmostEqual(scores["a"], 1 / 61)
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(scores["c"], 1 / 63)
        self.assertEqual(fused[0][0], "b")   # b appears in both lists and should win

    def test_ties_break_deterministically(self):
        first = rrf([["b", "a"], ["a", "b"]], k=60)
        second = rrf([["b", "a"], ["a", "b"]], k=60)
        self.assertEqual(first, second)
        self.assertEqual([d for d, _ in first], ["a", "b"])   # alphabetical tie-break

    def test_top_k_truncates(self):
        self.assertEqual(len(rrf([["a", "b", "c"]], top_k=2)), 2)

    def test_weights_shift_ranking(self):
        fused = rrf([["a", "b"], ["b", "a"]], weights=[2.0, 1.0])
        self.assertEqual(fused[0][0], "a")

    def test_empty(self):
        self.assertEqual(rrf([]), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = [("1", "alpha beta gamma"), ("2", "beta delta"), ("3", "epsilon")]
        self.index = BM25Index(k1=0.9, b=0.4).fit(self.docs)

    def test_tokenise(self):
        self.assertEqual(tokenize("Alpha-Beta, 3.5 mg!"), ["alpha", "beta", "3", "5", "mg"])

    def test_rare_term_ranks_first(self):
        hits = self.index.search("epsilon", top_k=3)
        self.assertEqual(hits[0][0], "3")

    def test_unseen_term_scores_zero_everywhere(self):
        self.assertEqual(self.index.search("zzz", top_k=3), [])

    def test_two_documents_share_term(self):
        hits = dict(self.index.search("beta", top_k=3))
        self.assertIn("1", hits)
        self.assertIn("2", hits)

    def test_scores_are_non_negative(self):
        for term in ("alpha", "beta", "gamma", "delta", "epsilon"):
            scores = self.index.get_scores(term)
            self.assertTrue((scores >= 0).all())

    def test_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "idx.pkl"
            self.index.save(path)
            loaded = BM25Index.load(path)
            self.assertEqual(loaded.search("epsilon", 3), self.index.search("epsilon", 3))

    def test_frozen_rankings_match_queries_and_cannot_serve_a_deeper_search(self):
        claims = [{"claim_id": "1", "claim": "beta"}]
        hits = [{"doc_id": int(doc_id), "score": score}
                for doc_id, score in self.index.search("beta", top_k=2)]
        ranking = {"claim_id": "1", "claim": "beta", "results": hits}
        fixed = FrozenBM25(claims, [ranking], depth=2)
        self.assertEqual(fixed.search("beta", top_k=1), hits[:1])
        with self.assertRaisesRegex(ValueError, "depth"):
            fixed.search("beta", top_k=3)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            FrozenBM25(claims, [{**ranking, "claim": "gamma"}], depth=2)


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_hit_and_miss(self):
        self.assertEqual(recall_at_k(["1", "2", "3"], ["3"], 3), 1.0)
        self.assertEqual(recall_at_k(["1", "2", "3"], ["9"], 3), 0.0)
        self.assertEqual(recall_at_k(["1", "2", "3"], ["3"], 2), 0.0)

    def test_recall_without_gold_is_nan(self):
        value = recall_at_k(["1"], [], 3)
        self.assertNotEqual(value, value)   # NaN

    def test_ndcg_perfect_and_worst(self):
        self.assertAlmostEqual(ndcg_at_k(["1", "2"], ["1"], 10), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["2", "1"], ["1"], 10), 1 / 1.5849625007211563)

    def test_ndcg_two_gold_at_top(self):
        # gains 1,1 at ranks 1,2 -> dcg = 1 + 1/log2(3); identical ideal
        self.assertAlmostEqual(ndcg_at_k(["1", "2", "3"], ["1", "2"], 10), 1.0)


if __name__ == "__main__":
    unittest.main()
