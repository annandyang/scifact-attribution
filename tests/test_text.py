"""Text handling: the sentence splitter and the output parser.

Both have a silent failure mode. A splitter that breaks on "et al." changes every prompt's
sentence numbering; a parser that defaults to a label hides format-following failures and
inflates accuracy. Hence the tests here are about specific, awkward inputs.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.corpus import build_sentence_table, sentence_key, split_sentences
from scifact_attr.reader.parse import citation_metrics, is_usable, normalise_label, parse_output


class TestSentenceSplitter(unittest.TestCase):
    def test_two_plain_sentences(self):
        self.assertEqual(len(split_sentences("One thing happened. Another thing followed.")), 2)

    def test_et_al_does_not_split(self):
        text = "Smith et al. (2020) reported a reduction. The effect was small."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].startswith("Smith et al."))

    def test_decimal_does_not_split(self):
        text = "The dose was 3.5 mg per day. No adverse events occurred."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_initials_do_not_split(self):
        text = "J. Smith and K. Lin measured the outcome. It was inconclusive."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_country_abbreviation(self):
        text = "We studied U.S. adults over two years. Results were mixed."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_numbered_list_keeps_its_marker(self):
        sentences = split_sentences("1. First result. 2. Second result.")
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].startswith("1."))

    def test_trailing_sentence_without_period(self):
        self.assertEqual(len(split_sentences("This ends without punctuation")), 1)

    def test_empty_and_whitespace(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n  "), [])

    def test_deterministic(self):
        text = "Alpha beta. Gamma delta. Epsilon zeta."
        self.assertEqual(split_sentences(text), split_sentences(text))

    def test_sentence_key_format(self):
        self.assertEqual(sentence_key(5, 0), "5#0")


class TestSentenceTable(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"doc_id": 1, "sentences": ["Alpha beta.", "Shared sentence."]},
            {"doc_id": 2, "sentences": ["Shared sentence.", "Gamma delta."]},
        ]

    def test_dedupe_shares_one_record(self):
        records, lookup = build_sentence_table(self.corpus, dedupe=True)
        self.assertEqual(len(records), 3)                   # 4 sentences, 1 duplicate
        self.assertEqual(lookup[(1, 1)], lookup[(2, 0)])

    def test_no_dedupe_keeps_both(self):
        records, lookup = build_sentence_table(self.corpus, dedupe=False)
        self.assertEqual(len(records), 4)
        self.assertNotEqual(lookup[(1, 1)], lookup[(2, 0)])

    def test_lookup_covers_every_sentence(self):
        _, lookup = build_sentence_table(self.corpus)
        self.assertEqual(len(lookup), 4)


class TestParser(unittest.TestCase):
    def test_plain_label(self):
        parsed = parse_output("LABEL: SUPPORT")
        self.assertEqual(parsed["label"], "SUPPORT")
        self.assertEqual(parsed["status"], "ok")

    def test_spaced_nei(self):
        self.assertEqual(parse_output("LABEL: NOT ENOUGH INFO")["label"], "NEI")

    def test_contract_maps_to_refute(self):
        self.assertEqual(parse_output("label: contract")["label"], "REFUTE")

    def test_normalise_label_directly(self):
        self.assertEqual(normalise_label("Not Enough Information"), "NEI")
        self.assertIsNone(normalise_label("banana"))

    def test_reasoning_block_with_citations(self):
        text = ('EVIDENCE: E1 "the trial found no effect"; E3 "a later study agreed"\n'
                "LABEL: REFUTE\n"
                "RATIONALE: E1 contradicts the claim.")
        parsed = parse_output(text, sid_by_alias={"E1": "100#0", "E3": "200#2"},
                              expect_reasoning=True)
        self.assertEqual(parsed["label"], "REFUTE")
        self.assertEqual(parsed["cited_ids"], ["100#0", "200#2"])
        self.assertIn("contradicts", parsed["rationale"])
        self.assertTrue(is_usable(parsed))

    def test_explicit_doc_sent_ids_are_preferred(self):
        text = 'EVIDENCE: [100#0] the trial found no effect\nLABEL: REFUTE\nRATIONALE: x'
        parsed = parse_output(text, sid_by_alias={}, expect_reasoning=True)
        self.assertEqual(parsed["cited_ids"], ["100#0"])

    def test_out_of_context_citation_is_flagged(self):
        text = "EVIDENCE: [999999#1]\nLABEL: SUPPORT\nRATIONALE: It supports the claim."
        parsed = parse_output(text, sid_by_alias={"E1": "100#0"}, expect_reasoning=True)
        self.assertEqual(parsed["cited_ids"], [])
        self.assertEqual(parsed["status"], "format_violation")

    def test_reasoning_output_cannot_list_alternative_labels(self):
        text = "EVIDENCE: E1\nLABEL: SUPPORT or REFUTE\nRATIONALE: E1 is related."
        parsed = parse_output(text, sid_by_alias={"E1": "100#0"}, expect_reasoning=True)
        self.assertIsNone(parsed["label"])

    def test_unparseable_output_is_flagged_not_defaulted(self):
        parsed = parse_output("I think this is probably fine.")
        self.assertIsNone(parsed["label"])
        self.assertEqual(parsed["status"], "label_unparseable")
        self.assertFalse(is_usable(parsed))

    def test_empty_output(self):
        self.assertEqual(parse_output("")["status"], "empty")

    def test_citation_precision_hand_computed(self):
        metrics = citation_metrics(["100#0", "100#1"], ["100#0"])
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=6)

    def test_citation_precision_with_no_citations(self):
        self.assertEqual(citation_metrics([], ["100#0"])["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
