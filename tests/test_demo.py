import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import gradio_app


class DemoCaseTests(unittest.TestCase):
    def test_selects_a_real_label_flip_and_an_abstention_from_dev(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)

            def save(condition, split, rows):
                contents = ''.join(json.dumps({**row, 'condition': condition, 'seed': 0}) + '\n'
                                   for row in rows)
                (path / f'{condition}_{split}.jsonl').write_text(contents, encoding='utf-8')

            save('c0', 'dev', [
                {'claim_id': '10', 'pred_label': 'SUPPORT', 'pred_prob': 0.9},
                {'claim_id': '20', 'pred_label': 'SUPPORT', 'pred_prob': 0.9},
                {'claim_id': '30', 'pred_label': 'SUPPORT', 'pred_prob': 0.9},
            ])
            save('c3', 'dev', [
                {'claim_id': '10', 'gold_label': 'REFUTE', 'pred_label': 'SUPPORT',
                 'top_score': 0.1, 'gold_sentence_ids': ['42#0'], 'selected_ids': ['42#0']},
                {'claim_id': '20', 'gold_label': 'REFUTE', 'pred_label': 'REFUTE',
                 'top_score': 0.1, 'gold_sentence_ids': ['43#0'], 'selected_ids': ['43#0']},
            ])
            save('c5', 'dev', [
                {'claim_id': '20', 'gold_label': 'REFUTE', 'pred_label': 'REFUTE', 'top_score': 0.1},
                {'claim_id': '30', 'gold_label': 'NEI', 'pred_label': 'ABSTAIN',
                 'gate_decision': 'ABSTAIN', 'gate_partition': 'evaluation',
                 'gate_threshold': 0.7, 'conformal_score': 0.65},
            ])
            save('c5', 'test', [
                {'claim_id': '99', 'gold_label': 'REFUTE', 'pred_label': 'REFUTE', 'top_score': 0.5},
            ])

            with patch.object(gradio_app, 'PREDICTIONS', path):
                self.assertEqual(gradio_app.pick_demo_claims(n=2), ['20', '30'])


if __name__ == '__main__':
    unittest.main()
