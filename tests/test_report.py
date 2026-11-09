from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReport(unittest.TestCase):
    def test_only_held_out_rows_enter_report(self):
        report = load_script("08_report_tables")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "c4_dev.jsonl"
            rows = [
                {"claim_id": "eval", "seed": 0, "gold_label": "NEI", "status": "ok"},
                {"claim_id": "cal", "seed": 0, "gold_label": "NEI", "status": "ok"},
            ]
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            rows = report.load_condition_rows(path, {"eval"})
        self.assertEqual(list(rows[0]), ["eval"])

    def test_c5_retains_base_evidence_and_does_not_predict_on_ambiguous_set(self):
        gate = load_script("06_calibrate")
        row = {"pred_label": "REFUTE", "p_refute": 0.5, "p_support": 0.4, "p_nei": 0.1,
               "status": "ok", "cited_ids": ["14#3"]}
        result = gate.c5_row(row, 0.1, 0.7, "evaluation")
        self.assertEqual(result["condition"], "c5")
        self.assertEqual(result["pred_label"], "ABSTAIN")
        self.assertEqual(result["base_pred_label"], "REFUTE")
        self.assertEqual(result["cited_ids"], ["14#3"])
        self.assertEqual(result["prediction_set"], ["SUPPORT", "REFUTE"])

    def test_citation_denominator_includes_annotated_claims_without_citations(self):
        report = load_script("08_report_tables")
        rows = [
            {"gold_sentence_ids": ["10#2"], "cited_ids": ["10#2"]},
            {"gold_sentence_ids": ["11#4"], "cited_ids": []},
            {"gold_sentence_ids": [], "cited_ids": []},
        ]
        metrics = report.evidence_metrics("c4", rows)
        self.assertEqual(metrics["n_gold_evidence"], 2)
        self.assertEqual(metrics["citation_f1"], 0.5)

    def test_calibrate_report_and_predict_use_the_same_frozen_split(self):
        gate = load_script("06_calibrate")
        report = load_script("08_report_tables")
        ids = {str(i) for i in range(1, 7)}
        cal_ids, eval_ids = gate.partition_claim_ids(ids)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = {"paths": {"processed": "data", "predictions": "outputs/predictions",
                             "tables": "outputs/tables", "figures": "outputs/figures"},
                   "gate": {"alphas": [0.4]}}
            predictions = root / cfg["paths"]["predictions"]
            gate.write_jsonl(root / "data" / "claims_dev.jsonl", (
                {"claim_id": cid, "label": "SUPPORT"} for cid in sorted(ids)
            ))
            support = {"condition": "c4", "seed": 0, "status": "ok", "gold_label": "SUPPORT",
                       "pred_label": "SUPPORT", "pred_prob": 0.9, "p_support": 0.9,
                       "p_refute": 0.05, "p_nei": 0.05}
            gate.write_jsonl(predictions / "c4_dev.jsonl", (
                dict(support, claim_id=cid) for cid in sorted(ids)
            ))
            gate.write_jsonl(predictions / "c4_test.jsonl", [dict(support, claim_id="99")])
            with patch.object(gate, "REPO_ROOT", root), patch.object(gate, "load_base", return_value=cfg):
                with patch("sys.argv", ["gate"]):
                    self.assertEqual(gate.main(), 0)
                with patch("sys.argv", ["gate", "--split", "test"]):
                    self.assertEqual(gate.main(), 0)
            rows = gate.load_jsonl_as_list(predictions / "c5_dev.jsonl")
            self.assertEqual({r["claim_id"] for r in rows if r["gate_partition"] == "calibration"}, cal_ids)
            self.assertEqual({r["claim_id"] for r in rows if r["gate_partition"] == "evaluation"}, eval_ids)
            self.assertEqual(gate.load_jsonl_as_list(predictions / "c5_test.jsonl")[0]["gold_label"], None)
            with patch.object(report, "REPO_ROOT", root), patch.object(report, "load_base", return_value=cfg):
                with patch("sys.argv", ["report", "--conditions", "c4,c5", "--resamples", "10"]):
                    self.assertEqual(report.main(), 0)
            text = (root / "outputs" / "tables" / "table_ladder_dev.md").read_text(encoding="utf-8")
            self.assertIn("Shared evaluation claims: 3 / 3.", text)
