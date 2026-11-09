"""View cached predictions and reports. No model is loaded by the demo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "predictions"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
CONDITIONS = {
    "c0": "Closed book",
    "c1": "+ Retrieval",
    "c2": "+ Selection",
    "c3": "+ LoRA",
    "c4": "+ CoT",
    "c5": "+ Conformal gate",
}


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv(path, seed):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("seed", "0") == str(seed)]


def load_predictions(split, seed, directory):
    return {
        condition: {
            str(row["claim_id"]): row
            for row in read_jsonl(directory / f"{condition}_{split}.jsonl")
            if row.get("seed", 0) == seed
        }
        for condition in CONDITIONS
    }


def usable(row):
    return bool(row) and row.get("status", "ok") == "ok" and row.get("mode") != "smoke"


def case_type(rows, cid):
    c0, c3, c5 = (rows[condition].get(cid, {}) for condition in ("c0", "c3", "c5"))
    if usable(c0) and usable(c3) and c0.get("pred_label") == "SUPPORT":
        if (c3.get("gold_label") == c3.get("pred_label") == "REFUTE"
                and set(c3.get("selected_ids") or []) & set(c3.get("gold_sentence_ids") or [])):
            return "A"
    if usable(c0) and usable(c5):
        if (c5.get("gold_label") == "NEI" and c0.get("pred_label") in {"SUPPORT", "REFUTE"}
                and (number(c0.get("pred_prob")) or 0) >= 0.5
                and c5.get("gate_decision", "").upper() == "ABSTAIN"
                and number(c5.get("gate_threshold")) is not None
                and number(c5.get("conformal_score")) is not None):
            return "B"
    return ""


def claim_choices(rows, pinned=(), n=3):
    claim_ids = sorted({cid for group in rows.values() for cid in group})
    cases = {cid: case_type(rows, cid) for cid in claim_ids}
    # Prefer held-out cases when the gate records a calibration partition.
    claim_ids.sort(key=lambda cid: rows["c5"].get(cid, {}).get("gate_partition") == "calibration")
    if pinned:
        picks = list(dict.fromkeys(str(cid) for cid in pinned))
        missing = [cid for cid in picks if cid not in cases]
        if missing:
            raise ValueError(f"no cached predictions for claim ID(s): {', '.join(missing)}")
    else:
        picks = []
        for role in ("A", "B"):
            match = next((cid for cid in claim_ids if cases[cid] == role), None)
            if match:
                picks.append(match)
        if len(picks) < n:
            switches = []
            for cid in claim_ids:
                gate = rows["c5"].get(cid, {})
                if cid in picks or gate.get("gate_partition") != "evaluation":
                    continue
                if gate.get("gate_decision") != "ABSTAIN":
                    continue
                for alpha, option in sorted(gate.get("gate_options", {}).items()):
                    if option["gate_decision"] == "answer":
                        switches.append((float(alpha), cid))
                        break
            if switches:
                picks.append(min(switches)[1])
        # A fallback is still a real claim. Only exact matches get a case letter.
        for cid in claim_ids:
            if len(picks) >= n:
                break
            if cid not in picks:
                picks.append(cid)
        picks = picks[:n]
    return [(f"Claim {cases[cid]} · {cid}" if cases[cid] else f"Claim {cid} · other case", cid)
            for cid in picks]


def pick_demo_claims(split="dev", n=3, seed=0):
    return [cid for _, cid in claim_choices(load_predictions(split, seed, PREDICTIONS), n=n)]


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decimal(value):
    value = number(value)
    return f"{value:.3f}" if value is not None else "—"


def show_claim(cid, rows, claims, corpus, sentences, alpha=None):
    cached = {condition: group.get(cid) for condition, group in rows.items()}
    if cached["c5"] and alpha:
        gate = cached["c5"]
        cached["c5"] = {**gate, **gate.get("gate_options", {}).get(alpha, {})}
    row = next((row for row in cached.values() if row), None)
    if row is None:
        return "<p>No cached predictions. Run the experiment before opening the demo.</p>"

    claim = claims.get(cid, {})
    text = claim.get("claim") or row.get("claim") or "Claim text unavailable."
    gold = claim.get("label") or next((r["gold_label"] for r in cached.values()
                                       if r and r.get("gold_label")), "unreleased")
    parts = [f"<h3>{escape(str(text))}</h3><p>Gold label: <b>{escape(str(gold))}</b></p>"]
    if any(r and r.get("status") == "dry_run" for r in cached.values()):
        parts.append("<p><b>Dry run. Placeholder labels are shown for pipeline debugging.</b></p>")
    modes = sorted({str(r["mode"]) for r in cached.values() if r and r.get("mode")})
    if "smoke" in modes:
        parts.append("<p><b>Smoke run. These samples verify the code path.</b></p>")
    parts.append('<div style="display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:0.5rem">')
    for condition, label in CONDITIONS.items():
        current = cached[condition]
        prediction = current.get("pred_label", "—") if current else "—"
        status = current.get("status", "ok") if current else "missing"
        note = {
            "missing": "No cached row",
            "dry_run": "Placeholder",
        }.get(status, status if status != "ok" else f"Confidence {decimal(current.get('pred_prob'))}")
        if prediction == "ABSTAIN":
            note = "Gate abstained"
        elif status == "ok" and gold in {"SUPPORT", "REFUTE", "NEI"}:
            note += " · correct" if prediction == gold else " · incorrect"
        accent = "2px solid #21815b" if status == "ok" and prediction == gold else "1px solid #8887"
        parts.append(
            f'<div style="border:{accent};border-radius:8px;padding:0.5rem">'
            f'<small>{condition.upper()} · {escape(label)}</small><br>'
            f'<b>{escape(str(prediction))}</b><br><small>{escape(str(note))}</small></div>'
        )
    parts.append("</div>")

    c5 = cached["c5"] or {}
    if c5.get("gate_decision"):
        threshold = number(c5.get("gate_threshold"))
        score = number(c5.get("conformal_score"))
        parts.append("<h4>C5 gate</h4>")
        parts.append(f"<p>Base label score 1 − p: <b>{decimal(score)}</b> · "
                     f"Set threshold: <b>{decimal(threshold)}</b> · "
                     f"Alpha: <b>{decimal(c5.get('gate_alpha'))}</b></p>")
        if score is not None and threshold is not None and 0 <= score <= 1 and 0 <= threshold <= 1:
            parts.append(
                '<div aria-label="Score and threshold on a zero to one scale" '
                'style="position:relative;background:#8883;border-radius:4px;height:12px">'
                f'<div style="width:{score * 100}%;height:12px;background:#387bce"></div>'
                f'<div style="position:absolute;left:{threshold * 100}%;top:-4px;height:20px;'
                'width:2px;background:currentColor"></div></div>'
                '<p><small>Blue = base label score · marker = threshold</small></p>'
            )
        if "prediction_set" in c5:
            labels = ", ".join(str(label) for label in (c5["prediction_set"] or [])) or "empty set"
            parts.append(f"<p>Prediction set: <b>{escape(labels)}</b>. "
                         "The gate answers when the set has one label, agrees with C4, "
                         "and the evidence assessment passes the format check.</p>")
        if c5.get("gate_partition"):
            parts.append(f"<p>Partition: {escape(str(c5['gate_partition']))}</p>")

    evidence_row = next((r for condition in ("c3", "c2", "c4", "c5")
                         if (r := cached[condition]) and r.get("selected_ids")), {})
    if evidence_row:
        condition = escape(str(evidence_row.get("condition", "")).upper())
        parts.append(f"<h4>Selected evidence · {condition}</h4>")
        gold_ids = set(evidence_row.get("gold_sentence_ids") or [])
        selected_ids = sorted(evidence_row["selected_ids"], key=lambda sid: sid not in gold_ids)
        for sid in selected_ids:
            doc_id, _, index = sid.partition("#")
            doc = corpus.get(doc_id, {})
            body = doc.get("sentences", [])
            sentence = sentences.get(sid)
            if not sentence and index.isdecimal() and int(index) < len(body):
                sentence = body[int(index)]
            title = doc.get("title") or "Abstract"
            parts.append(f"<p><small>{escape(title)} · {escape(sid)}</small><br>"
                         f"<mark>{escape(sentence or 'Sentence text unavailable.')}</mark></p>")
            if sid == selected_ids[0] and body:
                abstract = " ".join(f"<mark>{escape(text)}</mark>" if str(i) == index else escape(text)
                                    for i, text in enumerate(body))
                parts.append(f"<details><summary>Abstract context</summary><p>{abstract}</p></details>")
    else:
        parts.append("<p>No selected sentences are available for this claim.</p>")

    reasoning = cached["c4"] or cached["c5"] or {}
    cited = reasoning.get("cited_ids") or []
    if cited:
        parts.append(f"<p>Citations · C4/C5: {escape(', '.join(cited))}</p>")
    if reasoning.get("rationale"):
        parts.append(f"<details><summary>Generated rationale</summary>"
                     f"<p>{escape(str(reasoning['rationale']))}</p></details>")
    return "\n".join(parts)


def risk_chart(curve, calibration):
    points = [(number(row.get("coverage")), number(row.get("risk"))) for row in curve]
    points = [(x, y) for x, y in points if x is not None and y is not None and 0 <= x <= 1
              and 0 <= y <= 1]
    if not points:
        return "<p>No cached risk curve. Run the calibration step to generate it.</p>"

    def x_pos(x):
        return 68 + x * 520

    def y_pos(y):
        return 288 - y * 260

    svg = ['<svg viewBox="0 0 625 345" style="width:100%;max-width:780px" '
           'role="img" aria-label="Coverage versus error among answered claims">']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x, y = x_pos(tick), y_pos(tick)
        svg.extend([
            f'<path d="M68 {y} H588 M{x} 28 V288" stroke="currentColor" opacity="0.16"/>',
            f'<text x="{x}" y="307" text-anchor="middle" fill="currentColor">{tick:.0%}</text>',
            f'<text x="58" y="{y + 4}" text-anchor="end" fill="currentColor">{tick:.0%}</text>',
        ])
    line = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in points)
    svg.append(f'<polyline points="{line}" fill="none" stroke="#387bce" stroke-width="3"/>')
    for row in calibration:
        x, y = number(row.get("coverage")), number(row.get("selective_error"))
        if x is not None and y is not None and 0 <= x <= 1 and 0 <= y <= 1:
            alpha = decimal(row.get("alpha"))
            svg.append(f'<circle cx="{x_pos(x)}" cy="{y_pos(y)}" r="6" fill="#bf5b27">'
                       f'<title>alpha={alpha}, coverage={x:.1%}, error={y:.1%}</title></circle>')
    svg.extend([
        '<text x="328" y="333" text-anchor="middle" fill="currentColor">Coverage</text>',
        '<text x="-158" y="17" transform="rotate(-90)" text-anchor="middle" '
        'fill="currentColor">Error among answered claims</text>',
        '</svg>',
    ])
    return "\n".join(svg)


def build_app(split, seed, rows, choices, use_reports=True):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    claims = {str(row["claim_id"]): row for row in read_jsonl(PROCESSED / f"claims_{split}.jsonl")}
    corpus = {str(row["doc_id"]): row for row in read_jsonl(PROCESSED / "corpus.jsonl")}
    sentences = {row["sid"]: row["text"] for row in read_jsonl(PROCESSED / "sentences.jsonl")}
    ladder = read_csv(TABLES / f"table_ladder_{split}.csv", seed) if use_reports else []
    ladder = {row["condition"]: row for row in ladder if row["condition"] in CONDITIONS}
    calibration = read_csv(TABLES / "e5_conformal.csv", seed) if use_reports else []
    curve = read_csv(FIGURES / "risk_coverage_c5.csv", seed) if use_reports else []
    audit_path = TABLES / "e3_shortcut.md"
    first = choices[0][1] if choices else None
    gate_sample = next(iter(rows["c5"].values()), {})
    alphas = sorted(gate_sample.get("gate_options", {}))
    default_alpha = f"{gate_sample['gate_alpha']:.2f}" if alphas else None

    def render(cid, alpha):
        return show_claim(cid, rows, claims, corpus, sentences, alpha)

    with gr.Blocks(title="SciFact · Attribution Demo") as demo:
        gr.Markdown(f"# SciFact attribution\n"
                    f"Cached results · claim split **{split}** · example seed **{seed}**.")
        with gr.Tab("Claims"):
            if choices:
                picker = gr.Dropdown(choices=choices, value=first, label="Prepared claim",
                                     allow_custom_value=False)
                missing = [role for role in ("A", "B")
                           if not any(label.startswith(f"Claim {role} · ") for label, _ in choices)]
                if missing:
                    gr.Markdown("No matching cached case for " + ", ".join(missing)
                                + ". Other cases show the actual saved results.")
            alpha_choice = gr.Dropdown(choices=alphas, value=default_alpha, label="C5 alpha",
                                       visible=bool(alphas and choices))
            details = gr.HTML(value=render(first, default_alpha))
            if choices:
                picker.change(render, inputs=[picker, alpha_choice], outputs=details)
                alpha_choice.change(render, inputs=[picker, alpha_choice], outputs=details)
        with gr.Tab("Batch"):
            evaluation = "held-out development set" if split == "dev" else split
            gr.Markdown(f"### Six conditions · {evaluation} · seed {seed}")
            if default_alpha:
                gr.Markdown(f"The batch table uses the frozen C5 alpha of {default_alpha}.")
            if not ladder:
                gr.Markdown("No cached batch table. Run the reporting step after inference.")
            gr.Dataframe(
                headers=["Condition", "Labeled claims", "Accuracy", "Macro-F1", "Answer rate"],
                value=[
                    [f"{c.upper()} · {name}", ladder.get(c, {}).get("n", "—"),
                     decimal(ladder.get(c, {}).get("accuracy")),
                     decimal(ladder.get(c, {}).get("macro_f1")),
                     decimal(ladder.get(c, {}).get("coverage"))]
                    for c, name in CONDITIONS.items()
                ],
                interactive=False,
            )
            with gr.Accordion("Shortcut audit", open=False):
                gr.Markdown(audit_path.read_text(encoding="utf-8") if use_reports and audit_path.is_file()
                            else "No cached shortcut audit. Run scripts/07_shortcut_audit.py.")
        with gr.Tab("Risk–coverage"):
            gr.Markdown(f"### Held-out development results · seed {seed}\n"
                        "Blue: confidence sweep. Orange: measured conformal operating points.")
            gr.HTML(value=risk_chart(curve, calibration))
            if calibration:
                columns = {
                    "alpha": "Alpha",
                    "nominal_set_coverage": "Target set coverage",
                    "empirical_set_coverage": "Observed set coverage",
                    "coverage": "Answered fraction",
                    "selective_error": "Error among answers",
                }
                gr.Dataframe(headers=list(columns.values()),
                             value=[[decimal(row.get(key)) for key in columns]
                                    for row in calibration], interactive=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", action="append")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    pinned = args.claim_id or []
    if not pinned and args.split == "dev":
        import yaml

        with (ROOT / "configs" / "demo_claims.yaml").open(encoding="utf-8") as fh:
            pinned = (yaml.safe_load(fh) or {}).get("claims") or []
    rows = load_predictions(args.split, args.seed, args.predictions_dir)
    try:
        choices = claim_choices(rows, pinned)
    except ValueError as error:
        parser.error(str(error))
    print("Demo claims:", ", ".join(cid for _, cid in choices) or "no cached predictions")
    use_reports = args.predictions_dir.resolve() == PREDICTIONS.resolve()
    app = build_app(args.split, args.seed, rows, choices, use_reports=use_reports)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
               inbrowser=args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
