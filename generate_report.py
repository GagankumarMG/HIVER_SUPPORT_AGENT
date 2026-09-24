"""Assembles outputs/report/final_report.md from computed metrics JSON files.
Never writes a number that isn't present in the metrics files it reads; any
missing/pending metric is rendered as "PENDING (see below)" with the exact
command needed to produce it.
"""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(value, pct: bool = False, digits: int = 3) -> str:
    if value is None:
        return "PENDING"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table(metrics_dir: str | Path) -> str:
    metrics_dir = Path(metrics_dir)
    systems = {
        "Baseline A (trivial)": metrics_dir / "baseline_a_metrics.json",
        "Baseline B (TF-IDF+LR)": metrics_dir / "baseline_b_metrics.json",
        "Main System": metrics_dir / "main_system_metrics.json",
    }
    header = (
        "| System | Intent Macro-F1 | Intent Accuracy | Reply Grounding | "
        "Reply Helpfulness | Unsafe Auto-Handling | Automation Rate |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, path in systems.items():
        m = load_json(path)
        if m is None:
            rows.append(f"| {name} | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |")
            continue
        cls = m.get("classification", {})
        esc = m.get("escalation", {})
        judge = m.get("judge_reply_quality", {})
        rows.append(
            "| {name} | {f1} | {acc} | {ground} | {help} | {unsafe} | {auto} |".format(
                name=name,
                f1=_fmt(cls.get("macro_f1")),
                acc=_fmt(cls.get("accuracy"), pct=True),
                ground=_fmt(judge.get("mean_groundedness")),
                help=_fmt(judge.get("mean_helpfulness")),
                unsafe=_fmt(esc.get("unsafe_auto_handling_rate"), pct=True),
                auto=_fmt(esc.get("automation_rate"), pct=True),
            )
        )
    return header + "\n".join(rows)
