#!/usr/bin/env python
"""Final quality-gate check. Aggregates pass/fail/pending status across the
whole pipeline. Does NOT mark a gate complete unless it actually verified so.

Usage:
    python scripts/13_quality_gate.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _exists(path: str) -> bool:
    return Path(path).exists()


def _json_status_ok(path: str, ok_predicate) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return ok_predicate(data)


def main():
    tests_passed = None
    try:
        proc = subprocess.run(["pytest", "-q"], capture_output=True, text=True, timeout=600)
        tests_passed = proc.returncode == 0
        test_output = proc.stdout[-4000:] + proc.stderr[-2000:]
    except Exception as e:  # noqa: BLE001
        test_output = str(e)

    golden_complete = _json_status_ok(
        "outputs/metrics/evaluation_status.json", lambda d: False
    ) is False and _exists("data/golden/golden_set.csv")
    # golden_set_complete really means check_golden_set.py passed; re-run it.
    try:
        check_proc = subprocess.run(
            ["python", "scripts/check_golden_set.py"], capture_output=True, text=True
        )
        golden_set_complete = check_proc.returncode == 0
    except Exception:
        golden_set_complete = False

    leakage_ok = _json_status_ok(
        "outputs/metrics/leakage_check.json", lambda d: d.get("status") == "PASS"
    )

    model_artifacts_present = all(
        _exists(p)
        for p in [
            "models/main_semantic_classifier.joblib",
            "models/baseline_b_tfidf_classifier.joblib",
            "models/hybrid_retrieval_index.joblib",
        ]
    )

    evaluation_complete = _exists("outputs/metrics/main_system_metrics.json")

    judge_evaluation_complete = _json_status_ok(
        "outputs/metrics/llm_judge_scores.json", lambda d: d.get("status") == "ok"
    )

    human_agreement_available = _json_status_ok(
        "outputs/metrics/judge_human_agreement.json", lambda d: d.get("status") == "ok"
    )

    gate = {
        "tests_passed": tests_passed,
        "golden_set_complete": golden_set_complete,
        "data_leakage_check": leakage_ok,
        "model_artifacts_present": model_artifacts_present,
        "evaluation_complete": evaluation_complete,
        "judge_evaluation_complete": judge_evaluation_complete,
        "human_agreement_available": human_agreement_available,
    }

    with open("outputs/metrics/final_quality_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print(json.dumps(gate, indent=2))
    if tests_passed is False:
        print("\n--- pytest output (tail) ---")
        print(test_output)


if __name__ == "__main__":
    main()
