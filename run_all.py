#!/usr/bin/env python
"""Orchestrates the full pipeline end-to-end.

--fast: uses the bundled synthetic dataset (data/raw/twcs_synthetic.csv) and
        skips regenerating it if already present, for a <15-minute demo run.
--full: expects DATA_PATH in .env to point at the real Kaggle CSV.

Usage:
    python scripts/run_all.py --fast
    python scripts/run_all.py --full
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STEPS = [
    ["python", "scripts/00_inspect_dataset.py"],
    ["python", "scripts/01_prepare_data.py"],
    ["python", "scripts/02_select_brand.py"],
    ["python", "scripts/03_discover_intents.py"],
    ["python", "scripts/04_create_golden_candidates.py"],
    # NOTE: golden set annotation is a required HUMAN step between here and
    # scripts/05_train.py's leakage exclusion being fully effective; run_all
    # proceeds so the mechanical pipeline is demonstrably runnable, but
    # headline evaluation numbers are only trustworthy after real annotation.
    ["python", "scripts/05_train.py"],
    ["python", "scripts/06_build_index.py"],
    ["python", "scripts/08_evaluate.py"],
    ["python", "scripts/09_run_llm_judge.py"],
    ["python", "scripts/09b_human_agreement.py"],
    ["python", "scripts/10_analyze_failures.py"],
    ["python", "scripts/12_leakage_check.py"],
    ["python", "scripts/14_run_metadata.py"],
    ["python", "scripts/11_generate_report.py"],
    ["python", "scripts/13_quality_gate.py"],
]


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.fast:
        synthetic_path = Path("data/raw/twcs_synthetic.csv")
        if not synthetic_path.exists():
            subprocess.run(
                ["python", "scripts/00b_make_synthetic_dataset.py"], check=True, env=env
            )
        env["DATA_PATH"] = str(synthetic_path)
        print("Running in --fast mode using the bundled SYNTHETIC demo dataset.")
    else:
        if not env.get("DATA_PATH"):
            print("ERROR: --full mode requires DATA_PATH set in .env to the real "
                  "Kaggle CSV location.")
            sys.exit(1)
        print(f"Running in --full mode using DATA_PATH={env['DATA_PATH']}")

    for step in STEPS:
        print(f"\n{'='*70}\n>>> {' '.join(step)}\n{'='*70}")
        proc = subprocess.run(step, env=env)
        if proc.returncode != 0:
            print(f"STEP FAILED: {' '.join(step)} (exit code {proc.returncode})")
            print("Continuing with remaining steps to surface as much diagnostic "
                  "information as possible; check outputs/metrics for PENDING/FAIL statuses.")

    print("\nPipeline run complete. See outputs/metrics/final_quality_gate.json "
          "for an overall summary.")


if __name__ == "__main__":
    main()
