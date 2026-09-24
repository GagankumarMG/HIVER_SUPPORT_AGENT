#!/usr/bin/env python
"""Assemble outputs/report/final_report.md from all computed metrics.
This OVERWRITES the generated sections of the report; hand-written narrative
sections live in the checked-in outputs/report/final_report.md template and
are preserved by only replacing content between REPORT markers.

For simplicity and transparency, this script regenerates the "Results" table
and re-inserts it into the report file between the markers
<!-- RESULTS_TABLE_START --> ... <!-- RESULTS_TABLE_END -->.

Usage:
    python scripts/11_generate_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

from src.evaluation.generate_report import build_comparison_table
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    report_path = Path("outputs/report/final_report.md")
    if not report_path.exists():
        print(f"{report_path} not found; nothing to update. See docs/ for the template.")
        return

    table = build_comparison_table("outputs/metrics")
    content = report_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->", re.DOTALL
    )
    replacement = f"<!-- RESULTS_TABLE_START -->\n{table}\n<!-- RESULTS_TABLE_END -->"
    if pattern.search(content):
        content = pattern.sub(replacement, content)
    else:
        content += f"\n\n{replacement}\n"

    report_path.write_text(content, encoding="utf-8")
    logger.info("Updated results table in %s", report_path)
    print(f"Updated {report_path} with latest metrics.")


if __name__ == "__main__":
    main()
