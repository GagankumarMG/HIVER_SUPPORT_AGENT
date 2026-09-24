#!/usr/bin/env python
"""Records reproducibility metadata: versions, seeds, config snapshot.

Usage:
    python scripts/14_run_metadata.py
"""
from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version, PackageNotFoundError

from src.utils.config import load_settings

PACKAGES = [
    "pandas", "numpy", "scikit-learn", "scipy", "sentence-transformers",
    "joblib", "pydantic", "streamlit", "rank-bm25",
]


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def main():
    settings = load_settings()
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "seed": settings.seed,
        "brand": settings.brand_config.get("brand", {}),
        "classifier_confidence_threshold": settings.get(
            "classifier", "confidence_threshold", default=0.60
        ),
        "retrieval_weights": {
            "lexical": settings.get("retrieval", "lexical_weight", default=0.4),
            "semantic": settings.get("retrieval", "semantic_weight", default=0.6),
        },
        "golden_set_size_target": settings.get("evaluation", "golden_set_size", default=200),
    }
    with open("outputs/metrics/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
