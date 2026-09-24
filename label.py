"""Weak-labeling utilities to bootstrap intent labels for CLASSIFIER TRAINING
ONLY (never for the golden human-labeled evaluation set - see
docs/decision_log.md, "why generated labels are not human labels").

Given a human-named cluster -> intent mapping (produced after discover.py +
a human review of the representative examples), this module assigns intent
labels to the full training corpus by nearest-cluster-centroid assignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def apply_cluster_to_intent_map(
    df: pd.DataFrame, cluster_col: str, cluster_to_intent: dict[int, str]
) -> pd.DataFrame:
    """Map numeric cluster ids to human-assigned intent names.

    Any cluster not present in `cluster_to_intent` is mapped to
    OTHER_OR_UNKNOWN rather than silently dropped.
    """
    df = df.copy()
    df["intent"] = df[cluster_col].map(cluster_to_intent).fillna("OTHER_OR_UNKNOWN")
    unmapped = sorted(set(df[cluster_col].unique()) - set(cluster_to_intent.keys()))
    if unmapped:
        logger.warning(
            "Clusters %s had no human-provided intent name; labeled OTHER_OR_UNKNOWN. "
            "Review configs/intents.yaml and update the mapping.",
            unmapped,
        )
    return df
