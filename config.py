"""Central configuration loading for HIVER_SUPPORT_AGENT.

Precedence: environment variables (.env) > configs/config.yaml defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


def _load_yaml(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Settings:
    """Resolved runtime settings."""

    data_path: str
    brand_handle: str | None
    openai_api_key: str | None
    openai_model: str
    raw_config: dict[str, Any] = field(default_factory=dict)
    intents_config: dict[str, Any] = field(default_factory=dict)
    brand_config: dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.raw_config.get("seed", 42))

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested getter, e.g. settings.get('classifier', 'confidence_threshold')."""
        node: Any = self.raw_config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_settings() -> Settings:
    cfg = _load_yaml("config.yaml")
    intents_cfg = _load_yaml("intents.yaml")
    brand_cfg = _load_yaml("brand.yaml")

    data_path = os.environ.get("DATA_PATH") or cfg.get("data", {}).get(
        "raw_path", "data/raw/twcs.csv"
    )
    brand_handle = os.environ.get("BRAND_HANDLE") or None
    openai_api_key = os.environ.get("OPENAI_API_KEY") or None
    openai_model = os.environ.get("OPENAI_MODEL") or cfg.get("evaluation", {}).get(
        "llm_judge_model", "gpt-4o-mini"
    )

    return Settings(
        data_path=data_path,
        brand_handle=brand_handle,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        raw_config=cfg,
        intents_config=intents_cfg,
        brand_config=brand_cfg,
    )


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to repo root, creating parent dirs if needed."""
    p = REPO_ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
