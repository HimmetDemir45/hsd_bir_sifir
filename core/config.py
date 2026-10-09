"""config.yaml yükleyici. ${DEGISKEN} değerlerini .env / ortam değişkenlerinden doldurur."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f))


def resolve_path(p: str | Path) -> Path:
    """Proje köküne göre göreli yolu mutlak yola çevirir."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
