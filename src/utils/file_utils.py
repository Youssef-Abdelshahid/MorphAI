from __future__ import annotations

from pathlib import Path


def safe_suffix(path: Path) -> str:
    try:
        return Path(path).suffix.lower()
    except Exception:
        return ""
