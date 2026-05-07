from __future__ import annotations

from pathlib import Path
from typing import List


def file_must_exist(path: Path) -> List[str]:
    errors: List[str] = []
    if not path or not Path(path).exists():
        errors.append(f"Input path does not exist: {path}")
    return errors
