from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.utils.file_utils import safe_suffix


_SUFFIX_HINTS = {
    ".csv": ("Tabular", "csv_excel"),
    ".tsv": ("Tabular", "csv_excel"),
    ".xlsx": ("Tabular", "csv_excel"),
    ".xls": ("Tabular", "csv_excel"),
    ".json": (None, "json_records"),
    ".jsonl": (None, "json_records"),
    ".xml": ("Tabular", "xml_records"),
    ".yaml": ("Tabular", "yaml_records"),
    ".yml": ("Tabular", "yaml_records"),
    ".zip": (None, "zip_folder"),
}


def hint_format_from_suffix(path: Path) -> Optional[tuple]:
    return _SUFFIX_HINTS.get(safe_suffix(path))
