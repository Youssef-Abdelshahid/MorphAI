from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.text.internal import InternalTextDataset, parse_json_text_records


_SUPPORTED_EXTS = {".json", ".jsonl", ".ndjson"}


class JsonTextRecordsAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "JSON / JSONL text records"
    is_implemented = True
    format_key = "json_text_records"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() not in _SUPPORTED_EXTS:
            errors.append(
                "JSON / JSONL text records requires a .json, .jsonl, or .ndjson file."
            )
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(self, path: Path, record_path: str = "", **kwargs) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        ok, err, df, structure, parsing_summary, warnings = parse_json_text_records(path, record_path=record_path)
        if not ok or df is None:
            return AdapterResult(ok=False, message=err, errors=[err])
        if df.empty:
            return AdapterResult(ok=False, message="No usable records found in JSON / JSONL.", errors=["empty parsed dataset"])

        structure_profile: Dict[str, Any] = dict(structure)
        structure_profile.update({
            "input_format": self.format_key,
            "structure_type": "json_text_records",
            "columns": [str(c) for c in df.columns],
        })

        ps: Dict[str, Any] = dict(parsing_summary)
        ps.update({
            "input_format": self.format_key,
            "source_format": path.suffix.lower().lstrip("."),
            "conversion_strategy": "flatten_dot_paths",
        })

        dataset = InternalTextDataset(
            modality="text",
            input_format=self.format_key,
            original_format=f"text_json_records_{path.suffix.lower().lstrip('.') or 'json'}",
            dataframe=df,
            structure_profile=structure_profile,
            parsing_summary=ps,
            field_mapping={c: c for c in df.columns},
            warnings=list(warnings or []),
            dataset_root=path.parent,
        )
        return AdapterResult(ok=True, data={"internal_dataset": dataset})
