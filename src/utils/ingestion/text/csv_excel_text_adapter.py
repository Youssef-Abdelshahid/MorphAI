from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.text.internal import InternalTextDataset


_SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}


class CsvExcelTextAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "CSV / Excel text table"
    is_implemented = True
    format_key = "csv_excel"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() not in _SUPPORTED_EXTS:
            errors.append(
                "CSV / Excel text table requires a .csv, .xlsx, or .xls file. "
                "Please upload a structured text table with one sample per row."
            )
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(self, path: Path, **kwargs) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        suffix = path.suffix.lower()
        try:
            if suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            return AdapterResult(ok=False, message=f"Failed to read CSV/Excel: {exc}", errors=[str(exc)])
        if df.empty:
            return AdapterResult(ok=False, message="The uploaded file is empty.", errors=["empty file"])

        structure_profile: Dict[str, Any] = {
            "input_format": self.format_key,
            "structure_type": "csv_excel_text_table",
            "original_record_count": int(len(df)),
            "parsed_record_count": int(len(df)),
            "schema_variant_count": 1,
            "nested_field_count": 0,
            "flattened_metadata_fields": [],
            "max_nesting_depth": 1,
            "columns": [str(c) for c in df.columns],
        }
        parsing_summary: Dict[str, Any] = {
            "input_format": self.format_key,
            "source_format": suffix.lstrip("."),
            "conversion_strategy": "passthrough",
            "rows_read": int(len(df)),
            "columns_read": int(len(df.columns)),
        }

        dataset = InternalTextDataset(
            modality="text",
            input_format=self.format_key,
            original_format=f"text_table_{suffix.lstrip('.') or 'csv'}",
            dataframe=df,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            field_mapping={c: c for c in df.columns},
            warnings=[],
            dataset_root=path.parent,
        )
        return AdapterResult(ok=True, data={"internal_dataset": dataset})
