from __future__ import annotations

from pathlib import Path

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.tabular.parsers import parse_csv_excel


class CsvExcelTabularAdapter(BaseFormatAdapter):
    modality = "Tabular"
    input_format = "CSV / Excel"
    is_implemented = True
    format_key = "csv_excel"

    def to_internal_dataset(self, path: Path, record_path: str = "") -> AdapterResult:
        outcome = parse_csv_excel(Path(path))
        if not outcome.ok:
            return AdapterResult(ok=False, message=outcome.error, errors=[outcome.error])
        return AdapterResult(
            ok=True,
            data={
                "dataframe": outcome.df,
                "structure_profile": outcome.structure.to_dict(),
                "parsing_summary": outcome.parsing_summary,
                "warnings": outcome.warnings,
            },
            schema=outcome.structure.to_dict(),
        )
