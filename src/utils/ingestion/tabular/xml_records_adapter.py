from __future__ import annotations

from pathlib import Path

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.tabular.parsers import parse_xml_records


class XmlRecordsTabularAdapter(BaseFormatAdapter):
    modality = "Tabular"
    input_format = "XML records"
    is_implemented = True
    format_key = "xml_records"

    def to_internal_dataset(self, path: Path, record_path: str = "") -> AdapterResult:
        outcome = parse_xml_records(Path(path), record_path=record_path)
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
