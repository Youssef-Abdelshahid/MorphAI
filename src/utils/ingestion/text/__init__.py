from src.utils.ingestion.text.csv_excel_text_adapter import CsvExcelTextAdapter
from src.utils.ingestion.text.json_text_records_adapter import JsonTextRecordsAdapter
from src.utils.ingestion.text.txt_zip_adapter import TxtZipTextAdapter
from src.utils.ingestion.text.internal import (
    InternalTextDataset,
    SUPPORTED_TXT_EXTS,
    SUPPORTED_METADATA_EXTS,
    export_internal_metadata,
    parse_json_text_records,
)


TEXT_ADAPTERS = {
    "csv_excel": CsvExcelTextAdapter,
    "json_text_records": JsonTextRecordsAdapter,
    "txt_zip": TxtZipTextAdapter,
}


def get_text_adapter(format_key: str):
    cls = TEXT_ADAPTERS.get(format_key)
    if cls is None:
        return None
    return cls()


__all__ = [
    "CsvExcelTextAdapter",
    "JsonTextRecordsAdapter",
    "TxtZipTextAdapter",
    "TEXT_ADAPTERS",
    "get_text_adapter",
    "InternalTextDataset",
    "SUPPORTED_TXT_EXTS",
    "SUPPORTED_METADATA_EXTS",
    "export_internal_metadata",
    "parse_json_text_records",
]
