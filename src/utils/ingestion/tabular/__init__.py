from src.utils.ingestion.tabular.csv_excel_adapter import CsvExcelTabularAdapter
from src.utils.ingestion.tabular.json_records_adapter import JsonRecordsTabularAdapter
from src.utils.ingestion.tabular.xml_records_adapter import XmlRecordsTabularAdapter
from src.utils.ingestion.tabular.yaml_records_adapter import YamlRecordsTabularAdapter
from src.utils.ingestion.tabular.parsers import (
    StructureProfile,
    ParseOutcome,
    parse_for_format,
)

TABULAR_ADAPTERS = {
    "csv_excel": CsvExcelTabularAdapter,
    "json_records": JsonRecordsTabularAdapter,
    "xml_records": XmlRecordsTabularAdapter,
    "yaml_records": YamlRecordsTabularAdapter,
}


def get_tabular_adapter(format_key: str):
    cls = TABULAR_ADAPTERS.get(format_key)
    if cls is None:
        return None
    return cls()


__all__ = [
    "CsvExcelTabularAdapter",
    "JsonRecordsTabularAdapter",
    "XmlRecordsTabularAdapter",
    "YamlRecordsTabularAdapter",
    "TABULAR_ADAPTERS",
    "get_tabular_adapter",
    "StructureProfile",
    "ParseOutcome",
    "parse_for_format",
]
