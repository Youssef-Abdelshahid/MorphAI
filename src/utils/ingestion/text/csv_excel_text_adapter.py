from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class CsvExcelTextAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "CSV / Excel text table"
    is_implemented = True
