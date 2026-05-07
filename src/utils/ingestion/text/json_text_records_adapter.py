from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class JsonTextRecordsAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "JSON / JSONL text records"
    is_implemented = False
