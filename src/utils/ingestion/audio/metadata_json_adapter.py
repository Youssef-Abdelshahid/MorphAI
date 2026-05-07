from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class MetadataJsonAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio metadata JSON / JSONL"
    is_implemented = False
