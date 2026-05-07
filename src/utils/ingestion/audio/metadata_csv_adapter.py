from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class MetadataCsvAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio metadata CSV"
    is_implemented = False
