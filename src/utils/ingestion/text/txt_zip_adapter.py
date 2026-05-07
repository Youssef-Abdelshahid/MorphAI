from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class TxtZipTextAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "TXT document folder / ZIP"
    is_implemented = False
