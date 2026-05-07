from __future__ import annotations

from src.utils.ingestion.base_adapter import BaseFormatAdapter


class ZipFolderAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio folder / ZIP"
    is_implemented = True
