from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class InputFormat:
    key: str
    label: str
    modality: str
    implemented: bool
    file_dialog_filters: Tuple[Tuple[str, str], ...]
    file_dialog_title: str
    coming_soon_hint: str = ""


_TABULAR_CSV_FILTERS = (
    ("CSV files", "*.csv"),
    ("Excel files", "*.xlsx *.xls"),
    ("All files", "*.*"),
)
_TABULAR_JSON_FILTERS = (
    ("JSON / JSONL", "*.json *.jsonl *.ndjson"),
    ("All files", "*.*"),
)
_TABULAR_XML_FILTERS = (
    ("XML files", "*.xml"),
    ("All files", "*.*"),
)
_TABULAR_YAML_FILTERS = (
    ("YAML files", "*.yaml *.yml"),
    ("All files", "*.*"),
)
_TEXT_CSV_FILTERS = (
    ("CSV files", "*.csv"),
    ("Excel files", "*.xlsx *.xls"),
    ("All files", "*.*"),
)
_IMAGE_ZIP_FILTERS = (
    ("Zip archives", "*.zip"),
    ("All files", "*.*"),
)
_AUDIO_ZIP_FILTERS = (
    ("Zip archives", "*.zip"),
    ("All files", "*.*"),
)
_PLACEHOLDER_FILTERS = (("All files", "*.*"),)


INPUT_FORMATS: Dict[str, List[InputFormat]] = {
    "Tabular": [
        InputFormat(
            key="csv_excel",
            label="CSV / Excel",
            modality="Tabular",
            implemented=True,
            file_dialog_filters=_TABULAR_CSV_FILTERS,
            file_dialog_title="Select CSV or Excel dataset",
        ),
        InputFormat(
            key="json_records",
            label="JSON / JSONL records",
            modality="Tabular",
            implemented=True,
            file_dialog_filters=_TABULAR_JSON_FILTERS,
            file_dialog_title="Select JSON / JSONL records",
        ),
        InputFormat(
            key="xml_records",
            label="XML records",
            modality="Tabular",
            implemented=True,
            file_dialog_filters=_TABULAR_XML_FILTERS,
            file_dialog_title="Select XML records",
        ),
        InputFormat(
            key="yaml_records",
            label="YAML records",
            modality="Tabular",
            implemented=True,
            file_dialog_filters=_TABULAR_YAML_FILTERS,
            file_dialog_title="Select YAML records",
        ),
    ],
    "Text": [
        InputFormat(
            key="csv_excel",
            label="CSV / Excel text table",
            modality="Text",
            implemented=True,
            file_dialog_filters=_TEXT_CSV_FILTERS,
            file_dialog_title="Select CSV or Excel text dataset",
        ),
        InputFormat(
            key="json_text_records",
            label="JSON / JSONL text records",
            modality="Text",
            implemented=False,
            file_dialog_filters=_PLACEHOLDER_FILTERS,
            file_dialog_title="Select JSON / JSONL text records",
            coming_soon_hint="Use CSV / Excel for now.",
        ),
        InputFormat(
            key="txt_zip",
            label="TXT document folder / ZIP",
            modality="Text",
            implemented=False,
            file_dialog_filters=(("Zip archives", "*.zip"), ("All files", "*.*")),
            file_dialog_title="Select TXT document folder ZIP",
            coming_soon_hint="Use CSV / Excel for now.",
        ),
    ],
    "Image": [
        InputFormat(
            key="zip_folder",
            label="Image folder / ZIP",
            modality="Image",
            implemented=True,
            file_dialog_filters=_IMAGE_ZIP_FILTERS,
            file_dialog_title="Select image dataset ZIP archive",
        ),
        InputFormat(
            key="coco",
            label="COCO JSON annotations",
            modality="Image",
            implemented=True,
            file_dialog_filters=_IMAGE_ZIP_FILTERS,
            file_dialog_title="Select COCO ZIP archive",
        ),
        InputFormat(
            key="pascal_voc",
            label="Pascal VOC XML annotations",
            modality="Image",
            implemented=True,
            file_dialog_filters=_IMAGE_ZIP_FILTERS,
            file_dialog_title="Select Pascal VOC ZIP archive",
        ),
        InputFormat(
            key="yolo",
            label="YOLO annotation format",
            modality="Image",
            implemented=True,
            file_dialog_filters=_IMAGE_ZIP_FILTERS,
            file_dialog_title="Select YOLO annotation ZIP archive",
        ),
    ],
    "Audio": [
        InputFormat(
            key="zip_folder",
            label="Audio folder / ZIP",
            modality="Audio",
            implemented=True,
            file_dialog_filters=_AUDIO_ZIP_FILTERS,
            file_dialog_title="Select audio dataset ZIP archive",
        ),
        InputFormat(
            key="metadata_csv",
            label="Audio metadata CSV",
            modality="Audio",
            implemented=False,
            file_dialog_filters=_PLACEHOLDER_FILTERS,
            file_dialog_title="Select audio metadata CSV",
            coming_soon_hint="Use Audio folder / ZIP for now.",
        ),
        InputFormat(
            key="metadata_json",
            label="Audio metadata JSON / JSONL",
            modality="Audio",
            implemented=False,
            file_dialog_filters=_PLACEHOLDER_FILTERS,
            file_dialog_title="Select audio metadata JSON",
            coming_soon_hint="Use Audio folder / ZIP for now.",
        ),
    ],
}


def get_input_formats(modality: str) -> List[InputFormat]:
    return list(INPUT_FORMATS.get(modality, []))


def get_input_format(modality: str, key_or_label: str) -> Optional[InputFormat]:
    for fmt in INPUT_FORMATS.get(modality, []):
        if fmt.key == key_or_label or fmt.label == key_or_label:
            return fmt
    return None


def is_supported(modality: str, key_or_label: str) -> bool:
    fmt = get_input_format(modality, key_or_label)
    return bool(fmt and fmt.implemented)


def get_adapter(modality: str, key_or_label: str):
    from src.utils.ingestion.base_adapter import BaseFormatAdapter, UnsupportedFormatError

    fmt = get_input_format(modality, key_or_label)
    if fmt is None:
        raise UnsupportedFormatError(
            f"No input format '{key_or_label}' is registered for modality '{modality}'."
        )

    if modality == "Tabular":
        from src.utils.ingestion.tabular import get_tabular_adapter
        adapter = get_tabular_adapter(fmt.key)
        if adapter is not None:
            return adapter

    if modality == "Image":
        from src.utils.ingestion.image import get_image_adapter
        adapter = get_image_adapter(fmt.key)
        if adapter is not None:
            return adapter

    adapter = BaseFormatAdapter()
    adapter.modality = modality
    adapter.input_format = fmt.label
    adapter.is_implemented = fmt.implemented
    return adapter
