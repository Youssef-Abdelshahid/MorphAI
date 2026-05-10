from src.utils.ingestion.image.zip_folder_adapter import ZipFolderImageAdapter
from src.utils.ingestion.image.coco_adapter import CocoImageAdapter
from src.utils.ingestion.image.pascal_voc_adapter import PascalVocImageAdapter
from src.utils.ingestion.image.yolo_adapter import YoloImageAdapter
from src.utils.ingestion.image.internal import (
    InternalImageDataset,
    ImageSample,
    materialize_for_pipeline,
    has_class_label,
    has_bboxes,
    has_masks,
    has_keypoints,
    has_text_labels,
    has_depth_targets,
)


IMAGE_ADAPTERS = {
    "zip_folder": ZipFolderImageAdapter,
    "coco": CocoImageAdapter,
    "pascal_voc": PascalVocImageAdapter,
    "yolo": YoloImageAdapter,
}


def get_image_adapter(format_key: str):
    cls = IMAGE_ADAPTERS.get(format_key)
    if cls is None:
        return None
    return cls()


__all__ = [
    "ZipFolderImageAdapter",
    "CocoImageAdapter",
    "PascalVocImageAdapter",
    "YoloImageAdapter",
    "IMAGE_ADAPTERS",
    "get_image_adapter",
    "InternalImageDataset",
    "ImageSample",
    "materialize_for_pipeline",
    "has_class_label",
    "has_bboxes",
    "has_masks",
    "has_keypoints",
    "has_text_labels",
    "has_depth_targets",
]
