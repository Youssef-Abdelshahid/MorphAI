from src.utils.ingestion.audio.zip_folder_adapter import ZipFolderAudioAdapter
from src.utils.ingestion.audio.metadata_csv_adapter import MetadataCsvAudioAdapter
from src.utils.ingestion.audio.metadata_json_adapter import MetadataJsonAudioAdapter
from src.utils.ingestion.audio.internal import (
    AudioSample,
    InternalAudioDataset,
    SUPPORTED_AUDIO_EXTS,
    SUPPORTED_METADATA_EXTS,
    annotation_or_reference_profile,
    audio_profile_summary,
    materialize_for_pipeline,
    has_class_labels,
    has_speaker_labels,
    has_speaker_pairs,
    has_transcripts,
    has_temporal_segments,
    has_event_labels,
    has_anomaly_labels,
    has_noisy_clean_pairs,
)


AUDIO_ADAPTERS = {
    "zip_folder": ZipFolderAudioAdapter,
    "metadata_csv": MetadataCsvAudioAdapter,
    "metadata_json": MetadataJsonAudioAdapter,
}


def get_audio_adapter(format_key: str):
    cls = AUDIO_ADAPTERS.get(format_key)
    if cls is None:
        return None
    return cls()


__all__ = [
    "ZipFolderAudioAdapter",
    "MetadataCsvAudioAdapter",
    "MetadataJsonAudioAdapter",
    "AUDIO_ADAPTERS",
    "get_audio_adapter",
    "AudioSample",
    "InternalAudioDataset",
    "SUPPORTED_AUDIO_EXTS",
    "SUPPORTED_METADATA_EXTS",
    "annotation_or_reference_profile",
    "audio_profile_summary",
    "materialize_for_pipeline",
    "has_class_labels",
    "has_speaker_labels",
    "has_speaker_pairs",
    "has_transcripts",
    "has_temporal_segments",
    "has_event_labels",
    "has_anomaly_labels",
    "has_noisy_clean_pairs",
]
