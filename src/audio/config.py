from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

AUDIO_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]
ASR_METRICS = ["wer", "cer", "exact_match_accuracy", "normalized_edit_similarity"]
SPEAKER_IDENTIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1"]
SPEAKER_VERIFICATION_METRICS = ["equal_error_rate", "auroc", "verification_accuracy"]
SOUND_EVENT_METRICS = ["event_f1", "segment_f1", "precision", "recall", "error_rate"]
VAD_METRICS = ["frame_f1", "precision", "recall", "false_alarm_rate", "miss_rate"]
ANOMALY_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "proxy_score", "score_separation", "reconstruction_consistency", "stability"]
NOISE_SUPPRESSION_METRICS = ["si_sdr_improvement", "snr_improvement", "spectral_distance", "proxy_score"]

_AUD_TASK_BACKEND = {
    "Audio classification": "classification",
    "Speech recognition (ASR)": "asr",
    "Speaker recognition": "speaker_recognition",
    "Sound event detection": "sound_event_detection",
    "Voice activity detection": "vad",
    "Audio anomaly detection": "anomaly",
    "Noise suppression": "noise_suppression",
}

VALID_TASK_TYPES = list(_AUD_TASK_BACKEND.values())
SUPPORTED_TASK_TYPES = set(VALID_TASK_TYPES)

DEPRECATED_TASK_TYPES = {
    "speaker_diarization",
}

TASK_DISPLAY_NAMES = {
    "classification": "Audio classification",
    "asr": "Speech recognition (ASR)",
    "speaker_recognition": "Speaker recognition",
    "sound_event_detection": "Sound event detection",
    "vad": "Voice activity detection",
    "anomaly": "Audio anomaly detection",
    "noise_suppression": "Noise suppression",
}

_TASK_FAMILIES = {
    "classification": "classification",
    "asr": "speech",
    "speaker_recognition": "speaker",
    "sound_event_detection": "event",
    "vad": "speech",
    "anomaly": "anomaly",
    "noise_suppression": "enhancement",
}

_TASK_METRICS = {
    "classification": AUDIO_CLASSIFICATION_METRICS,
    "asr": ASR_METRICS,
    "speaker_recognition": sorted(set(SPEAKER_IDENTIFICATION_METRICS + SPEAKER_VERIFICATION_METRICS)),
    "sound_event_detection": SOUND_EVENT_METRICS,
    "vad": VAD_METRICS,
    "anomaly": ANOMALY_METRICS,
    "noise_suppression": NOISE_SUPPRESSION_METRICS,
}

_DEFAULT_METRICS = {
    "classification": "macro_f1",
    "asr": "normalized_edit_similarity",
    "speaker_recognition": "macro_f1",
    "sound_event_detection": "event_f1",
    "vad": "frame_f1",
    "anomaly": "auroc",
    "noise_suppression": "snr_improvement",
}

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "precision": "Precision",
    "recall": "Recall",
    "wer": "Word error rate",
    "cer": "Character error rate",
    "exact_match_accuracy": "Exact match accuracy",
    "normalized_edit_similarity": "Normalized edit similarity",
    "equal_error_rate": "Equal error rate",
    "auroc": "AUROC",
    "verification_accuracy": "Verification accuracy",
    "event_f1": "Event-based F1",
    "segment_f1": "Segment-based F1",
    "error_rate": "Error rate",
    "frame_f1": "Frame-level F1",
    "false_alarm_rate": "False alarm rate",
    "miss_rate": "Miss rate",
    "auprc": "AUPRC",
    "f1": "F1",
    "proxy_score": "Proxy score",
    "score_separation": "Score separation",
    "reconstruction_consistency": "Reconstruction consistency",
    "stability": "Stability",
    "si_sdr_improvement": "SI-SDR improvement",
    "snr_improvement": "SNR improvement",
    "stoi": "STOI",
    "pesq": "PESQ",
    "spectral_distance": "Spectral distance",
}


def normalize_task_type(task_type: str) -> str:
    return (task_type or "").strip().lower()


def task_family(task_type: str) -> str:
    return _TASK_FAMILIES.get(normalize_task_type(task_type), "other")


def valid_metrics_for_task(task_type: str) -> list:
    return list(_TASK_METRICS.get(normalize_task_type(task_type), []))


def default_metric_for_task(task_type: str) -> str:
    return _DEFAULT_METRICS.get(normalize_task_type(task_type), "")


def metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def is_deprecated_task(task_type: str) -> bool:
    return normalize_task_type(task_type) in DEPRECATED_TASK_TYPES


def task_display_name(task_type: str, label_mode: str = "") -> str:
    return TASK_DISPLAY_NAMES.get(normalize_task_type(task_type), task_type or "")


@dataclass
class AudioConfig:
    data_path: Path
    metric: str = ""
    task_type: str = "classification"
    domain: str = ""
    constraints: str = ""
    notes: str = ""
    modality: str = "Audio"
    input_format: str = ""
    input_format_key: str = ""
    metadata_path: str = ""
    record_path: str = ""
    audio_format: str = ""
    channel_layout: str = ""
    sample_rate: str = ""
    field_overrides: Dict[str, str] = field(default_factory=dict)

    @property
    def supervision(self) -> str:
        return "unsupervised" if normalize_task_type(self.task_type) == "anomaly" else "supervised"

    @property
    def task_family(self) -> str:
        return task_family(self.task_type)

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    def task_context(self) -> dict:
        task_type = normalize_task_type(self.task_type)
        return {
            "task_type": task_type,
            "task_family": task_family(task_type),
            "task_name": task_display_name(task_type),
            "domain": self.domain,
            "constraints": self.constraints,
            "active_constraints": self.active_constraints,
            "notes": self.notes,
            "modality": self.modality,
            "input_format": self.input_format,
            "input_format_key": self.input_format_key,
            "metadata_path": self.metadata_path,
            "record_path": self.record_path,
            "audio_format": self.audio_format,
            "channel_layout": self.channel_layout,
            "sample_rate": self.sample_rate,
            "supervision": self.supervision,
            "field_overrides": dict(self.field_overrides or {}),
        }
