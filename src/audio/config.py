from dataclasses import dataclass
from pathlib import Path

AUDIO_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]
ASR_METRICS = ["wer", "cer", "exact_match_accuracy", "normalized_edit_similarity"]
SPEAKER_IDENTIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1"]
SPEAKER_VERIFICATION_METRICS = ["equal_error_rate", "auroc", "verification_accuracy"]
DIARIZATION_METRICS = ["diarization_error_rate", "speaker_confusion", "missed_speech", "false_alarm_speech"]
SOUND_EVENT_METRICS = ["event_f1", "segment_f1", "precision", "recall", "error_rate"]
VAD_METRICS = ["frame_f1", "precision", "recall", "false_alarm_rate", "miss_rate"]
ANOMALY_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "proxy_score", "score_separation", "reconstruction_consistency", "stability"]
NOISE_SUPPRESSION_METRICS = ["si_sdr_improvement", "snr_improvement", "stoi", "pesq", "spectral_distance", "proxy_score"]

_AUD_TASK_BACKEND = {
    "Audio classification": "classification",
    "Speech recognition (ASR)": "asr",
    "Speaker recognition": "speaker_recognition",
    "Speaker diarization": "speaker_diarization",
    "Sound event detection": "sound_event_detection",
    "Voice activity detection": "vad",
    "Audio anomaly detection": "anomaly",
    "Noise suppression": "noise_suppression",
}

VALID_TASK_TYPES = list(_AUD_TASK_BACKEND.values())
SUPPORTED_TASK_TYPES = set(VALID_TASK_TYPES)

_TASK_FAMILIES = {
    "classification": "classification",
    "asr": "speech",
    "speaker_recognition": "speaker",
    "speaker_diarization": "speaker",
    "sound_event_detection": "event",
    "vad": "speech",
    "anomaly": "anomaly",
    "noise_suppression": "enhancement",
}

_TASK_METRICS = {
    "classification": AUDIO_CLASSIFICATION_METRICS,
    "asr": ASR_METRICS,
    "speaker_recognition": sorted(set(SPEAKER_IDENTIFICATION_METRICS + SPEAKER_VERIFICATION_METRICS)),
    "speaker_diarization": DIARIZATION_METRICS,
    "sound_event_detection": SOUND_EVENT_METRICS,
    "vad": VAD_METRICS,
    "anomaly": ANOMALY_METRICS,
    "noise_suppression": NOISE_SUPPRESSION_METRICS,
}

_DEFAULT_METRICS = {
    "classification": "macro_f1",
    "asr": "normalized_edit_similarity",
    "speaker_recognition": "macro_f1",
    "speaker_diarization": "diarization_error_rate",
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
    "diarization_error_rate": "Diarization error rate",
    "speaker_confusion": "Speaker confusion",
    "missed_speech": "Missed speech",
    "false_alarm_speech": "False alarm speech",
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
    audio_format: str = ""
    channel_layout: str = ""
    sample_rate: str = ""

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
            "domain": self.domain,
            "constraints": self.constraints,
            "active_constraints": self.active_constraints,
            "notes": self.notes,
            "modality": self.modality,
            "input_format": self.input_format,
            "audio_format": self.audio_format,
            "channel_layout": self.channel_layout,
            "sample_rate": self.sample_rate,
            "supervision": self.supervision,
        }
