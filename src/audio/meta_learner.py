import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MEMORY_DIR = Path("memory") / "audio"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"
_SCORE_SYSTEM = "normalized_audio_v2"

MIN_RUNS_TO_USE = 5
MIN_RUNS_FULL_WEIGHT = 20
MAX_META_WEIGHT = 0.25

_TASKS = ["classification", "asr", "speaker_recognition", "sound_event_detection", "vad", "anomaly", "noise_suppression", "other"]
_TASK_MAP = {name: i for i, name in enumerate(_TASKS)}
_FEAT_MAP = {"mfcc": 0, "mel_spectrogram": 1, "log_mel_spectrogram": 2, "raw_waveform": 3}
_NORM_MAP = {"none": 0, "rms": 1, "peak": 2}
_NOISE_MAP = {"none": 0, "highpass": 1, "spectral_gate": 2}
_FORMAT_MAP = {"zip_folder": 0, "metadata_csv": 1, "metadata_json": 2, "audio_folder_zip": 0}


def _encode_task(task_type: str) -> float:
    return _TASK_MAP.get((task_type or "other").lower().strip(), len(_TASKS) - 1) / max(len(_TASKS) - 1, 1)


def _profile_features(summary: dict) -> List[float]:
    return [
        min(math.log10(max(int(summary.get("n_audio_files", 1)), 1)) / 6.0, 1.0),
        min(float(summary.get("n_classes", 0)), 100.0) / 100.0,
        min(float(summary.get("avg_duration_sec", 0.0)), 120.0) / 120.0,
        min(float(summary.get("duration_std_sec", 0.0)), 120.0) / 120.0,
        min(float(summary.get("imbalance_ratio", 1.0)), 20.0) / 20.0,
        float(summary.get("silence_ratio", 0.0)),
        float(summary.get("clipping_ratio", 0.0)),
        float(summary.get("corruption_ratio", 0.0)),
        float(summary.get("estimated_noise_ratio", 0.0)),
        min(len(summary.get("sample_rate_distribution", {})), 10) / 10.0,
        min(len(summary.get("channel_count_distribution", {})), 10) / 10.0,
        _FORMAT_MAP.get((summary.get("input_format") or "zip_folder"), 0) / max(len(_FORMAT_MAP) - 1, 1),
        float(bool(summary.get("has_class_labels"))),
        float(bool(summary.get("has_transcripts"))),
        float(bool(summary.get("has_speaker_labels"))),
        float(bool(summary.get("has_speaker_pairs"))),
        float(bool(summary.get("has_temporal_segments"))),
        float(bool(summary.get("has_anomaly_labels"))),
        float(bool(summary.get("has_noisy_clean_pairs"))),
    ]


def _pipeline_features(pipeline: dict) -> List[float]:
    sr = int(pipeline.get("target_sample_rate", 16000) or 0)
    return [
        min(sr / 48000.0, 1.0),
        float(bool(pipeline.get("mono", True))),
        float(bool(pipeline.get("trim_silence", False))),
        _NORM_MAP.get(pipeline.get("loudness_normalization", "rms"), 1) / 2.0,
        _NOISE_MAP.get(pipeline.get("noise_filter", "none"), 0) / 2.0,
        float(pipeline.get("clipping_handling", "none") != "none"),
        _FEAT_MAP.get(pipeline.get("feature_representation", "mfcc"), 0) / 3.0,
        float(pipeline.get("augmentation", "none") != "none"),
        float(pipeline.get("imbalance", "none") != "none"),
    ]


def _vector(task_context: dict, profile_summary: dict, pipeline: dict) -> List[float]:
    return [_encode_task(task_context.get("task_type", "other"))] + _profile_features(profile_summary) + _pipeline_features(pipeline)


class AudioMetaLearner:
    def __init__(self) -> None:
        self._model: Any = None
        self._n_train = 0

    def load(self) -> None:
        if META_LEARNER_FILE.exists():
            try:
                with open(META_LEARNER_FILE, "rb") as fh:
                    data = pickle.load(fh)
                if data.get("feature_size") == len(_vector({}, {}, {})) and data.get("score_system") == _SCORE_SYSTEM:
                    self._model = data.get("model")
                    self._n_train = int(data.get("n_train", 0))
                else:
                    self._model = None
                    self._n_train = 0
            except Exception:
                self._model = None
                self._n_train = 0

    def save(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(META_LEARNER_FILE, "wb") as fh:
            pickle.dump({"model": self._model, "n_train": self._n_train, "feature_size": len(_vector({}, {}, {})), "score_system": _SCORE_SYSTEM}, fh)

    @property
    def is_mature(self) -> bool:
        return self._n_train >= MIN_RUNS_TO_USE

    @property
    def weight(self) -> float:
        if self._n_train < MIN_RUNS_TO_USE:
            return 0.0
        ratio = min((self._n_train - MIN_RUNS_TO_USE) / max(MIN_RUNS_FULL_WEIGHT - MIN_RUNS_TO_USE, 1), 1.0)
        return MAX_META_WEIGHT * ratio

    def train_from_memory(self, runs: List[Dict[str, Any]]) -> int:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return 0
        X = []
        y = []
        for run in runs:
            tc = run.get("task_context", {})
            ps = run.get("profile_summary", {})
            for entry in run.get("all_pipelines_tested", []):
                pipeline = entry.get("pipeline_config") or entry.get("pipeline")
                score = entry.get("normalized_score")
                if pipeline and score is not None:
                    X.append(_vector(tc, ps, pipeline))
                    y.append(float(score))
        if len(X) < 3:
            return 0
        model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=1)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        self._model = model
        self._n_train = len(X)
        return len(X)

    def predict_score(self, task_context: dict, profile_summary: dict, pipeline_dict: dict) -> Optional[float]:
        if not self.is_mature or self._model is None:
            return None
        try:
            return max(0.0, min(1.0, float(self._model.predict(np.asarray([_vector(task_context, profile_summary, pipeline_dict)], dtype=float))[0])))
        except Exception:
            return None

    def rank_candidates(self, candidates: list, task_context: dict, profile_summary: dict) -> Tuple[list, List[str]]:
        if not self.is_mature or self._model is None or len(candidates) <= 1:
            return candidates, []
        n = len(candidates)
        combined = []
        for idx, spec in enumerate(candidates):
            heuristic = (n - idx) / n
            meta = self.predict_score(task_context, profile_summary, spec.to_dict())
            if meta is None:
                meta = heuristic
            combined.append(((1.0 - self.weight) * heuristic + self.weight * meta, idx, spec))
        combined.sort(key=lambda x: -x[0])
        return [spec for _, _, spec in combined], [f"Meta-learner (advisory, weight={self.weight:.2f}, samples={self._n_train}): reordered {n} audio candidate(s)."]

    def status_summary(self) -> dict:
        return {"is_mature": self.is_mature, "n_train": self._n_train, "weight": round(self.weight, 3), "min_to_use": MIN_RUNS_TO_USE, "min_full_wt": MIN_RUNS_FULL_WEIGHT, "max_weight": MAX_META_WEIGHT}
