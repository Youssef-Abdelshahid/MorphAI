import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MEMORY_DIR = Path("memory") / "image"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"
_SCORE_SYSTEM = "normalized_v2"

MIN_RUNS_TO_USE = 5
MIN_RUNS_FULL_WEIGHT = 20
MAX_META_WEIGHT = 0.25

_TASK_TYPES = [
    "classification",
    "multilabel",
    "detection",
    "semantic_segmentation",
    "instance_segmentation",
    "keypoint",
    "retrieval",
    "anomaly",
    "ocr",
    "generation",
    "depth",
    "other",
]
_TASK_TYPE_MAP = {task: idx for idx, task in enumerate(_TASK_TYPES)}

_COLOR_MODE_MAP = {"rgb": 0, "grayscale": 1}
_NORM_MAP = {"none": 0, "standard": 1, "minmax": 2}
_ROTATION_MAP = {"none": 0, "light": 1, "moderate": 2}
_IMBALANCE_MAP = {"none": 0, "oversample": 1}


def _encode_task_type(task_type: str) -> float:
    idx = _TASK_TYPE_MAP.get((task_type or "other").lower().strip(), len(_TASK_TYPES) - 1)
    return idx / max(len(_TASK_TYPES) - 1, 1)


def _encode_domain(domain: str) -> float:
    if not domain:
        return 0.0
    return (hash(domain.lower().strip()) % 1_000) / 1_000.0


def _profile_features(profile_summary: dict) -> List[float]:
    n_images = max(int(profile_summary.get("n_images", 1)), 1)
    n_classes = max(int(profile_summary.get("n_classes", 1)), 1)
    return [
        min(math.log10(n_images) / 6.0, 1.0),
        min(n_classes / 100.0, 1.0),
        min(float(profile_summary.get("imbalance_ratio", 1.0)), 20.0) / 20.0,
        float(profile_summary.get("avg_brightness", 0.5)),
        float(profile_summary.get("brightness_std", 0.0)),
        float(profile_summary.get("avg_contrast", 0.15)),
        float(profile_summary.get("contrast_std", 0.0)),
        float(profile_summary.get("grayscale_ratio", 0.0)),
        float(bool(profile_summary.get("has_varied_sizes", False))),
        float(bool(profile_summary.get("has_low_contrast", False))),
        float(bool(profile_summary.get("is_imbalanced", False))),
        float(bool(profile_summary.get("is_highly_imbalanced", False))),
    ]


def _pipeline_features(pipeline_dict: dict) -> List[float]:
    resize = int(pipeline_dict.get("resize", 64))
    return [
        min(resize / 256.0, 1.0),
        _COLOR_MODE_MAP.get(pipeline_dict.get("color_mode", "rgb"), 0),
        _NORM_MAP.get(pipeline_dict.get("normalization", "standard"), 1) / 2.0,
        float(bool(pipeline_dict.get("histogram_eq", False))),
        float(bool(pipeline_dict.get("denoise", False))),
        float(bool(pipeline_dict.get("sharpen", False))),
        float(bool(pipeline_dict.get("augment_h_flip", False))),
        float(bool(pipeline_dict.get("augment_v_flip", False))),
        _ROTATION_MAP.get(pipeline_dict.get("augment_rotation", "none"), 0) / 2.0,
        float(bool(pipeline_dict.get("augment_color_jitter", False))),
        _IMBALANCE_MAP.get(pipeline_dict.get("imbalance", "none"), 0),
    ]


def _build_feature_vector(task_context: dict, profile_summary: dict, pipeline_dict: dict) -> List[float]:
    return [
        _encode_task_type(task_context.get("task_type", "other")),
        _encode_domain(task_context.get("domain", "")),
    ] + _profile_features(profile_summary) + _pipeline_features(pipeline_dict)


class ImageMetaLearner:
    def __init__(self) -> None:
        self._model: Any = None
        self._n_train: int = 0

    def load(self) -> None:
        if META_LEARNER_FILE.exists():
            try:
                with open(META_LEARNER_FILE, "rb") as fh:
                    data = pickle.load(fh)
                model = data.get("model")
                n_train = int(data.get("n_train", 0))
                stored_feature_size = int(data.get("feature_size", 0))
                current_feature_size = len(_build_feature_vector({}, {}, {}))
                score_system = data.get("score_system")
                if stored_feature_size != current_feature_size or score_system != _SCORE_SYSTEM:
                    self._model = None
                    self._n_train = 0
                else:
                    self._model = model
                    self._n_train = n_train
            except Exception:
                self._model = None
                self._n_train = 0

    def save(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(META_LEARNER_FILE, "wb") as fh:
            pickle.dump(
                {
                    "model": self._model,
                    "n_train": self._n_train,
                    "feature_size": len(_build_feature_vector({}, {}, {})),
                    "score_system": _SCORE_SYSTEM,
                },
                fh,
            )

    @property
    def is_mature(self) -> bool:
        return self._n_train >= MIN_RUNS_TO_USE

    @property
    def weight(self) -> float:
        if self._n_train < MIN_RUNS_TO_USE:
            return 0.0
        ratio = min((self._n_train - MIN_RUNS_TO_USE) / max(MIN_RUNS_FULL_WEIGHT - MIN_RUNS_TO_USE, 1), 1.0)
        return MAX_META_WEIGHT * ratio

    @property
    def n_train(self) -> int:
        return self._n_train

    def train_from_memory(self, runs: List[Dict[str, Any]]) -> int:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return 0
        X: List[List[float]] = []
        y: List[float] = []
        for run in runs:
            task_context = run.get("task_context", {})
            profile_summary = run.get("profile_summary", {})
            all_pipelines = run.get("all_pipelines_tested", [])
            if all_pipelines:
                for entry in all_pipelines:
                    pipeline_dict = entry.get("pipeline_config") or entry.get("pipeline")
                    score = entry.get("normalized_score", entry.get("final_score"))
                    if pipeline_dict and score is not None:
                        X.append(_build_feature_vector(task_context, profile_summary, pipeline_dict))
                        y.append(float(score))
            else:
                best_pipeline = run.get("best_pipeline")
                best_score = run.get("best_score")
                if best_pipeline and best_score is not None:
                    X.append(_build_feature_vector(task_context, profile_summary, best_pipeline))
                    y.append(float(best_score))
        if len(X) < 3:
            return 0
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        self._model = model
        self._n_train = len(X)
        return len(X)

    def predict_score(self, task_context: dict, profile_summary: dict, pipeline_dict: dict) -> Optional[float]:
        if not self.is_mature or self._model is None:
            return None
        try:
            fv = _build_feature_vector(task_context, profile_summary, pipeline_dict)
            pred = float(self._model.predict(np.asarray([fv], dtype=float))[0])
            return max(0.0, min(1.0, pred))
        except Exception:
            return None

    def rank_candidates(
        self,
        candidates: list,
        task_context: dict,
        profile_summary: dict,
        existing_scores: Optional[List[float]] = None,
    ) -> Tuple[list, List[str]]:
        if not self.is_mature or self._model is None:
            return candidates, []
        weight = self.weight
        if weight <= 0.0 or len(candidates) <= 1:
            return candidates, []
        n = len(candidates)
        if existing_scores and len(existing_scores) == n:
            mn, mx = min(existing_scores), max(existing_scores)
            span = (mx - mn) if mx > mn else 1.0
            heuristic_scores = [(score - mn) / span for score in existing_scores]
        else:
            heuristic_scores = [(n - idx) / n for idx in range(n)]
        combined: List[Tuple[float, int, Any]] = []
        for idx, spec in enumerate(candidates):
            meta_score = self.predict_score(task_context, profile_summary, spec.to_dict())
            if meta_score is None:
                meta_score = heuristic_scores[idx]
            combined_score = (1.0 - weight) * heuristic_scores[idx] + weight * meta_score
            combined.append((combined_score, idx, spec))
        combined.sort(key=lambda item: -item[0])
        reordered = [spec for _, _, spec in combined]
        msgs = [f"Meta-learner (advisory, weight={weight:.2f}, samples={self._n_train}): reordered {n} candidate(s)."]
        return reordered, msgs

    def status_summary(self) -> dict:
        return {
            "is_mature": self.is_mature,
            "n_train": self._n_train,
            "weight": round(self.weight, 3),
            "min_to_use": MIN_RUNS_TO_USE,
            "min_full_wt": MIN_RUNS_FULL_WEIGHT,
            "max_weight": MAX_META_WEIGHT,
        }
