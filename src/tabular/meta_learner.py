import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MEMORY_DIR        = Path("memory") / "tabular"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"

MIN_RUNS_TO_USE      = 5
MIN_RUNS_FULL_WEIGHT  = 20
MAX_META_WEIGHT      = 0.25

_TASK_TYPES    = ["classification", "binary", "multiclass", "regression", "other"]
_TASK_TYPE_MAP = {t: i for i, t in enumerate(_TASK_TYPES)}

_NUM_IMP_MAP   = {"mean": 0, "median": 1, "knn": 2}
_SCALER_MAP    = {"none": 0, "standard": 1, "minmax": 2, "robust": 3}
_ENCODER_MAP   = {"onehot": 0, "ordinal": 1}
_IMBALANCE_MAP = {"none": 0, "oversample": 1, "smote": 2}

_FE_BUDGET_MAP = {
    "minimal":  0.0,
    "light":    0.33,
    "moderate": 0.67,
    "heavy":    1.0,
}

_DATA_QUALITY_MAP = {
    "clean":       0.0,
    "mostly_clean": 0.25,
    "mixed":       0.5,
    "unknown":     0.75,
    "noisy":       1.0,
}


def _encode_task_type(task_type: str) -> float:
    idx = _TASK_TYPE_MAP.get((task_type or "other").lower().strip(), 4)
    return idx / 4.0


def _encode_domain(domain: str) -> float:
    if not domain:
        return 0.0
    return (hash(domain.lower().strip()) % 1_000) / 1_000.0


def _encode_fe_budget(fe_budget_norm: str) -> float:
    return _FE_BUDGET_MAP.get((fe_budget_norm or "moderate").lower(), 0.67)


def _encode_data_quality(data_quality_norm: str) -> float:
    return _DATA_QUALITY_MAP.get((data_quality_norm or "unknown").lower(), 0.75)


def _encode_supervision(supervision: str) -> float:
    return 1.0 if supervision == "supervised" else 0.0


def _encode_task_type_bin(task_type: str) -> float:
    t = (task_type or "").lower()
    if t in ("binary", "classification"):
        return 0.0
    if t == "multiclass":
        return 0.5
    if t == "regression":
        return 1.0
    return 0.25


def _profile_features(profile_summary: dict) -> List[float]:
    n_rows = max(int(profile_summary.get("n_rows", 1)), 1)
    n_cols = max(int(profile_summary.get("n_cols", 1)), 1)
    return [
        min(math.log10(n_rows) / 6.0, 1.0),
        min(n_cols / 100.0, 1.0),
        float(profile_summary.get("missing_ratio", 0.0)),
        min(float(profile_summary.get("imbalance_ratio", 1.0)), 20.0) / 20.0,
        float(profile_summary.get("num_col_ratio", 1.0)),
        float(profile_summary.get("cat_col_ratio", 0.0)),
        float(bool(profile_summary.get("has_outliers", False))),
        float(bool(profile_summary.get("has_high_skew", False))),
        float(bool(profile_summary.get("is_imbalanced", False))),
        float(bool(profile_summary.get("is_highly_imbalanced", False))),
    ]


def _pipeline_features(pipeline_dict: dict) -> List[float]:
    return [
        _NUM_IMP_MAP.get(pipeline_dict.get("num_imputer", "mean"), 0) / 2.0,
        _SCALER_MAP.get(pipeline_dict.get("scaler", "standard"), 1) / 3.0,
        _ENCODER_MAP.get(pipeline_dict.get("encoder", "onehot"), 0),
        _IMBALANCE_MAP.get(pipeline_dict.get("imbalance", "none"), 0) / 2.0,
        float(bool(pipeline_dict.get("power_transform", False))),
        float(bool(pipeline_dict.get("outlier_clip", False))),
        float(bool(pipeline_dict.get("remove_duplicates", False))),
        float(bool(pipeline_dict.get("remove_low_variance", False))),
        float(bool(pipeline_dict.get("drop_high_missing_cols", False))),
    ]


def _build_feature_vector(
    task_context: dict,
    profile_summary: dict,
    pipeline_dict: dict,
) -> List[float]:
    task_feats = [
        _encode_task_type(task_context.get("task_type", "other")),
        _encode_domain(task_context.get("domain", "")),
        _encode_fe_budget(task_context.get("fe_budget_norm", "moderate")),
        _encode_data_quality(task_context.get("data_quality_norm", "unknown")),
        _encode_supervision(task_context.get("supervision", "supervised")),
        _encode_task_type_bin(task_context.get("task_type", "")),
    ]
    return task_feats + _profile_features(profile_summary) + _pipeline_features(pipeline_dict)


class MetaLearner:

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
                stored_feature_size = int(data.get("feature_size", 21))
                current_feature_size = len(_build_feature_vector({}, {}, {}))
                if stored_feature_size != current_feature_size:
                    self._model   = None
                    self._n_train = 0
                else:
                    self._model   = model
                    self._n_train = n_train
            except Exception:
                self._model   = None
                self._n_train = 0

    def save(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        current_feature_size = len(_build_feature_vector({}, {}, {}))
        with open(META_LEARNER_FILE, "wb") as fh:
            pickle.dump({
                "model":        self._model,
                "n_train":      self._n_train,
                "feature_size": current_feature_size,
            }, fh)

    @property
    def is_mature(self) -> bool:
        return self._n_train >= MIN_RUNS_TO_USE

    @property
    def weight(self) -> float:
        if self._n_train < MIN_RUNS_TO_USE:
            return 0.0
        ratio = min(
            (self._n_train - MIN_RUNS_TO_USE)
            / max(MIN_RUNS_FULL_WEIGHT - MIN_RUNS_TO_USE, 1),
            1.0,
        )
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
        y: List[float]       = []

        for run in runs:
            task_context    = run.get("task_context", {})
            profile_summary = run.get("profile_summary", {})
            metric          = run.get("metric_priority", "f1")
            all_pipelines   = run.get("all_pipelines_tested", [])

            if all_pipelines:
                for entry in all_pipelines:
                    pipe_dict = entry.get("pipeline_config") or entry.get("pipeline")
                    score     = entry.get("metrics", {}).get(metric)
                    if pipe_dict and score is not None:
                        fv = _build_feature_vector(task_context, profile_summary, pipe_dict)
                        X.append(fv)
                        y.append(float(score))
            else:
                bp    = run.get("best_pipeline")
                bs    = run.get("best_score")
                if bp and bs is not None:
                    fv = _build_feature_vector(task_context, profile_summary, bp)
                    X.append(fv)
                    y.append(float(bs))

        if len(X) < 3:
            return 0

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        model = __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_arr, y_arr)
        self._model   = model
        self._n_train = len(X)
        return len(X)

    def predict_score(
        self,
        task_context: dict,
        profile_summary: dict,
        pipeline_dict: dict,
    ) -> Optional[float]:
        if not self.is_mature or self._model is None:
            return None
        try:
            fv   = _build_feature_vector(task_context, profile_summary, pipeline_dict)
            pred = float(self._model.predict(np.array([fv]))[0])
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

        w = self.weight
        if w <= 0.0 or len(candidates) <= 1:
            return candidates, []

        n = len(candidates)

        if existing_scores and len(existing_scores) == n:
            mn, mx = min(existing_scores), max(existing_scores)
            span   = (mx - mn) if mx > mn else 1.0
            h_scores = [(s - mn) / span for s in existing_scores]
        else:
            h_scores = [(n - i) / n for i in range(n)]

        combined: List[Tuple[float, int, Any]] = []
        for i, spec in enumerate(candidates):
            meta_score = self.predict_score(task_context, profile_summary, spec.to_dict())
            if meta_score is None:
                meta_score = h_scores[i]
            combined_score = (1.0 - w) * h_scores[i] + w * meta_score
            combined.append((combined_score, i, spec))

        combined.sort(key=lambda x: -x[0])
        reordered = [spec for _, _, spec in combined]

        msgs = [
            f"Meta-learner (advisory, weight={w:.2f}, "
            f"samples={self._n_train}): reordered {n} candidate(s)."
        ]
        return reordered, msgs

    def status_summary(self) -> dict:
        return {
            "is_mature":   self.is_mature,
            "n_train":     self._n_train,
            "weight":      round(self.weight, 3),
            "min_to_use":  MIN_RUNS_TO_USE,
            "min_full_wt": MIN_RUNS_FULL_WEIGHT,
            "max_weight":  MAX_META_WEIGHT,
        }
