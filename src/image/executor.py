import csv
import json
import math
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from imblearn.over_sampling import RandomOverSampler
from PIL import Image, ImageFilter
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import OneClassSVM

from .config import default_metric_for_task, metric_label, normalize_task_type
from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

_DEVICE = torch.device("cpu")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_ANNOTATION_EXTENSIONS = {".json", ".txt", ".xml", ".csv", ".png", ".npy", ".npz"}
_MODEL_NAMES: List[str] = ["mobilenet_v3_small", "shufflenet_v2_x0_5", "squeezenet1_1"]
_CNN_CACHE: Dict[str, nn.Module] = {}


def _build_cnn_extractor(name: str) -> nn.Module:
    if name == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    elif name == "shufflenet_v2_x0_5":
        model = tv_models.shufflenet_v2_x0_5(weights=tv_models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        extractor = nn.Sequential(
            model.conv1, model.maxpool, model.stage2, model.stage3, model.stage4,
            model.conv5, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
    elif name == "squeezenet1_1":
        model = tv_models.squeezenet1_1(weights=tv_models.SqueezeNet1_1_Weights.DEFAULT)
        extractor = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(name)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)
    return extractor


def _get_extractor(name: str) -> nn.Module:
    if name not in _CNN_CACHE:
        _CNN_CACHE[name] = _build_cnn_extractor(name)
    return _CNN_CACHE[name]


def _clamp_01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else 0.0


def _safe_std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(finite)) if finite else 0.0


def _safe_scale(value: float, fallback: float = 1.0) -> float:
    if not np.isfinite(value) or abs(value) <= 1e-12:
        return fallback
    return float(abs(value))


def _flat_to_tensor(X_flat: np.ndarray, img_size: int, channels: int) -> torch.Tensor:
    n = X_flat.shape[0]
    if channels == 1:
        imgs = X_flat.reshape(n, img_size, img_size)
        tensor = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)
        tensor = tensor.expand(-1, 3, -1, -1)
    else:
        imgs = X_flat.reshape(n, img_size, img_size, 3).transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(imgs.astype(np.float32))
    return tensor


def _extract_cnn_features(X_flat: np.ndarray, extractor: nn.Module, img_size: int, channels: int, batch_size: int = 64) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, X_flat.shape[0], batch_size):
            batch = _flat_to_tensor(X_flat[start:start + batch_size], img_size, channels)
            parts.append(extractor(batch).numpy())
    return np.concatenate(parts, axis=0)


def _preprocess_single_image(img: Image.Image, spec: ImagePipelineSpec) -> np.ndarray:
    if spec.color_mode == "grayscale":
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    if spec.resize > 0:
        img = img.resize((spec.resize, spec.resize), Image.LANCZOS)
    if spec.histogram_eq:
        try:
            from PIL import ImageOps
            img = ImageOps.equalize(img)
        except Exception:
            pass
    if spec.denoise:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    if spec.sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    arr = np.asarray(img, dtype=np.float32)
    if spec.normalization == "standard":
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / std if std > 0 else arr - mean
    elif spec.normalization == "minmax":
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        arr = (arr - mn) / rng if rng > 0 else arr * 0.0
    return arr.flatten()


def _load_original_image_array(path_str: str, resize: int, color_mode: str) -> Optional[np.ndarray]:
    try:
        image = Image.open(path_str)
        image.load()
        mode = "L" if color_mode == "grayscale" else "RGB"
        image = image.convert(mode)
        image = image.resize((resize, resize), Image.LANCZOS)
        return np.asarray(image, dtype=np.float32)
    except Exception:
        return None


def _parse_multilabel_value(value: str) -> List[str]:
    if not value:
        return []
    for sep in ["|", ";", ",", "+"]:
        if sep in value:
            return [part.strip() for part in value.split(sep) if part.strip()]
    return [value.strip()]


def _load_images(profile: ImageProfile, spec: ImagePipelineSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    processed = []
    originals = []
    kept_paths: List[str] = []
    kept_labels: List[str] = []
    resize = spec.resize if spec.resize > 0 else max(32, int(round(profile.avg_height or 64)))
    for path_str, label in zip(profile.image_paths, profile.image_labels):
        try:
            image = Image.open(path_str)
            image.load()
            processed.append(_preprocess_single_image(image, spec))
            original = _load_original_image_array(path_str, resize, spec.color_mode)
            if original is None:
                processed.pop()
                continue
            originals.append(original.flatten())
            kept_paths.append(path_str)
            kept_labels.append(label)
        except Exception:
            continue
    if not processed:
        return np.array([]), np.array([]), np.array([]), [], []
    X_proc = np.asarray(processed, dtype=np.float32)
    X_orig = np.asarray(originals, dtype=np.float32)
    labels = np.asarray(kept_labels, dtype=object)
    return X_proc, X_orig, labels, kept_paths, kept_labels


def _resolve_metric(task_type: str, requested: str, available: Sequence[str], fallback: Optional[str] = None) -> Tuple[str, Optional[str]]:
    requested = (requested or "").strip().lower()
    available_set = set(available)
    if requested and requested in available_set:
        return requested, None
    default = fallback or default_metric_for_task(task_type)
    if default in available_set:
        if requested and requested not in available_set:
            return default, f"Requested metric '{requested}' was unavailable; used {metric_label(default)} instead."
        return default, None
    if available:
        chosen = list(available)[0]
        if requested and requested != chosen:
            return chosen, f"Requested metric '{requested}' was unavailable; used {metric_label(chosen)} instead."
        return chosen, None
    return requested or default or "score", "No valid metric was available."


def _make_result(
    spec: ImagePipelineSpec,
    task_type: str,
    metric_priority: str,
    selected_metric: str,
    raw_metrics: Dict[str, float],
    normalized_metrics: Dict[str, float],
    model_scores: Dict[str, Dict[str, float]],
    evaluator_details: Dict[str, Any],
    evaluation_mode: str,
    evaluation_summary: str,
    elapsed_sec: float,
    metrics_std: Optional[Dict[str, float]] = None,
    normalized_metrics_std: Optional[Dict[str, float]] = None,
    n_splits: int = 0,
    n_models: int = 0,
    success: bool = True,
    reason: str = "",
) -> Dict[str, Any]:
    final_score = _clamp_01(normalized_metrics.get(selected_metric, 0.0))
    final_score_std = float((normalized_metrics_std or {}).get(f"{selected_metric}_std", 0.0))
    return {
        "spec": spec,
        "task_type": task_type,
        "metric_priority": metric_priority,
        "selected_metric": selected_metric,
        "metrics": raw_metrics,
        "raw_metrics": raw_metrics,
        "metrics_std": metrics_std or {},
        "normalized_metrics": normalized_metrics,
        "normalized_metrics_std": normalized_metrics_std or {},
        "final_score": final_score,
        "normalized_score": final_score,
        "final_score_std": final_score_std,
        "model_scores": model_scores,
        "per_model_metrics": model_scores,
        "evaluator_details": evaluator_details,
        "evaluation_mode": evaluation_mode,
        "evaluation_summary": evaluation_summary,
        "n_splits": n_splits,
        "n_models": n_models,
        "elapsed_sec": round(elapsed_sec, 3),
        "success": success,
        "reason": reason,
    }


def _failed_result(spec: ImagePipelineSpec, task_type: str, metric_priority: str, reason: str, elapsed_sec: float, evaluation_mode: str = "failed") -> Dict[str, Any]:
    fallback_metric = default_metric_for_task(task_type) or metric_priority or "score"
    return _make_result(
        spec,
        task_type,
        metric_priority,
        fallback_metric,
        {fallback_metric: 0.0},
        {fallback_metric: 0.0},
        {},
        {"failure_reason": reason},
        evaluation_mode,
        reason,
        elapsed_sec,
        n_splits=0,
        n_models=0,
        success=False,
        reason=reason,
    )


def _normalize_lower_better(value: float, scale: float) -> float:
    return _clamp_01(1.0 / (1.0 + max(value, 0.0) / _safe_scale(scale)))


def _cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    denom = np.clip(a_norm * b_norm, 1e-12, None)
    return np.sum(a * b, axis=1, keepdims=True) / denom


def _aggregate_model_metrics(per_model_raw: Dict[str, Dict[str, float]], per_model_norm: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    metric_names = sorted({key for metrics in per_model_raw.values() for key in metrics})
    raw_metrics = {key: _safe_mean([metrics.get(key, np.nan) for metrics in per_model_raw.values()]) for key in metric_names}
    raw_std = {f"{key}_std": _safe_std([metrics.get(key, np.nan) for metrics in per_model_raw.values()]) for key in metric_names}
    norm_metrics = {key: _safe_mean([metrics.get(key, np.nan) for metrics in per_model_norm.values()]) for key in metric_names}
    norm_std = {f"{key}_std": _safe_std([metrics.get(key, np.nan) for metrics in per_model_norm.values()]) for key in metric_names}
    return raw_metrics, raw_std, norm_metrics, norm_std


def _parse_binary_anomaly_labels(labels: Sequence[str]) -> Optional[np.ndarray]:
    positives = {"anomaly", "defect", "fraud", "outlier", "abnormal", "positive"}
    negatives = {"normal", "good", "ok", "negative", "clean"}
    mapped: List[int] = []
    has_positive = False
    has_negative = False
    for label in labels:
        lower = str(label).lower()
        if any(token in lower for token in positives):
            mapped.append(1)
            has_positive = True
        elif any(token in lower for token in negatives):
            mapped.append(0)
            has_negative = True
        else:
            return None
    if has_positive and has_negative:
        return np.asarray(mapped, dtype=int)
    return None


def _build_embeddings(X_proc: np.ndarray, spec: ImagePipelineSpec) -> Dict[str, np.ndarray]:
    img_size = spec.resize if spec.resize > 0 else int(round(math.sqrt(X_proc.shape[1] / (1 if spec.color_mode == "grayscale" else 3))))
    channels = 1 if spec.color_mode == "grayscale" else 3
    embeddings: Dict[str, np.ndarray] = {}
    for model_name in _MODEL_NAMES:
        extractor = _get_extractor(model_name)
        embeddings[model_name] = _extract_cnn_features(X_proc, extractor, img_size, channels)
    return embeddings


def _evaluate_single_label_classification(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    X_proc, _, labels, _, kept_labels = _load_images(profile, spec)
    if len(kept_labels) < 10:
        raise ValueError("Single-label image classification needs at least 10 valid images after preprocessing.")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    if len(np.unique(y)) < 2:
        raise ValueError("Single-label image classification needs at least 2 classes.")
    min_class = int(np.bincount(y.astype(int)).min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError("Each class needs at least 2 samples for stratified image classification evaluation.")
    embeddings = _build_embeddings(X_proc, spec)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    for model_name, features in embeddings.items():
        fold_metrics: Dict[str, List[float]] = {}
        for train_idx, test_idx in splitter.split(features, y):
            X_train = features[train_idx]
            y_train = y[train_idx]
            X_test = features[test_idx]
            y_test = y[test_idx]
            if spec.imbalance == "oversample":
                sampler = RandomOverSampler(random_state=42)
                X_train, y_train = sampler.fit_resample(X_train, y_train)
            clf = LogisticRegression(solver="saga", max_iter=500, tol=1e-3, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            }
            for key, value in metrics.items():
                fold_metrics.setdefault(key, []).append(float(value))
        per_model_raw[model_name] = {key: _safe_mean(values) for key, values in fold_metrics.items()}
        per_model_norm[model_name] = {key: _clamp_01(value) for key, value in per_model_raw[model_name].items()}
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    selected_metric, metric_note = _resolve_metric("classification", metric_priority, raw_metrics.keys(), fallback="macro_f1")
    summary = (
        f"Single-label image classification evaluated across {len(per_model_raw)} feature extractors "
        f"with stratified {n_splits}-fold validation. Selected metric: {metric_label(selected_metric)} = "
        f"{raw_metrics.get(selected_metric, 0.0):.4f}. Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "classification", metric_priority, selected_metric, raw_metrics, norm_metrics,
        per_model_raw, {"metric_note": metric_note}, "supervised", summary, 0.0,
        metrics_std=raw_std, normalized_metrics_std=norm_std, n_splits=n_splits, n_models=len(per_model_raw)
    )


def _evaluate_multilabel_classification(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    X_proc, _, _, _, kept_labels = _load_images(profile, spec)
    if len(kept_labels) < 10:
        raise ValueError("Multi-label image classification needs at least 10 valid images after preprocessing.")
    parsed = [_parse_multilabel_value(str(label)) for label in kept_labels]
    if all(len(items) == 1 for items in parsed):
        note = "Folder labels looked single-label, so they were treated as single labels in a multi-label wrapper fallback."
    else:
        note = None
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(parsed)
    if y.shape[1] < 1:
        raise ValueError("No usable multi-label targets were found.")
    embeddings = _build_embeddings(X_proc, spec)
    n_splits = min(5, max(2, len(X_proc) // 10))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    for model_name, features in embeddings.items():
        fold_metrics: Dict[str, List[float]] = {}
        for train_idx, test_idx in splitter.split(features):
            X_train = features[train_idx]
            y_train = y[train_idx]
            X_test = features[test_idx]
            y_test = y[test_idx]
            clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics = {
                "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "hamming_loss": float(np.not_equal(y_test, y_pred).mean()),
                "subset_accuracy": accuracy_score(y_test, y_pred),
            }
            for key, value in metrics.items():
                fold_metrics.setdefault(key, []).append(float(value))
        per_model_raw[model_name] = {key: _safe_mean(values) for key, values in fold_metrics.items()}
        per_model_norm[model_name] = {
            "micro_f1": _clamp_01(per_model_raw[model_name].get("micro_f1", 0.0)),
            "macro_f1": _clamp_01(per_model_raw[model_name].get("macro_f1", 0.0)),
            "hamming_loss": _clamp_01(1.0 - per_model_raw[model_name].get("hamming_loss", 1.0)),
            "subset_accuracy": _clamp_01(per_model_raw[model_name].get("subset_accuracy", 0.0)),
        }
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    selected_metric, metric_note = _resolve_metric("multilabel", metric_priority, raw_metrics.keys(), fallback="micro_f1")
    summary = (
        f"Multi-label image classification evaluated with One-vs-Rest logistic baselines across {len(per_model_raw)} feature extractors. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    details = {"metric_note": metric_note, "label_note": note}
    if note:
        summary = f"{summary} {note}"
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "multilabel", metric_priority, selected_metric, raw_metrics, norm_metrics,
        per_model_raw, details, "supervised", summary, 0.0,
        metrics_std=raw_std, normalized_metrics_std=norm_std, n_splits=n_splits, n_models=len(per_model_raw)
    )


def _label_relevance_matrix(labels: Sequence[str]) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=object)
    return (labels_arr[:, None] == labels_arr[None, :]).astype(int)


def _evaluate_retrieval(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    X_proc, _, _, _, kept_labels = _load_images(profile, spec)
    if len(kept_labels) < 10:
        raise ValueError("Image retrieval needs at least 10 valid images after preprocessing.")
    embeddings = _build_embeddings(X_proc, spec)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    labels_arr = np.asarray(kept_labels, dtype=object)
    if len(np.unique(labels_arr)) < 2:
        raise ValueError("Image retrieval needs labels or at least 2 groups/classes to build a query-gallery structure.")
    for model_name, features in embeddings.items():
        sim = features @ features.T
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        sim = sim / np.clip(norms * norms.T, 1e-12, None)
        relevance = _label_relevance_matrix(kept_labels)
        recall_values = []
        precision_values = []
        ap_values = []
        rr_values = []
        k = min(5, len(features) - 1)
        for idx in range(len(features)):
            order = np.argsort(-sim[idx])
            order = order[order != idx]
            if len(order) == 0:
                continue
            rel = relevance[idx, order]
            topk = rel[:k]
            total_rel = max(int(np.sum(relevance[idx])) - 1, 1)
            recall_values.append(float(np.sum(topk) / total_rel))
            precision_values.append(float(np.mean(topk)) if len(topk) else 0.0)
            if np.sum(rel) > 0:
                precisions = [np.mean(rel[:rank + 1]) for rank in range(len(rel)) if rel[rank] == 1]
                ap_values.append(float(np.mean(precisions)) if precisions else 0.0)
                first_hit = np.argmax(rel == 1) + 1 if np.any(rel == 1) else 0
                rr_values.append(float(1.0 / first_hit) if first_hit > 0 else 0.0)
        metrics = {
            "recall_at_k": _safe_mean(recall_values),
            "precision_at_k": _safe_mean(precision_values),
            "map": _safe_mean(ap_values),
            "mrr": _safe_mean(rr_values),
        }
        per_model_raw[model_name] = metrics
        per_model_norm[model_name] = {key: _clamp_01(value) for key, value in metrics.items()}
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    selected_metric, metric_note = _resolve_metric("retrieval", metric_priority, raw_metrics.keys(), fallback="recall_at_k")
    summary = (
        f"Image retrieval evaluated with embedding-based nearest-neighbor ranking across {len(per_model_raw)} feature extractors. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "retrieval", metric_priority, selected_metric, raw_metrics, norm_metrics,
        per_model_raw, {"metric_note": metric_note, "k": min(5, len(X_proc) - 1)}, "retrieval", summary, 0.0,
        metrics_std=raw_std, normalized_metrics_std=norm_std, n_splits=1, n_models=len(per_model_raw)
    )


def _supervised_anomaly_metrics(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics = {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        try:
            metrics["auroc"] = roc_auc_score(y_true, scores)
        except Exception:
            pass
        try:
            metrics["auprc"] = average_precision_score(y_true, scores)
        except Exception:
            pass
    return metrics


def _proxy_anomaly_metrics(scores: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    inlier_scores = scores[preds == 0]
    outlier_scores = scores[preds == 1]
    if len(inlier_scores) and len(outlier_scores):
        separation = _clamp_01(0.5 + 0.5 * np.tanh((np.mean(outlier_scores) - np.mean(inlier_scores)) / _safe_scale(np.std(scores), 1.0)))
    else:
        separation = 0.0
    contamination = float(np.mean(preds))
    target_contamination = min(0.15, max(0.02, 5.0 / max(len(preds), 1)))
    stability = _clamp_01(1.0 - abs(contamination - target_contamination) / max(target_contamination, 1e-6))
    proxy_score = _clamp_01(0.6 * separation + 0.4 * stability)
    return {"proxy_score": proxy_score, "precision": stability, "recall": separation, "f1": proxy_score}


def _evaluate_anomaly(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    X_proc, _, _, _, kept_labels = _load_images(profile, spec)
    if len(kept_labels) < 10:
        raise ValueError("Anomaly / defect detection needs at least 10 valid images after preprocessing.")
    embeddings = _build_embeddings(X_proc, spec)
    ground_truth = _parse_binary_anomaly_labels(kept_labels)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    for model_name, features in embeddings.items():
        contamination = min(0.15, max(0.02, 5.0 / max(len(features), 1)))
        detectors = {
            "iforest": IsolationForest(contamination=contamination, random_state=42),
            "ocsvm": OneClassSVM(nu=contamination, gamma="scale"),
        }
        detector_metrics = []
        for det_name, detector in detectors.items():
            if det_name == "iforest":
                detector.fit(features)
                pred = detector.predict(features)
                scores = -np.asarray(detector.decision_function(features), dtype=float)
            else:
                detector.fit(features)
                pred = detector.predict(features)
                scores = -np.asarray(detector.decision_function(features), dtype=float)
            pred_bin = (pred == -1).astype(int)
            if ground_truth is not None and len(ground_truth) == len(pred_bin):
                detector_metrics.append(_supervised_anomaly_metrics(ground_truth, scores, pred_bin))
            else:
                detector_metrics.append(_proxy_anomaly_metrics(scores, pred_bin))
        merged = {}
        for key in {k for metrics in detector_metrics for k in metrics}:
            merged[key] = _safe_mean([metrics.get(key, np.nan) for metrics in detector_metrics])
        per_model_raw[model_name] = merged
        per_model_norm[model_name] = {key: _clamp_01(value) for key, value in merged.items()}
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    fallback = "auroc" if ground_truth is not None and "auroc" in raw_metrics else "proxy_score"
    selected_metric, metric_note = _resolve_metric("anomaly", metric_priority, raw_metrics.keys(), fallback=fallback)
    mode = "supervised" if ground_truth is not None else "proxy"
    summary = (
        f"Anomaly / defect detection evaluated across lightweight anomaly detectors and image embeddings. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if mode == "proxy":
        summary = f"{summary} No explicit anomaly ground truth was found, so this run used proxy/internal metrics."
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "anomaly", metric_priority, selected_metric, raw_metrics, norm_metrics,
        per_model_raw, {"metric_note": metric_note, "has_ground_truth": ground_truth is not None},
        mode, summary, 0.0, metrics_std=raw_std, normalized_metrics_std=norm_std, n_splits=1, n_models=len(per_model_raw)
    )


def _ssim_like(orig: np.ndarray, proc: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = float(np.mean(orig))
    mu_y = float(np.mean(proc))
    sigma_x = float(np.var(orig))
    sigma_y = float(np.var(proc))
    sigma_xy = float(np.mean((orig - mu_x) * (proc - mu_y)))
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return _clamp_01(num / den) if den != 0 else 0.0


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    covmean = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.T
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _evaluate_generation(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    X_proc, X_orig, _, _, _ = _load_images(profile, spec)
    if len(X_proc) < 10:
        raise ValueError("Image generation / synthesis evaluation needs at least 10 valid images after preprocessing.")
    embeddings = _build_embeddings(X_proc, spec)
    baseline_embeddings = _build_embeddings(X_orig, spec)
    raw_metrics_per_model: Dict[str, Dict[str, float]] = {}
    norm_metrics_per_model: Dict[str, Dict[str, float]] = {}
    for model_name in embeddings:
        proc_features = embeddings[model_name]
        orig_features = baseline_embeddings[model_name]
        cos = _cosine_similarity_rows(orig_features, proc_features).reshape(-1)
        mse = np.mean((X_orig - X_proc) ** 2, axis=1)
        psnr = 20.0 * np.log10(255.0 / np.sqrt(np.clip(mse, 1e-12, None)))
        ssim = [_ssim_like(orig, proc) for orig, proc in zip(X_orig, X_proc)]
        fid = _frechet_distance(
            np.mean(orig_features, axis=0), np.cov(orig_features, rowvar=False),
            np.mean(proc_features, axis=0), np.cov(proc_features, rowvar=False),
        )
        lpips = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
        raw_metrics_per_model[model_name] = {
            "clip_similarity": _safe_mean(cos),
            "ssim": _safe_mean(ssim),
            "psnr": _safe_mean(psnr) / 40.0,
            "fid": fid,
            "lpips": lpips,
        }
        norm_metrics_per_model[model_name] = {
            "clip_similarity": _clamp_01(raw_metrics_per_model[model_name]["clip_similarity"]),
            "ssim": _clamp_01(raw_metrics_per_model[model_name]["ssim"]),
            "psnr": _clamp_01(raw_metrics_per_model[model_name]["psnr"]),
            "fid": _normalize_lower_better(raw_metrics_per_model[model_name]["fid"], 50.0),
            "lpips": _normalize_lower_better(raw_metrics_per_model[model_name]["lpips"], 0.5),
        }
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(raw_metrics_per_model, norm_metrics_per_model)
    selected_metric, metric_note = _resolve_metric("generation", metric_priority, raw_metrics.keys(), fallback="clip_similarity")
    summary = (
        f"Image generation / synthesis evaluated with paired original-vs-processed quality baselines across {len(raw_metrics_per_model)} feature extractors. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "generation", metric_priority, selected_metric, raw_metrics, norm_metrics,
        raw_metrics_per_model, {"metric_note": metric_note}, "generation", summary, 0.0,
        metrics_std=raw_std, normalized_metrics_std=norm_std, n_splits=1, n_models=len(raw_metrics_per_model)
    )


def _match_annotation_path(image_path: str, root: Path, keywords: Optional[List[str]] = None) -> Optional[Path]:
    image = Path(image_path)
    stem = image.stem.lower()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _ANNOTATION_EXTENSIONS:
            continue
        lower_stem = path.stem.lower()
        if lower_stem == stem or lower_stem.startswith(stem) or stem.startswith(lower_stem):
            if keywords and not any(keyword in path.name.lower() for keyword in keywords):
                continue
            return path
    return None


def _parse_boxes(annotation_path: Path) -> List[Tuple[float, float, float, float]]:
    suffix = annotation_path.suffix.lower()
    boxes: List[Tuple[float, float, float, float]] = []
    if suffix == ".txt":
        with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                nums = [float(token) for token in line.replace(",", " ").split() if token.replace(".", "", 1).replace("-", "", 1).isdigit()]
                if len(nums) >= 4:
                    boxes.append(tuple(nums[:4]))
    elif suffix == ".xml":
        root = ET.parse(annotation_path).getroot()
        for box in root.findall(".//bndbox"):
            try:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
                boxes.append((xmin, ymin, xmax, ymax))
            except Exception:
                continue
    elif suffix == ".json":
        with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
            data = json.load(handle)
        items = data.get("boxes") if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    values = [item.get(key, 0.0) for key in ["xmin", "ymin", "xmax", "ymax"]]
                    if len(values) == 4:
                        boxes.append(tuple(float(v) for v in values))
                elif isinstance(item, list) and len(item) >= 4:
                    boxes.append(tuple(float(v) for v in item[:4]))
    return boxes


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _evaluate_detection(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    image_entries = []
    for path_str in profile.image_paths:
        ann = _match_annotation_path(path_str, profile.root_path)
        if ann is None:
            continue
        boxes = _parse_boxes(ann)
        if not boxes:
            continue
        try:
            image = Image.open(path_str)
            image.load()
            width, height = image.size
            image_entries.append((width, height, boxes))
        except Exception:
            continue
    if not image_entries:
        raise ValueError("Object detection requires readable bounding box annotations; none were found.")
    metrics = {"map": [], "map_50": [], "precision": [], "recall": [], "mean_iou": []}
    for width, height, boxes in image_entries:
        pred_box = (0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height)
        ious = [_box_iou(pred_box, gt) for gt in boxes]
        best_iou = max(ious) if ious else 0.0
        precision = 1.0 if best_iou >= 0.5 else 0.0
        recall = precision / max(len(boxes), 1)
        metrics["mean_iou"].append(best_iou)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["map_50"].append(precision * recall)
        metrics["map"].append(best_iou * recall)
    raw_metrics = {key: _safe_mean(values) for key, values in metrics.items()}
    norm_metrics = {key: _clamp_01(value) for key, value in raw_metrics.items()}
    selected_metric, metric_note = _resolve_metric("detection", metric_priority, raw_metrics.keys(), fallback="map")
    summary = (
        f"Object detection used a lightweight annotation-aware fallback baseline with central-box proposals. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit lightweight proxy, not a full detector."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "detection", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"detection_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(image_entries)},
        "detection", summary, 0.0,
        metrics_std={f"{k}_std": _safe_std(v) for k, v in metrics.items()},
        normalized_metrics_std={f"{k}_std": _safe_std(v) for k, v in metrics.items()},
        n_splits=1, n_models=1
    )


def _load_mask(annotation_path: Path, resize: int) -> Optional[np.ndarray]:
    suffix = annotation_path.suffix.lower()
    try:
        if suffix in {".png", ".bmp", ".tiff", ".tif", ".webp", ".jpg", ".jpeg"}:
            img = Image.open(annotation_path)
            img.load()
            img = img.convert("L").resize((resize, resize), Image.NEAREST)
            arr = np.asarray(img, dtype=np.float32)
            return (arr > np.mean(arr)).astype(np.uint8)
        if suffix == ".npy":
            arr = np.load(annotation_path)
            arr = np.asarray(Image.fromarray(arr.astype(np.float32)).resize((resize, resize), Image.NEAREST))
            return (arr > np.mean(arr)).astype(np.uint8)
        if suffix == ".npz":
            data = np.load(annotation_path)
            first = data[list(data.files)[0]]
            arr = np.asarray(Image.fromarray(first.astype(np.float32)).resize((resize, resize), Image.NEAREST))
            return (arr > np.mean(arr)).astype(np.uint8)
    except Exception:
        return None
    return None


def _binary_mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    intersection = float(np.sum((pred_mask == 1) & (true_mask == 1)))
    union = float(np.sum((pred_mask == 1) | (true_mask == 1)))
    pred_sum = float(np.sum(pred_mask == 1))
    true_sum = float(np.sum(true_mask == 1))
    iou = intersection / union if union > 0 else 0.0
    dice = 2.0 * intersection / (pred_sum + true_sum) if pred_sum + true_sum > 0 else 0.0
    pixel_accuracy = float(np.mean(pred_mask == true_mask))
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / true_sum if true_sum > 0 else 0.0
    return {"mean_iou": iou, "pixel_accuracy": pixel_accuracy, "dice_score": dice, "precision": precision, "recall": recall}


def _evaluate_semantic_segmentation(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    resize = spec.resize if spec.resize > 0 else 64
    collected = []
    for path_str in profile.image_paths:
        ann = _match_annotation_path(path_str, profile.root_path, keywords=["mask", "seg", "label"])
        if ann is None:
            continue
        true_mask = _load_mask(ann, resize)
        image_arr = _load_original_image_array(path_str, resize, "grayscale")
        if true_mask is None or image_arr is None:
            continue
        pred_mask = (image_arr.reshape(resize, resize) > np.mean(image_arr)).astype(np.uint8)
        collected.append(_binary_mask_metrics(pred_mask, true_mask))
    if not collected:
        raise ValueError("Semantic segmentation requires segmentation masks; none were found.")
    raw_metrics = {key: _safe_mean([entry.get(key, np.nan) for entry in collected]) for key in ["mean_iou", "pixel_accuracy", "dice_score"]}
    norm_metrics = {key: _clamp_01(value) for key, value in raw_metrics.items()}
    selected_metric, metric_note = _resolve_metric("semantic_segmentation", metric_priority, raw_metrics.keys(), fallback="mean_iou")
    summary = (
        f"Semantic segmentation used a lightweight threshold-mask fallback against available masks. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit lightweight segmentation baseline."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "semantic_segmentation", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"semantic_segmentation_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(collected)},
        "segmentation", summary, 0.0,
        metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        normalized_metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        n_splits=1, n_models=1
    )


def _evaluate_instance_segmentation(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    resize = spec.resize if spec.resize > 0 else 64
    collected = []
    for path_str in profile.image_paths:
        ann = _match_annotation_path(path_str, profile.root_path, keywords=["mask", "instance", "seg"])
        if ann is None:
            continue
        true_mask = _load_mask(ann, resize)
        image_arr = _load_original_image_array(path_str, resize, "grayscale")
        if true_mask is None or image_arr is None:
            continue
        pred_mask = (image_arr.reshape(resize, resize) > np.mean(image_arr)).astype(np.uint8)
        metrics = _binary_mask_metrics(pred_mask, true_mask)
        collected.append({
            "mask_map": metrics["precision"] * metrics["recall"],
            "mask_iou": metrics["mean_iou"],
            "dice_score": metrics["dice_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    if not collected:
        raise ValueError("Instance segmentation requires instance masks; none were found.")
    raw_metrics = {key: _safe_mean([entry.get(key, np.nan) for entry in collected]) for key in ["mask_map", "mask_iou", "dice_score", "precision", "recall"]}
    norm_metrics = {key: _clamp_01(value) for key, value in raw_metrics.items()}
    selected_metric, metric_note = _resolve_metric("instance_segmentation", metric_priority, raw_metrics.keys(), fallback="mask_map")
    summary = (
        f"Instance segmentation used a lightweight mask-overlap fallback baseline. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit lightweight instance-mask baseline."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "instance_segmentation", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"instance_segmentation_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(collected)},
        "segmentation", summary, 0.0,
        metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        normalized_metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        n_splits=1, n_models=1
    )


def _parse_keypoints(annotation_path: Path) -> List[Tuple[float, float]]:
    suffix = annotation_path.suffix.lower()
    points: List[Tuple[float, float]] = []
    try:
        if suffix == ".json":
            with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
                data = json.load(handle)
            items = data.get("keypoints") if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "x" in item and "y" in item:
                        points.append((float(item["x"]), float(item["y"])))
                    elif isinstance(item, list) and len(item) >= 2:
                        points.append((float(item[0]), float(item[1])))
        elif suffix == ".csv":
            with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if "x" in row and "y" in row:
                        points.append((float(row["x"]), float(row["y"])))
        elif suffix == ".txt":
            with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    nums = [float(token) for token in line.replace(",", " ").split() if token.replace(".", "", 1).replace("-", "", 1).isdigit()]
                    if len(nums) >= 2:
                        points.append((nums[0], nums[1]))
    except Exception:
        return []
    return points


def _evaluate_keypoints(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    collected = []
    for path_str in profile.image_paths:
        ann = _match_annotation_path(path_str, profile.root_path, keywords=["key", "pose", "joint"])
        if ann is None:
            continue
        points = _parse_keypoints(ann)
        if not points:
            continue
        try:
            image = Image.open(path_str)
            image.load()
            width, height = image.size
        except Exception:
            continue
        scale = math.sqrt(width * height)
        pred = np.array([[width / 2.0, height / 2.0] for _ in points], dtype=float)
        true = np.array(points, dtype=float)
        errors = np.linalg.norm(pred - true, axis=1) / max(scale, 1e-6)
        mean_error = float(np.mean(errors))
        pck = float(np.mean(errors <= 0.2))
        oks = float(np.mean(np.exp(-(errors ** 2) / 0.08)))
        collected.append({"pck": pck, "oks_map": oks, "normalized_keypoint_error": mean_error})
    if not collected:
        raise ValueError("Keypoint / pose estimation requires keypoint annotations; none were found.")
    raw_metrics = {key: _safe_mean([entry.get(key, np.nan) for entry in collected]) for key in ["pck", "oks_map", "normalized_keypoint_error"]}
    norm_metrics = {
        "pck": _clamp_01(raw_metrics["pck"]),
        "oks_map": _clamp_01(raw_metrics["oks_map"]),
        "normalized_keypoint_error": _clamp_01(1.0 - min(raw_metrics["normalized_keypoint_error"], 1.0)),
    }
    selected_metric, metric_note = _resolve_metric("keypoint", metric_priority, raw_metrics.keys(), fallback="pck")
    summary = (
        f"Keypoint / pose estimation used a lightweight center-point fallback against available keypoint annotations. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit lightweight keypoint baseline."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "keypoint", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"keypoint_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(collected)},
        "keypoint", summary, 0.0,
        metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        normalized_metrics_std={f"{key}_std": _safe_std([norm_metrics[key] for _ in collected]) for key in norm_metrics},
        n_splits=1, n_models=1
    )


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        curr = [i]
        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _read_transcription(annotation_path: Path) -> Optional[str]:
    suffix = annotation_path.suffix.lower()
    try:
        if suffix == ".txt":
            return annotation_path.read_text(encoding="utf-8", errors="ignore").strip()
        if suffix == ".json":
            data = json.loads(annotation_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                for key in ["text", "transcription", "label"]:
                    if key in data:
                        return str(data[key]).strip()
        if suffix == ".csv":
            with open(annotation_path, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for key in ["text", "transcription", "label"]:
                        if key in row:
                            return str(row[key]).strip()
    except Exception:
        return None
    return None


def _evaluate_ocr(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    collected = []
    for path_str, label in zip(profile.image_paths, profile.image_labels):
        ann = _match_annotation_path(path_str, profile.root_path, keywords=["ocr", "text", "trans"])
        if ann is None:
            continue
        truth = _read_transcription(ann)
        if not truth:
            continue
        pred = Path(path_str).stem.replace("_", " ").replace("-", " ").strip() or str(label)
        char_dist = _levenshtein(pred.lower(), truth.lower())
        word_dist = _levenshtein(pred.lower().split(), truth.lower().split()) if False else None
        char_scale = max(len(truth), 1)
        word_truth = truth.split()
        word_pred = pred.split()
        word_scale = max(len(word_truth), 1)
        word_dist = _levenshtein(" ".join(word_pred), " ".join(word_truth))
        cer = char_dist / char_scale
        wer = word_dist / max(len(truth), 1)
        edit_similarity = 1.0 - min(cer, 1.0)
        exact = 1.0 if pred.strip().lower() == truth.strip().lower() else 0.0
        collected.append({
            "normalized_edit_similarity": edit_similarity,
            "exact_match_accuracy": exact,
            "cer": cer,
            "wer": wer,
        })
    if not collected:
        raise ValueError("OCR requires transcription labels; none were found.")
    raw_metrics = {key: _safe_mean([entry.get(key, np.nan) for entry in collected]) for key in ["normalized_edit_similarity", "exact_match_accuracy", "cer", "wer"]}
    norm_metrics = {
        "normalized_edit_similarity": _clamp_01(raw_metrics["normalized_edit_similarity"]),
        "exact_match_accuracy": _clamp_01(raw_metrics["exact_match_accuracy"]),
        "cer": _clamp_01(1.0 - min(raw_metrics["cer"], 1.0)),
        "wer": _clamp_01(1.0 - min(raw_metrics["wer"], 1.0)),
    }
    selected_metric, metric_note = _resolve_metric("ocr", metric_priority, raw_metrics.keys(), fallback="normalized_edit_similarity")
    summary = (
        f"OCR used a lightweight filename/text fallback against available transcriptions. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit OCR fallback baseline."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "ocr", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"ocr_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(collected)},
        "ocr", summary, 0.0,
        metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        normalized_metrics_std={f"{key}_std": _safe_std([norm_metrics[key] for _ in collected]) for key in norm_metrics},
        n_splits=1, n_models=1
    )


def _load_depth_map(annotation_path: Path, resize: int) -> Optional[np.ndarray]:
    suffix = annotation_path.suffix.lower()
    try:
        if suffix in {".png", ".bmp", ".tiff", ".tif", ".webp", ".jpg", ".jpeg"}:
            image = Image.open(annotation_path)
            image.load()
            image = image.convert("F").resize((resize, resize), Image.BILINEAR)
            return np.asarray(image, dtype=np.float32)
        if suffix == ".npy":
            data = np.load(annotation_path)
            image = Image.fromarray(data.astype(np.float32))
            return np.asarray(image.resize((resize, resize), Image.BILINEAR), dtype=np.float32)
        if suffix == ".npz":
            data = np.load(annotation_path)
            first = data[list(data.files)[0]]
            image = Image.fromarray(first.astype(np.float32))
            return np.asarray(image.resize((resize, resize), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return None
    return None


def _evaluate_depth(spec: ImagePipelineSpec, profile: ImageProfile, metric_priority: str) -> Dict[str, Any]:
    resize = spec.resize if spec.resize > 0 else 64
    collected = []
    for path_str in profile.image_paths:
        ann = _match_annotation_path(path_str, profile.root_path, keywords=["depth", "disp"])
        if ann is None:
            continue
        true_depth = _load_depth_map(ann, resize)
        image_arr = _load_original_image_array(path_str, resize, "grayscale")
        if true_depth is None or image_arr is None:
            continue
        pred_depth = image_arr.reshape(resize, resize).astype(np.float32)
        pred_depth = pred_depth / max(np.max(pred_depth), 1e-6)
        true_depth = true_depth.astype(np.float32)
        true_depth = true_depth / max(np.max(true_depth), 1e-6)
        diff = pred_depth - true_depth
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        abs_rel = float(np.mean(np.abs(diff) / np.clip(np.abs(true_depth), 1e-6, None)))
        ratio = np.maximum(pred_depth / np.clip(true_depth, 1e-6, None), true_depth / np.clip(pred_depth, 1e-6, None))
        delta = float(np.mean(ratio < 1.25))
        collected.append({"rmse": rmse, "mae": mae, "abs_rel": abs_rel, "delta_accuracy": delta})
    if not collected:
        raise ValueError("Depth estimation requires depth maps; none were found.")
    raw_metrics = {key: _safe_mean([entry.get(key, np.nan) for entry in collected]) for key in ["delta_accuracy", "rmse", "mae", "abs_rel"]}
    scale = max(raw_metrics["rmse"], raw_metrics["mae"], raw_metrics["abs_rel"], 1.0)
    norm_metrics = {
        "delta_accuracy": _clamp_01(raw_metrics["delta_accuracy"]),
        "rmse": _normalize_lower_better(raw_metrics["rmse"], scale),
        "mae": _normalize_lower_better(raw_metrics["mae"], scale),
        "abs_rel": _normalize_lower_better(raw_metrics["abs_rel"], scale),
    }
    selected_metric, metric_note = _resolve_metric("depth", metric_priority, raw_metrics.keys(), fallback="delta_accuracy")
    summary = (
        f"Depth estimation used a lightweight grayscale-to-depth fallback baseline against available depth targets. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}. This is an explicit lightweight depth baseline."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec, "depth", metric_priority, selected_metric, raw_metrics, norm_metrics,
        {"depth_fallback": raw_metrics}, {"metric_note": metric_note, "annotated_images": len(collected)},
        "depth", summary, 0.0,
        metrics_std={f"{key}_std": _safe_std([entry.get(key, np.nan) for entry in collected]) for key in raw_metrics},
        normalized_metrics_std={f"{key}_std": _safe_std([norm_metrics[key] for _ in collected]) for key in norm_metrics},
        n_splits=1, n_models=1
    )


def evaluate_pipeline(spec: ImagePipelineSpec, profile: ImageProfile, task_type: str, metric_priority: str) -> Dict[str, Any]:
    started = time.perf_counter()
    task_type = normalize_task_type(task_type)
    try:
        if spec.resize == 0 and task_type in {"classification", "multilabel", "retrieval", "anomaly", "generation"}:
            raise ValueError("This image evaluator requires resizing to be enabled for feature extraction.")
        if task_type == "classification":
            result = _evaluate_single_label_classification(spec, profile, metric_priority)
        elif task_type == "multilabel":
            result = _evaluate_multilabel_classification(spec, profile, metric_priority)
        elif task_type == "detection":
            result = _evaluate_detection(spec, profile, metric_priority)
        elif task_type == "semantic_segmentation":
            result = _evaluate_semantic_segmentation(spec, profile, metric_priority)
        elif task_type == "instance_segmentation":
            result = _evaluate_instance_segmentation(spec, profile, metric_priority)
        elif task_type == "keypoint":
            result = _evaluate_keypoints(spec, profile, metric_priority)
        elif task_type == "retrieval":
            result = _evaluate_retrieval(spec, profile, metric_priority)
        elif task_type == "anomaly":
            result = _evaluate_anomaly(spec, profile, metric_priority)
        elif task_type == "ocr":
            result = _evaluate_ocr(spec, profile, metric_priority)
        elif task_type == "generation":
            result = _evaluate_generation(spec, profile, metric_priority)
        elif task_type == "depth":
            result = _evaluate_depth(spec, profile, metric_priority)
        else:
            raise ValueError(f"Task type '{task_type}' is not supported by the image evaluator.")
        result["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:
        return _failed_result(spec, task_type, metric_priority, str(exc), time.perf_counter() - started)
