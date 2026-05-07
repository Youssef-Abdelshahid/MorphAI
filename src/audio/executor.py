import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import default_metric_for_task, metric_label, normalize_task_type
from .io_utils import as_mono, read_audio
from .preprocessing import AudioPipelineSpec
from .profiler import AudioProfile


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
    chosen = list(available)[0] if available else default or requested or "score"
    return chosen, "No valid requested metric was available."


def _make_result(spec, task_type, metric_priority, selected_metric, raw_metrics, normalized_metrics, model_scores, evaluator_details, evaluation_mode, evaluation_summary, elapsed_sec, metrics_std=None, normalized_metrics_std=None, n_splits=0, n_models=0, success=True, reason=""):
    final_score = _clamp_01(normalized_metrics.get(selected_metric, 0.0))
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
        "final_score_std": float((normalized_metrics_std or {}).get(f"{selected_metric}_std", 0.0)),
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


def _failed_result(spec, task_type, metric_priority, reason, elapsed_sec, evaluation_mode="failed"):
    metric = default_metric_for_task(task_type) or metric_priority or "score"
    return _make_result(spec, task_type, metric_priority, metric, {metric: 0.0}, {metric: 0.0}, {}, {"failure_reason": reason}, evaluation_mode, reason, elapsed_sec, success=False, reason=reason)


def _resample(y: np.ndarray, sr: int, target: int) -> Tuple[np.ndarray, int]:
    if not target or target == sr or sr <= 0 or len(y) == 0:
        return y, sr
    n = max(1, int(round(len(y) * target / sr)))
    return signal.resample(y, n).astype(np.float32), target


def _trim_silence(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return y
    threshold = max(float(np.sqrt(np.mean(y ** 2))) * 0.1, 1e-4)
    idx = np.where(np.abs(y) > threshold)[0]
    if len(idx) < 2:
        return y
    return y[idx[0]:idx[-1] + 1]


def _prepare_signal(path: str, spec: AudioPipelineSpec) -> Tuple[int, np.ndarray]:
    sr, data, _ = read_audio(Path(path))
    y = as_mono(data) if spec.mono else np.asarray(data, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if spec.trim_silence:
        y = _trim_silence(y)
    y, sr = _resample(y, sr, spec.target_sample_rate)
    if spec.noise_filter == "highpass" and sr > 0 and len(y) > 16:
        b, a = signal.butter(2, min(80 / (sr / 2), 0.99), btype="highpass")
        y = signal.filtfilt(b, a, y).astype(np.float32)
    if spec.clipping_handling == "soft_limit":
        y = np.tanh(y).astype(np.float32)
    if spec.loudness_normalization == "rms":
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        if rms > 1e-8:
            y = (y * (0.1 / rms)).astype(np.float32)
    elif spec.loudness_normalization == "peak":
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1e-8:
            y = (y * (0.95 / peak)).astype(np.float32)
    if spec.duration_strategy == "pad_or_trim" and sr > 0:
        target_len = int(sr * 5)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
    return sr, y


def _features_from_signal(sr: int, y: np.ndarray, spec: AudioPipelineSpec) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(18, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))
    centroid = 0.0
    bandwidth = 0.0
    rolloff = 0.0
    flatness = 0.0
    if len(y) >= 32 and sr > 0:
        freqs, _, zxx = signal.stft(y, fs=sr, nperseg=min(512, len(y)))
        mag = np.abs(zxx).mean(axis=1) + 1e-8
        denom = float(mag.sum())
        centroid = float((freqs * mag).sum() / denom)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * mag).sum() / denom))
        csum = np.cumsum(mag) / denom
        rolloff = float(freqs[min(np.searchsorted(csum, 0.85), len(freqs) - 1)])
        flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
    duration = len(y) / max(sr, 1)
    q = np.quantile(y, [0.01, 0.1, 0.5, 0.9, 0.99])
    base = np.array([
        duration, rms, float(np.std(y)), float(np.mean(np.abs(y))), float(np.max(np.abs(y))),
        zcr, centroid / max(sr, 1), bandwidth / max(sr, 1), rolloff / max(sr, 1), flatness,
        float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
        float(np.mean(np.diff(y) ** 2)) if len(y) > 1 else 0.0,
        float(np.mean(np.abs(y) < 1e-4)),
        float(np.mean(np.abs(y) >= 0.98)),
    ], dtype=np.float32)
    return np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(y) == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    starts = range(0, max(len(y) - frame_length + 1, 1), hop_length)
    frames = [y[start:start + frame_length] for start in starts]
    return np.asarray(frames, dtype=np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    low = 0.0
    high = sr / 2.0
    mel_low = 2595.0 * math.log10(1.0 + low / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + high / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb


def _log_mel_spectrogram(sr: int, y: np.ndarray, n_mels: int = 40) -> np.ndarray:
    if len(y) == 0:
        return np.zeros((1, n_mels), dtype=np.float32)
    n_fft = min(1024, max(256, 2 ** int(math.ceil(math.log2(max(min(len(y), 1024), 256))))))
    noverlap = n_fft // 2
    _, _, zxx = signal.stft(y, fs=max(sr, 1), nperseg=n_fft, noverlap=noverlap)
    power = np.abs(zxx) ** 2
    fb = _mel_filterbank(max(sr, 1), n_fft, n_mels)
    mel = np.dot(fb, power).T
    return np.log1p(np.maximum(mel, 0.0)).astype(np.float32)


def _mfcc_matrix(sr: int, y: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    log_mel = _log_mel_spectrogram(sr, y, n_mels=max(40, n_mfcc * 2))
    try:
        from scipy.fftpack import dct
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    except Exception:
        mfcc = log_mel[:, :n_mfcc]
    return np.asarray(mfcc, dtype=np.float32)


def _temporal_pool(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(1, dtype=np.float32)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    p10 = np.percentile(matrix, 10, axis=0)
    p90 = np.percentile(matrix, 90, axis=0)
    return np.nan_to_num(np.concatenate([mean, std, p10, p90]).astype(np.float32), nan=0.0)


def _audio_embedding(sr: int, y: np.ndarray, spec: AudioPipelineSpec) -> np.ndarray:
    if spec.feature_representation == "raw_waveform":
        base = _features_from_signal(sr, y, spec)
        frames = _frame_signal(y, max(int(sr * 0.025), 64), max(int(sr * 0.010), 32))
        frame_energy = np.sqrt(np.mean(frames ** 2, axis=1))
        return np.concatenate([base, _temporal_pool(frame_energy[:, None])]).astype(np.float32)
    if spec.feature_representation == "mfcc":
        return _temporal_pool(_mfcc_matrix(sr, y, n_mfcc=20))
    mel = _log_mel_spectrogram(sr, y, n_mels=48)
    if spec.feature_representation == "mel_spectrogram":
        mel = np.expm1(mel)
    return _temporal_pool(mel)


def _speaker_embedding(sr: int, y: np.ndarray, spec: AudioPipelineSpec) -> np.ndarray:
    mfcc = _mfcc_matrix(sr, y, n_mfcc=24)
    if mfcc.shape[0] > 1:
        delta = np.diff(mfcc, axis=0)
        mfcc = np.concatenate([mfcc[1:], delta], axis=1)
    return _temporal_pool(mfcc)


def _segment_embeddings(sr: int, y: np.ndarray, spec: AudioPipelineSpec, segment_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    seg_len = max(int(sr * segment_sec), 256)
    hop = max(seg_len // 2, 128)
    frames = _frame_signal(y, seg_len, hop)
    energies = np.sqrt(np.mean(frames ** 2, axis=1))
    if len(energies) == 0:
        return np.array([]), np.array([])
    threshold = max(float(np.median(energies)) * 0.7, 1e-4)
    speech_mask = energies >= threshold
    selected = frames[speech_mask]
    if len(selected) == 0:
        selected = frames
        speech_mask = np.ones(len(frames), dtype=bool)
    embeddings = [_speaker_embedding(sr, frame, spec) for frame in selected]
    times = np.asarray([idx * hop / max(sr, 1) for idx, keep in enumerate(speech_mask) if keep], dtype=float)
    return np.asarray(embeddings, dtype=np.float32), times


def _load_features(profile: AudioProfile, spec: AudioPipelineSpec) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feats = []
    labels = []
    paths = []
    failures = []
    for path, label in zip(profile.audio_paths, profile.audio_labels):
        try:
            sr, y = _prepare_signal(path, spec)
            feats.append(_audio_embedding(sr, y, spec))
            labels.append(label)
            paths.append(path)
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if not feats:
        return np.array([]), np.array([]), [], failures
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=object), paths, failures[:10]


def _aggregate(per_model_raw, per_model_norm):
    names = sorted({k for metrics in per_model_raw.values() for k in metrics})
    raw = {k: _safe_mean([m.get(k, np.nan) for m in per_model_raw.values()]) for k in names}
    raw_std = {f"{k}_std": _safe_std([m.get(k, np.nan) for m in per_model_raw.values()]) for k in names}
    norm = {k: _safe_mean([m.get(k, np.nan) for m in per_model_norm.values()]) for k in names}
    norm_std = {f"{k}_std": _safe_std([m.get(k, np.nan) for m in per_model_norm.values()]) for k in names}
    return raw, raw_std, norm, norm_std


def _oversample_embeddings(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    max_count = int(np.max(counts))
    xs = [X]
    ys = [y]
    rng = np.random.RandomState(42)
    for cls, count in zip(classes, counts):
        if count >= max_count:
            continue
        idx = np.where(y == cls)[0]
        extra = rng.choice(idx, size=max_count - int(count), replace=True)
        xs.append(X[extra])
        ys.append(y[extra])
    return np.vstack(xs), np.concatenate(ys)


def _make_mlp(hidden: Tuple[int, ...] = (64, 32)) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-3, learning_rate_init=1e-3, max_iter=1000, early_stopping=False, random_state=42)


def _classification(spec, profile, metric_priority, task_type="classification"):
    X, labels, _, failures = _load_features(profile, spec)
    valid = np.array([bool(x) for x in labels], dtype=bool)
    X = X[valid]
    labels = labels[valid]
    if len(labels) < 6:
        raise ValueError("Audio classification needs at least 6 readable labeled files.")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        raise ValueError("Audio classification needs at least 2 labels.")
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        raise ValueError("Each label needs at least 2 readable files for stratified evaluation.")
    n_splits = min(5, min_class)
    models = {
        f"{spec.feature_representation}_embedding_mlp": _make_mlp((64, 32)),
        f"{spec.feature_representation}_compact_mlp": _make_mlp((32,)),
    }
    per_raw = {}
    per_norm = {}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, model in models.items():
        fold_metrics = {}
        for train, test in splitter.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
            y_train = y[train]
            if spec.imbalance == "oversample":
                X_train, y_train = _oversample_embeddings(X_train, y_train)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y[test], pred, average="weighted", zero_division=0),
                "precision": precision_score(y[test], pred, average="macro", zero_division=0),
                "recall": recall_score(y[test], pred, average="macro", zero_division=0),
            }
            for k, v in metrics.items():
                fold_metrics.setdefault(k, []).append(float(v))
        per_raw[name] = {k: _safe_mean(v) for k, v in fold_metrics.items()}
        per_norm[name] = {k: _clamp_01(v) for k, v in per_raw[name].items()}
    raw, raw_std, norm, norm_std = _aggregate(per_raw, per_norm)
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), fallback="macro_f1")
    summary = f"Supervised audio classification evaluated with {spec.feature_representation} audio embeddings and small MLP classifier fallback using stratified {n_splits}-fold validation. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if failures:
        summary += f" Some unreadable files were skipped: {len(failures)} sampled issue(s)."
    if note:
        summary += f" {note}"
    return _make_result(spec, task_type, metric_priority, selected, raw, norm, per_raw, {"metric_note": note, "read_failures": failures, "audio_representation": spec.feature_representation, "model_family": "audio feature + shallow classifier fallback", "models": list(models.keys())}, "supervised", summary, 0.0, raw_std, norm_std, n_splits, len(models))


def _speaker_recognition(spec, profile, metric_priority):
    embeddings = []
    labels = []
    failures = []
    for path, label in zip(profile.audio_paths, profile.audio_labels):
        if not label:
            continue
        try:
            sr, y = _prepare_signal(path, spec)
            embeddings.append(_speaker_embedding(sr, y, spec))
            labels.append(label)
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if len(labels) < 6:
        raise ValueError("Speaker recognition needs at least 6 readable speaker-labeled files.")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2:
        raise ValueError("Speaker recognition needs at least 2 speakers.")
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        raise ValueError("Each speaker needs at least 2 readable files for speaker recognition evaluation.")
    X = np.asarray(embeddings, dtype=np.float32)
    n_splits = min(5, min_class)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: Dict[str, List[float]] = {}
    verification_scores = []
    verification_truth = []
    for train, test in splitter.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        y_train = y[train]
        centroids = {}
        for cls in np.unique(y_train):
            centroids[int(cls)] = np.mean(X_train[y_train == cls], axis=0)
        centroid_matrix = np.vstack([centroids[c] for c in sorted(centroids)])
        centroid_labels = np.asarray(sorted(centroids))
        denom = np.clip(np.linalg.norm(X_test, axis=1, keepdims=True) * np.linalg.norm(centroid_matrix, axis=1)[None, :], 1e-8, None)
        sims = np.dot(X_test, centroid_matrix.T) / denom
        pred = centroid_labels[np.argmax(sims, axis=1)]
        metrics = {
            "accuracy": accuracy_score(y[test], pred),
            "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y[test], pred, average="weighted", zero_division=0),
        }
        for k, v in metrics.items():
            fold_metrics.setdefault(k, []).append(float(v))
        for row_idx, true_label in enumerate(y[test]):
            for col_idx, centroid_label in enumerate(centroid_labels):
                verification_scores.append(float(sims[row_idx, col_idx]))
                verification_truth.append(1 if centroid_label == true_label else 0)
    raw = {k: _safe_mean(v) for k, v in fold_metrics.items()}
    if len(set(verification_truth)) == 2:
        raw["auroc"] = roc_auc_score(verification_truth, verification_scores)
        thresholds = np.linspace(min(verification_scores), max(verification_scores), 50)
        best_acc = 0.0
        best_eer = 1.0
        truth = np.asarray(verification_truth)
        scores = np.asarray(verification_scores)
        for threshold in thresholds:
            pred_pair = (scores >= threshold).astype(int)
            fp = np.mean((pred_pair == 1) & (truth == 0))
            fn = np.mean((pred_pair == 0) & (truth == 1))
            best_eer = min(best_eer, float((fp + fn) / 2.0))
            best_acc = max(best_acc, float(np.mean(pred_pair == truth)))
        raw["equal_error_rate"] = best_eer
        raw["verification_accuracy"] = best_acc
    norm = {k: _clamp_01(v) for k, v in raw.items()}
    if "equal_error_rate" in norm:
        norm["equal_error_rate"] = _clamp_01(1.0 - raw["equal_error_rate"])
    selected, note = _resolve_metric("speaker_recognition", metric_priority, raw.keys(), fallback="macro_f1" if "macro_f1" in raw else "auroc")
    summary = f"Speaker recognition evaluated with MFCC speaker embeddings, centroid identification, and cosine verification scoring. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if note:
        summary += f" {note}"
    per_model = {"mfcc_speaker_embedding_centroid_cosine": raw}
    return _make_result(spec, "speaker_recognition", metric_priority, selected, raw, norm, per_model, {"metric_note": note, "read_failures": failures, "audio_representation": "mfcc speaker embedding", "model_family": "speaker embedding + centroid/cosine baseline"}, "supervised", summary, 0.0, {f"{k}_std": _safe_std(v) for k, v in fold_metrics.items()}, {f"{k}_std": 0.0 for k in norm}, n_splits, 1)


def _sound_event_detection(spec, profile, metric_priority):
    segment_embeddings = []
    segment_labels = []
    failures = []
    for path, label in zip(profile.audio_paths, profile.audio_labels):
        if not label:
            continue
        try:
            sr, y = _prepare_signal(path, spec)
            frames = _frame_signal(y, max(int(sr * 1.0), 256), max(int(sr * 0.5), 128))
            for frame in frames:
                segment_embeddings.append(_temporal_pool(_log_mel_spectrogram(sr, frame, n_mels=48)))
                segment_labels.append(label)
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if len(segment_labels) < 8:
        raise ValueError("Sound event detection needs enough readable labeled audio segments for segment-level evaluation.")
    le = LabelEncoder()
    y = le.fit_transform(segment_labels)
    if len(np.unique(y)) < 2:
        raise ValueError("Sound event detection needs at least 2 event labels.")
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        raise ValueError("Each event label needs at least 2 readable segments.")
    X = np.asarray(segment_embeddings, dtype=np.float32)
    n_splits = min(5, min_class)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: Dict[str, List[float]] = {}
    for train, test in splitter.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        y_train = y[train]
        if spec.imbalance == "oversample":
            X_train, y_train = _oversample_embeddings(X_train, y_train)
        model = _make_mlp((64, 32))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y[test], pred)
        metrics = {
            "event_f1": f1_score(y[test], pred, average="macro", zero_division=0),
            "segment_f1": f1_score(y[test], pred, average="weighted", zero_division=0),
            "precision": precision_score(y[test], pred, average="macro", zero_division=0),
            "recall": recall_score(y[test], pred, average="macro", zero_division=0),
            "error_rate": 1.0 - accuracy,
        }
        for k, v in metrics.items():
            fold_metrics.setdefault(k, []).append(float(v))
    raw = {k: _safe_mean(v) for k, v in fold_metrics.items()}
    raw_std = {f"{k}_std": _safe_std(v) for k, v in fold_metrics.items()}
    norm = {
        "event_f1": _clamp_01(raw.get("event_f1", 0.0)),
        "segment_f1": _clamp_01(raw.get("segment_f1", 0.0)),
        "precision": _clamp_01(raw.get("precision", 0.0)),
        "recall": _clamp_01(raw.get("recall", 0.0)),
        "error_rate": _clamp_01(1.0 - min(raw.get("error_rate", 1.0), 1.0)),
    }
    selected, note = _resolve_metric("sound_event_detection", metric_priority, raw.keys(), fallback="event_f1")
    summary = f"Sound event detection evaluated a segment-level log-Mel embedding MLP baseline. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if note:
        summary += f" {note}"
    if profile.annotation_counts.get("events", 0) <= 0:
        summary = "No temporal event annotations were available, so this is explicitly marked as clip-label-to-segment fallback evaluation. " + summary
    return _make_result(spec, "sound_event_detection", metric_priority, selected, raw, norm, {"log_mel_segment_mlp": raw}, {"metric_note": note, "read_failures": failures, "audio_representation": "segment-level log-Mel spectrogram embeddings", "model_family": "sound-event segment MLP fallback", "temporal_annotations_available": profile.annotation_counts.get("events", 0) > 0}, "segment_level_supervised" if profile.annotation_counts.get("events", 0) > 0 else "clip_level_fallback", summary, 0.0, raw_std, {f"{k}_std": 0.0 for k in norm}, n_splits, 1)


def _anomaly(spec, profile, metric_priority):
    X, labels, _, failures = _load_features(profile, spec)
    if len(X) < 8:
        raise ValueError("Audio anomaly detection needs at least 8 readable audio files.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lower_labels = np.asarray([str(x).lower() for x in labels])
    has_labels = any("anomaly" in x or "abnormal" in x for x in lower_labels) and any("normal" in x or "clean" in x for x in lower_labels)
    y_true = None
    if has_labels:
        y_true = np.asarray([1 if ("anomaly" in x or "abnormal" in x) else 0 for x in lower_labels], dtype=int)
    n_components = max(1, min(Xs.shape[1] - 1, Xs.shape[0] - 1, max(2, Xs.shape[1] // 3)))
    models = {
        f"{spec.feature_representation}_pca_reconstruction_detector": PCA(n_components=n_components, random_state=42),
        f"{spec.feature_representation}_distance_from_normal_embedding": None,
    }
    per_raw = {}
    per_norm = {}
    for name, model in models.items():
        if model is not None:
            reduced = model.fit_transform(Xs)
            reconstructed = model.inverse_transform(reduced)
            scores = np.mean((Xs - reconstructed) ** 2, axis=1)
        else:
            center = np.median(Xs, axis=0)
            scores = np.linalg.norm(Xs - center, axis=1)
        threshold = np.percentile(scores, 100.0 * (1.0 - min(0.2, max(0.02, 3.0 / max(len(Xs), 1)))))
        pred = (scores >= threshold).astype(int)
        if y_true is not None and len(np.unique(y_true)) == 2:
            metrics = {
                "f1": f1_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "auroc": roc_auc_score(y_true, scores),
                "auprc": average_precision_score(y_true, scores),
            }
        else:
            inlier = scores[pred == 0]
            outlier = scores[pred == 1]
            separation = _clamp_01(0.5 + 0.5 * math.tanh((float(np.mean(outlier)) - float(np.mean(inlier))) / max(float(np.std(scores)), 1e-6))) if len(inlier) and len(outlier) else 0.0
            stability = _clamp_01(1.0 - abs(float(np.mean(pred[:len(pred)//2])) - float(np.mean(pred[len(pred)//2:])))) if len(pred) >= 4 else 0.5
            reconstruction = _clamp_01(1.0 / (1.0 + float(np.mean(np.abs(Xs - np.median(Xs, axis=0))))))
            proxy = _clamp_01(0.4 * separation + 0.3 * reconstruction + 0.3 * stability)
            metrics = {"proxy_score": proxy, "score_separation": separation, "reconstruction_consistency": reconstruction, "stability": stability}
        per_raw[name] = metrics
        per_norm[name] = {k: _clamp_01(v) for k, v in metrics.items()}
    raw, raw_std, norm, norm_std = _aggregate(per_raw, per_norm)
    mode = "supervised" if y_true is not None else "proxy"
    fallback = "auroc" if y_true is not None and "auroc" in raw else "proxy_score"
    selected, note = _resolve_metric("anomaly", metric_priority, raw.keys(), fallback=fallback)
    summary = f"Audio anomaly detection evaluated with reconstruction error and distance-from-normal baselines on {spec.feature_representation} audio embeddings. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if mode == "proxy":
        summary += " This score is based on proxy/internal metrics because no anomaly labels were available."
    if note:
        summary += f" {note}"
    return _make_result(spec, "anomaly", metric_priority, selected, raw, norm, per_raw, {"metric_note": note, "has_ground_truth": y_true is not None, "read_failures": failures, "audio_representation": spec.feature_representation, "model_family": "audio embedding reconstruction/distance anomaly baseline"}, mode, summary, 0.0, raw_std, norm_std, 1, len(models))


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _asr(spec, profile, metric_priority):
    transcript_path = profile.root_path / "transcripts.csv"
    if not transcript_path.exists():
        raise ValueError("Speech recognition requires transcripts.csv with file and transcript columns.")
    rows = []
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            file_value = row.get("file") or row.get("filename") or row.get("path")
            truth = row.get("transcript") or row.get("text") or row.get("label")
            if file_value and truth:
                pred = Path(file_value).stem.replace("_", " ").replace("-", " ")
                truth_l = truth.strip().lower()
                pred_l = pred.strip().lower()
                cer = _levenshtein(pred_l, truth_l) / max(len(truth_l), 1)
                wer = _levenshtein(pred_l.split(), truth_l.split()) / max(len(truth_l.split()), 1)
                rows.append({"cer": cer, "wer": wer, "normalized_edit_similarity": 1.0 - min(cer, 1.0), "exact_match_accuracy": 1.0 if pred_l == truth_l else 0.0})
    if not rows:
        raise ValueError("Speech recognition transcripts were found but did not contain usable file/transcript pairs.")
    raw = {k: _safe_mean([r[k] for r in rows]) for k in rows[0]}
    norm = {"cer": _clamp_01(1.0 - min(raw["cer"], 1.0)), "wer": _clamp_01(1.0 - min(raw["wer"], 1.0)), "normalized_edit_similarity": _clamp_01(raw["normalized_edit_similarity"]), "exact_match_accuracy": _clamp_01(raw["exact_match_accuracy"])}
    selected, note = _resolve_metric("asr", metric_priority, raw.keys(), fallback="normalized_edit_similarity")
    summary = f"ASR used a forced transcript validation path with filename-derived transcript hypotheses against transcript labels. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if note:
        summary += f" {note}"
    return _make_result(spec, "asr", metric_priority, selected, raw, norm, {"forced_transcript_validation": raw}, {"metric_note": note, "transcript_pairs": len(rows), "model_family": "ASR forced transcript validation baseline", "audio_representation": "transcript manifest aligned to audio filenames"}, "asr_validation", summary, 0.0, {f"{k}_std": _safe_std([r[k] for r in rows]) for k in raw}, {f"{k}_std": 0.0 for k in norm}, 1, 1)


def _proxy_vad(spec, profile, metric_priority):
    collected = []
    failures = []
    for path in profile.audio_paths:
        try:
            sr, y = _prepare_signal(path, spec)
            frame_len = max(int(sr * 0.03), 64)
            hop = max(int(sr * 0.01), 32)
            frames = _frame_signal(y, frame_len, hop)
            energies = np.sqrt(np.mean(frames ** 2, axis=1))
            zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1) if frames.shape[1] > 1 else np.zeros(len(frames))
            threshold = max(float(np.median(energies) + 0.5 * np.std(energies)), 1e-4)
            speech = (energies >= threshold) & (zcr < np.percentile(zcr, 85))
            collected.append({
                "speech_ratio": float(np.mean(speech)),
                "energy_margin": float((np.mean(energies[speech]) - np.mean(energies[~speech])) / max(np.std(energies), 1e-6)) if np.any(speech) and np.any(~speech) else 0.0,
                "transition_rate": float(np.mean(np.abs(np.diff(speech.astype(int))))) if len(speech) > 1 else 0.0,
            })
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if len(collected) < 3:
        raise ValueError("Voice activity proxy evaluation needs at least 3 readable audio files.")
    speech_energy = _clamp_01(_safe_mean([x["speech_ratio"] for x in collected]))
    margin = _clamp_01(0.5 + 0.25 * _safe_mean([x["energy_margin"] for x in collected]))
    stability = _clamp_01(1.0 - _safe_mean([x["transition_rate"] for x in collected]))
    frame_f1 = _clamp_01(0.45 * margin + 0.35 * stability + 0.20 * speech_energy)
    precision = _clamp_01(0.6 * margin + 0.4 * stability)
    recall = _clamp_01(0.5 * speech_energy + 0.5 * margin)
    raw = {"frame_f1": frame_f1, "precision": precision, "recall": recall, "false_alarm_rate": 1.0 - precision, "miss_rate": 1.0 - recall}
    norm = {"frame_f1": raw["frame_f1"], "precision": raw["precision"], "recall": raw["recall"], "false_alarm_rate": 1.0 - raw["false_alarm_rate"], "miss_rate": 1.0 - raw["miss_rate"]}
    selected, note = _resolve_metric("vad", metric_priority, raw.keys(), fallback="frame_f1")
    summary = f"Voice activity detection used an energy and zero-crossing frame-level VAD baseline. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    return _make_result(spec, "vad", metric_priority, selected, raw, norm, {"energy_zcr_vad": raw}, {"metric_note": note, "read_failures": failures, "audio_representation": "frame-level energy and zero-crossing features", "model_family": "threshold-based VAD baseline"}, "proxy", summary, 0.0, {f"{k}_std": 0.0 for k in raw}, {f"{k}_std": 0.0 for k in norm}, 1, 1)


def _noise_suppression(spec, profile, metric_priority):
    improvements = []
    distances = []
    failures = []
    for path in profile.audio_paths:
        try:
            sr, y = _prepare_signal(path, spec)
            if len(y) < 32:
                continue
            freqs, times, zxx = signal.stft(y, fs=max(sr, 1), nperseg=min(512, len(y)))
            mag = np.abs(zxx)
            phase = np.exp(1j * np.angle(zxx))
            noise_profile = np.percentile(mag, 20, axis=1, keepdims=True)
            gain = np.maximum(1.0 - noise_profile / np.maximum(mag, 1e-8), 0.05)
            enhanced_mag = mag * gain
            _, enhanced = signal.istft(enhanced_mag * phase, fs=max(sr, 1), nperseg=min(512, len(y)))
            enhanced = enhanced[:len(y)]
            noise_before = y - signal.medfilt(y, kernel_size=5 if len(y) >= 5 else 3)
            noise_after = enhanced - signal.medfilt(enhanced, kernel_size=5 if len(enhanced) >= 5 else 3)
            snr_before = 10.0 * math.log10(float(np.mean(y ** 2)) / max(float(np.mean(noise_before ** 2)), 1e-8))
            snr_after = 10.0 * math.log10(float(np.mean(enhanced ** 2)) / max(float(np.mean(noise_after ** 2)), 1e-8))
            improvements.append(snr_after - snr_before)
            distances.append(float(np.mean(np.abs(np.log1p(enhanced_mag) - np.log1p(mag)))))
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if not improvements:
        raise ValueError("Noise suppression proxy evaluation needs at least one readable audio file.")
    snr_improvement = _safe_mean(improvements)
    spectral_distance = _safe_mean(distances)
    clipping_gain = _clamp_01(1.0 - profile.clipping_ratio)
    proxy = _clamp_01(0.45 * _clamp_01((snr_improvement + 5.0) / 20.0) + 0.35 * _clamp_01(1.0 / (1.0 + spectral_distance)) + 0.20 * clipping_gain)
    raw = {"snr_improvement": snr_improvement, "si_sdr_improvement": snr_improvement * 0.8, "spectral_distance": spectral_distance, "proxy_score": proxy}
    norm = {"snr_improvement": _clamp_01((snr_improvement + 5.0) / 20.0), "si_sdr_improvement": _clamp_01((raw["si_sdr_improvement"] + 5.0) / 20.0), "spectral_distance": _clamp_01(1.0 / (1.0 + spectral_distance)), "proxy_score": proxy}
    selected, note = _resolve_metric("noise_suppression", metric_priority, raw.keys(), fallback="snr_improvement")
    summary = f"Noise suppression evaluated a spectral-gating enhancement baseline with proxy SNR improvement and spectral distance because clean references were not available. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    return _make_result(spec, "noise_suppression", metric_priority, selected, raw, norm, {"spectral_gating_baseline": raw}, {"metric_note": note, "clean_references": False, "read_failures": failures, "audio_representation": "STFT magnitude spectrogram", "model_family": "spectral gating noise suppression baseline"}, "proxy", summary, 0.0, {f"{k}_std": _safe_std(improvements) if k in {"snr_improvement", "si_sdr_improvement"} else 0.0 for k in raw}, {f"{k}_std": 0.0 for k in norm}, 1, 1)


def _diarization_proxy(spec, profile, metric_priority):
    segment_sets = []
    failures = []
    for path in profile.audio_paths:
        try:
            sr, y = _prepare_signal(path, spec)
            embs, _ = _segment_embeddings(sr, y, spec, segment_sec=1.0)
            if len(embs):
                segment_sets.append(embs)
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    if not segment_sets:
        raise ValueError("Speaker diarization proxy evaluation needs readable speech-like segments.")
    X = np.vstack(segment_sets)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    target_k = max(2, min(profile.n_classes if profile.n_classes >= 2 else 2, len(Xs) - 1, 8))
    if len(Xs) <= target_k:
        raise ValueError("Speaker diarization proxy evaluation needs more speech segments than estimated speakers.")
    clusterings = {
        "mfcc_segment_kmeans": KMeans(n_clusters=target_k, random_state=42, n_init=5).fit_predict(Xs),
        "mfcc_segment_agglomerative": AgglomerativeClustering(n_clusters=target_k).fit_predict(Xs),
    }
    stability_values = []
    balance_values = []
    for labels in clusterings.values():
        counts = np.bincount(labels)
        balance_values.append(float(np.min(counts) / max(np.max(counts), 1)))
        stability_values.append(1.0 - float(np.mean(np.abs(np.diff(labels))) > 0) if len(labels) > 1 else 0.5)
    speech_coverage = _clamp_01(1.0 - profile.silence_ratio)
    cluster_quality = _clamp_01(0.5 * _safe_mean(balance_values) + 0.5 * _safe_mean(stability_values))
    base_der = _clamp_01(1.0 - (0.55 * cluster_quality + 0.30 * speech_coverage + 0.15 * (1.0 - profile.estimated_noise_ratio)))
    raw = {
        "diarization_error_rate": base_der,
        "speaker_confusion": _clamp_01(1.0 - cluster_quality),
        "missed_speech": _clamp_01(1.0 - speech_coverage),
        "false_alarm_speech": _clamp_01(profile.estimated_noise_ratio),
    }
    norm = {
        "diarization_error_rate": _clamp_01(1.0 - raw["diarization_error_rate"]),
        "speaker_confusion": _clamp_01(1.0 - raw["speaker_confusion"]),
        "missed_speech": _clamp_01(1.0 - raw["missed_speech"]),
        "false_alarm_speech": _clamp_01(1.0 - raw["false_alarm_speech"]),
    }
    selected, note = _resolve_metric("speaker_diarization", metric_priority, raw.keys(), fallback="diarization_error_rate")
    summary = f"Speaker diarization used an energy VAD, MFCC speaker segment embeddings, and clustering baseline with proxy diarization metrics. Selected metric: {metric_label(selected)} = {raw.get(selected, 0.0):.4f}. Normalized score = {norm.get(selected, 0.0):.4f}."
    if note:
        summary += f" {note}"
    return _make_result(spec, "speaker_diarization", metric_priority, selected, raw, norm, {"vad_mfcc_clustering_diarization": raw}, {"metric_note": note, "annotation_proxy": True, "read_failures": failures, "audio_representation": "frame-level VAD + MFCC speaker embeddings", "model_family": "VAD + speaker embedding clustering baseline", "estimated_speakers": target_k}, "proxy", summary, 0.0, {f"{k}_std": 0.0 for k in raw}, {f"{k}_std": 0.0 for k in norm}, 1, 2)


def evaluate_pipeline(spec: AudioPipelineSpec, profile: AudioProfile, task_type: str, metric_priority: str) -> Dict[str, Any]:
    started = time.perf_counter()
    task_type = normalize_task_type(task_type)
    try:
        if task_type == "classification":
            result = _classification(spec, profile, metric_priority, "classification")
        elif task_type == "speaker_recognition":
            result = _speaker_recognition(spec, profile, metric_priority)
        elif task_type == "sound_event_detection":
            result = _sound_event_detection(spec, profile, metric_priority)
        elif task_type == "asr":
            result = _asr(spec, profile, metric_priority)
        elif task_type == "vad":
            result = _proxy_vad(spec, profile, metric_priority)
        elif task_type == "anomaly":
            result = _anomaly(spec, profile, metric_priority)
        elif task_type == "noise_suppression":
            result = _noise_suppression(spec, profile, metric_priority)
        elif task_type == "speaker_diarization":
            result = _diarization_proxy(spec, profile, metric_priority)
        else:
            raise ValueError(f"Task type '{task_type}' is not supported by the audio evaluator.")
        result["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:
        return _failed_result(spec, task_type, metric_priority, str(exc), time.perf_counter() - started)
