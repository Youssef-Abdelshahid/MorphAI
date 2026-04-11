import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageFilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder

from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

_DEVICE = torch.device("cpu")

_MODEL_NAMES: List[str] = [
    "mobilenet_v3_small",
    "shufflenet_v2_x0_5",
    "squeezenet1_1",
]

_CNN_CACHE: Dict[str, nn.Module] = {}


def _build_cnn_extractor(name: str) -> nn.Module:
    if name == "mobilenet_v3_small":
        m = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        extractor = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    elif name == "shufflenet_v2_x0_5":
        m = tv_models.shufflenet_v2_x0_5(
            weights=tv_models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        )
        extractor = nn.Sequential(
            m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5,
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
    elif name == "squeezenet1_1":
        m = tv_models.squeezenet1_1(
            weights=tv_models.SqueezeNet1_1_Weights.DEFAULT
        )
        extractor = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(name)

    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


def _get_extractor(name: str) -> nn.Module:
    if name not in _CNN_CACHE:
        _CNN_CACHE[name] = _build_cnn_extractor(name)
    return _CNN_CACHE[name]


def _flat_to_tensor(X_flat: np.ndarray, img_size: int, channels: int) -> torch.Tensor:
    n = X_flat.shape[0]
    if channels == 1:
        imgs = X_flat.reshape(n, img_size, img_size)
        t = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)
        t = t.expand(-1, 3, -1, -1)
    else:
        imgs = X_flat.reshape(n, img_size, img_size, 3).transpose(0, 3, 1, 2)
        t = torch.from_numpy(imgs.astype(np.float32))
    return t


def _extract_cnn_features(
    X_flat: np.ndarray,
    extractor: nn.Module,
    img_size: int,
    channels: int,
    batch_size: int = 64,
) -> np.ndarray:
    parts = []
    n = X_flat.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = _flat_to_tensor(X_flat[i : i + batch_size], img_size, channels)
            parts.append(extractor(batch).numpy())
    return np.concatenate(parts, axis=0)


def _preprocess_single_image(
    img: Image.Image,
    spec: ImagePipelineSpec,
) -> np.ndarray:
    if spec.color_mode == "grayscale":
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    if spec.resize > 0:
        img = img.resize((spec.resize, spec.resize), Image.LANCZOS)

    if spec.histogram_eq:
        try:
            from PIL import ImageOps
            if img.mode == "L":
                img = ImageOps.equalize(img)
            else:
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
        if std > 0:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
    elif spec.normalization == "minmax":
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        if rng > 0:
            arr = (arr - mn) / rng
        else:
            arr = arr * 0.0

    return arr.flatten()


def _augment_batch(
    X: np.ndarray,
    y: np.ndarray,
    spec: ImagePipelineSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    if not spec.has_augmentation:
        return X, y

    if spec.color_mode == "grayscale":
        h = w = spec.resize if spec.resize > 0 else int(np.sqrt(X.shape[1]))
        channels = 1
    else:
        total = X.shape[1]
        side = int(np.round((total / 3) ** 0.5))
        h = w = side
        channels = 3

    augmented_X = []
    augmented_y = []
    rng = np.random.RandomState(42)

    for i in range(X.shape[0]):
        vec = X[i]
        try:
            if channels == 1:
                img_arr = vec.reshape(h, w)
            else:
                img_arr = vec.reshape(h, w, channels)
        except ValueError:
            continue

        aug_imgs = []

        if spec.augment_h_flip:
            if channels == 1:
                aug_imgs.append(np.fliplr(img_arr))
            else:
                aug_imgs.append(np.fliplr(img_arr))

        if spec.augment_v_flip:
            if channels == 1:
                aug_imgs.append(np.flipud(img_arr))
            else:
                aug_imgs.append(np.flipud(img_arr))

        if spec.augment_rotation != "none":
            from scipy.ndimage import rotate as nd_rotate
            max_angle = 15 if spec.augment_rotation == "light" else 30
            angle = rng.uniform(-max_angle, max_angle)
            rotated = nd_rotate(img_arr, angle, reshape=False, mode="nearest")
            aug_imgs.append(rotated)

        if spec.augment_color_jitter and channels > 1:
            factor = rng.uniform(0.8, 1.2)
            jittered = np.clip(img_arr * factor, img_arr.min(), img_arr.max())
            aug_imgs.append(jittered)

        for aug in aug_imgs:
            augmented_X.append(aug.flatten())
            augmented_y.append(y[i])

    if augmented_X:
        X_out = np.vstack([X, np.array(augmented_X, dtype=X.dtype)])
        y_out = np.concatenate([y, np.array(augmented_y, dtype=y.dtype)])
        return X_out, y_out

    return X, y


def load_and_preprocess(
    profile: ImageProfile,
    spec: ImagePipelineSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    le = LabelEncoder()
    le.fit(profile.class_names)

    for path_str, label in zip(profile.image_paths, profile.image_labels):
        try:
            img = Image.open(path_str)
            img.load()
            vec = _preprocess_single_image(img, spec)
            X_list.append(vec)
            y_list.append(label)
        except Exception:
            continue

    if not X_list:
        return np.array([]), np.array([])

    X = np.array(X_list, dtype=np.float32)
    y = le.transform(y_list)
    return X, y


def evaluate_pipeline(
    spec: ImagePipelineSpec,
    profile: ImageProfile,
) -> Optional[Dict[str, Any]]:
    t_start = time.perf_counter()

    try:
        X, y = load_and_preprocess(profile, spec)

        if X.shape[0] < 10:
            return None

        n_classes = len(np.unique(y))
        if n_classes < 2:
            return None

        min_class = int(np.bincount(y.astype(int)).min())
        n_splits = min(5, min_class)
        if n_splits < 2:
            return None

        nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
        if nan_mask.any():
            X = X[~nan_mask]
            y = y[~nan_mask]
            if X.shape[0] < 10:
                return None

        if spec.resize <= 0:
            return None

        img_size = spec.resize
        channels = 1 if spec.color_mode == "grayscale" else 3

        avg = "macro"
        scoring = {
            "accuracy":  "accuracy",
            "f1":        make_scorer(f1_score,        average=avg, zero_division=0),
            "precision": make_scorer(precision_score, average=avg, zero_division=0),
            "recall":    make_scorer(recall_score,    average=avg, zero_division=0),
        }
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        per_model_metrics: Dict[str, Dict[str, float]] = {}

        for model_name in _MODEL_NAMES:
            try:
                extractor = _get_extractor(model_name)
            except Exception:
                continue

            if spec.has_augmentation:
                model_scores: Dict[str, List[float]] = {
                    k: [] for k in ["accuracy", "f1", "precision", "recall"]
                }

                for train_idx, val_idx in cv.split(X, y):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    X_train_aug, y_train_aug = _augment_batch(X_train, y_train, spec)

                    if spec.imbalance == "oversample":
                        ros = RandomOverSampler(random_state=42)
                        X_train_aug, y_train_aug = ros.fit_resample(
                            X_train_aug, y_train_aug
                        )

                    F_train = _extract_cnn_features(
                        X_train_aug, extractor, img_size, channels
                    )
                    F_val = _extract_cnn_features(X_val, extractor, img_size, channels)

                    clf = LogisticRegression(
                        solver="saga", max_iter=500, tol=1e-3, random_state=42
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        clf.fit(F_train, y_train_aug)

                    y_pred = clf.predict(F_val)
                    model_scores["accuracy"].append(float(np.mean(y_pred == y_val)))
                    model_scores["f1"].append(
                        float(f1_score(y_val, y_pred, average=avg, zero_division=0))
                    )
                    model_scores["precision"].append(
                        float(precision_score(y_val, y_pred, average=avg, zero_division=0))
                    )
                    model_scores["recall"].append(
                        float(recall_score(y_val, y_pred, average=avg, zero_division=0))
                    )

                per_model_metrics[model_name] = {
                    k: float(np.mean(v)) for k, v in model_scores.items()
                }

            else:
                F = _extract_cnn_features(X, extractor, img_size, channels)

                steps: list = []
                if spec.imbalance == "oversample":
                    steps.append(("sampler", RandomOverSampler(random_state=42)))
                steps.append((
                    "classifier",
                    LogisticRegression(
                        solver="saga", max_iter=500, tol=1e-3, random_state=42
                    ),
                ))

                if spec.imbalance != "none":
                    pipeline = ImbPipeline(steps)
                else:
                    pipeline = SklearnPipeline(steps)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    cv_result = cross_validate(
                        pipeline, F, y,
                        cv=cv,
                        scoring=scoring,
                        error_score=0.0,
                        n_jobs=1,
                    )

                per_model_metrics[model_name] = {
                    k: float(np.mean(cv_result[f"test_{k}"]))
                    for k in ["accuracy", "f1", "precision", "recall"]
                }

        if not per_model_metrics:
            return None

        metrics: Dict[str, float] = {}
        metrics_std: Dict[str, float] = {}
        for k in ["accuracy", "f1", "precision", "recall"]:
            model_means = [per_model_metrics[mn][k] for mn in per_model_metrics]
            metrics[k] = float(np.mean(model_means))
            metrics_std[f"{k}_std"] = float(np.std(model_means))

        elapsed = time.perf_counter() - t_start
        return {
            "spec":              spec,
            "metrics":           metrics,
            "metrics_std":       metrics_std,
            "per_model_metrics": per_model_metrics,
            "n_splits":          n_splits,
            "n_models":          len(per_model_metrics),
            "elapsed_sec":       round(elapsed, 3),
        }

    except Exception:
        return None
