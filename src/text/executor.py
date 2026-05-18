from __future__ import annotations

import ast
import json
import math
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, hamming_loss, normalized_mutual_info_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from .config import default_metric_for_task, metric_label, normalize_task_type
from .preprocessing import TextPipelineSpec, clean_text_value
from .profiler import TextProfile


def _clamp_01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _safe_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(vals)) if vals else 0.0


def _resolve_metric(task_type: str, requested: str, available: Sequence[str], fallback: str = "") -> Tuple[str, str]:
    requested = (requested or "").strip().lower()
    if requested and requested in set(available):
        return requested, ""
    default = fallback or default_metric_for_task(task_type)
    if default in set(available):
        if requested:
            return default, f"Requested metric '{requested}' was unavailable; used {metric_label(default)} instead."
        return default, ""
    chosen = list(available)[0] if available else default or requested or "score"
    return chosen, "No valid requested metric was available."


def _norm_metric(metric: str, value: float) -> float:
    metric = metric.lower()
    if metric in {"hamming_loss"}:
        return _clamp_01(1.0 - min(max(float(value), 0.0), 1.0))
    if metric in {"spearman", "pearson", "silhouette", "ari"}:
        return _clamp_01((float(value) + 1.0) / 2.0)
    return _clamp_01(value)


def _make_result(spec, task_type, metric_priority, selected_metric, raw_metrics, model_scores, evaluator_details, evaluation_mode, evaluation_summary, elapsed_sec, metrics_std=None, n_splits=0, n_models=0, success=True, reason=""):
    normalized_metrics = {k: _norm_metric(k, v) for k, v in raw_metrics.items()}
    normalized_metrics_std = {f"{k}_std": 0.0 for k in raw_metrics}
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
        "normalized_metrics_std": normalized_metrics_std,
        "final_score": final_score,
        "normalized_score": final_score,
        "final_score_std": float((metrics_std or {}).get(f"{selected_metric}_std", 0.0)),
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
    return _make_result(spec, task_type, metric_priority, metric, {metric: 0.0}, {}, {"failure_reason": reason}, evaluation_mode, reason, elapsed_sec, success=False, reason=reason)


def _preprocess_series(series: pd.Series, spec: TextPipelineSpec, preserve_alignment: bool = False) -> List[str]:
    return [clean_text_value(v, spec, preserve_alignment=preserve_alignment) for v in series.fillna("").astype(str)]


def _vectorizer(spec: TextPipelineSpec) -> TfidfVectorizer:
    analyzer = "char_wb" if spec.representation in {"tfidf_char", "tfidf_char_word"} else "word"
    ngram_range = (3, 5) if analyzer == "char_wb" else (1, 2)
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=max(int(spec.min_df), 1), max_features=20000)


def _split(X, y, multilabel=False):
    n = len(X)
    if n < 4:
        idx = np.arange(n)
        return idx, idx
    test_size = max(1, int(round(n * 0.25)))
    stratify = None if multilabel else y
    try:
        tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=42, stratify=stratify)
    except Exception:
        tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=42)
    return tr, te


def _classification_metrics(y_true, y_pred, multilabel=False) -> Dict[str, float]:
    if multilabel:
        return {
            "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "subset_accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _parse_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    sep = "|" if "|" in s else ","
    return [x.strip() for x in s.split(sep) if x.strip()]


def _build_tabular_features(
    df: pd.DataFrame,
    profile: TextProfile,
    spec: TextPipelineSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    numeric_cols = list(profile.auxiliary_numeric_columns or [])
    categorical_cols = list(profile.auxiliary_categorical_columns or [])
    if not numeric_cols and not categorical_cols:
        return None

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    if numeric_cols:
        train_num = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        test_num = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if spec.numeric_imputation == "median":
            fill = train_num.median()
        else:
            fill = train_num.mean()
        fill = fill.fillna(0.0)
        train_num = train_num.fillna(fill)
        test_num = test_num.fillna(fill)
        if spec.numeric_scaling == "standard":
            mean = train_num.mean()
            std = train_num.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
            train_num = (train_num - mean) / std
            test_num = (test_num - mean) / std
        elif spec.numeric_scaling == "robust":
            med = train_num.median()
            q1 = train_num.quantile(0.25)
            q3 = train_num.quantile(0.75)
            iqr = (q3 - q1).replace(0.0, 1.0).fillna(1.0)
            train_num = (train_num - med) / iqr
            test_num = (test_num - med) / iqr
        train_blocks.append(train_num.to_numpy(dtype=float))
        test_blocks.append(test_num.to_numpy(dtype=float))
        feature_names.extend([f"num::{c}" for c in numeric_cols])

    if categorical_cols:
        if spec.categorical_encoding == "ordinal":
            train_cat_df = train_df[categorical_cols].fillna("__missing__").astype(str)
            test_cat_df = test_df[categorical_cols].fillna("__missing__").astype(str)
            train_arr = np.zeros((len(train_cat_df), len(categorical_cols)), dtype=float)
            test_arr = np.zeros((len(test_cat_df), len(categorical_cols)), dtype=float)
            for j, col in enumerate(categorical_cols):
                vocab = {v: i for i, v in enumerate(sorted(train_cat_df[col].unique()))}
                train_arr[:, j] = train_cat_df[col].map(vocab).fillna(-1).to_numpy(dtype=float)
                test_arr[:, j] = test_cat_df[col].map(vocab).fillna(-1).to_numpy(dtype=float)
                feature_names.append(f"cat::{col}")
            train_blocks.append(train_arr)
            test_blocks.append(test_arr)
        else:
            train_dummies = pd.get_dummies(train_df[categorical_cols].fillna("__missing__").astype(str), prefix=categorical_cols, dummy_na=False)
            test_dummies = pd.get_dummies(test_df[categorical_cols].fillna("__missing__").astype(str), prefix=categorical_cols, dummy_na=False)
            test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
            train_blocks.append(train_dummies.to_numpy(dtype=float))
            test_blocks.append(test_dummies.to_numpy(dtype=float))
            feature_names.extend([f"cat::{c}" for c in train_dummies.columns])

    if not train_blocks:
        return None
    train_mat = np.concatenate(train_blocks, axis=1) if len(train_blocks) > 1 else train_blocks[0]
    test_mat = np.concatenate(test_blocks, axis=1) if len(test_blocks) > 1 else test_blocks[0]
    train_mat = np.nan_to_num(train_mat, nan=0.0, posinf=0.0, neginf=0.0)
    test_mat = np.nan_to_num(test_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return train_mat, test_mat, feature_names


def _maybe_fuse(X_train, X_test, spec: TextPipelineSpec, profile: TextProfile, df: pd.DataFrame, train_idx, test_idx, allow_fusion: bool):
    if not allow_fusion or spec.fusion_strategy == "text_only":
        return X_train, X_test, False
    extra = _build_tabular_features(df, profile, spec, train_idx, test_idx)
    if extra is None:
        return X_train, X_test, False
    train_mat, test_mat, _ = extra
    train_sp = csr_matrix(train_mat)
    test_sp = csr_matrix(test_mat)
    fused_train = sp_hstack([X_train, train_sp]).tocsr()
    fused_test = sp_hstack([X_test, test_sp]).tocsr()
    return fused_train, fused_test, True


def _evaluate_single_label(spec, df, profile, task_type, metric_priority, text_col, label_col, model_defs=None, allow_fusion=False):
    texts = _preprocess_series(df[text_col], spec)
    labels = df[label_col].fillna("").astype(str).to_numpy()
    keep = np.asarray([bool(t.strip()) and bool(y.strip()) for t, y in zip(texts, labels)])
    df_kept = df[keep].reset_index(drop=True)
    texts = [t for t, k in zip(texts, keep) if k]
    labels = labels[keep]
    if len(set(labels)) < 2 or len(labels) < 3:
        raise ValueError("At least two labels and three non-empty samples are required for supervised text classification.")
    train_idx, test_idx = _split(texts, labels)
    models = model_defs or [
        ("tfidf_logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced" if spec.imbalance == "class_weight" else None)),
        ("tfidf_linear_svc", LinearSVC(class_weight="balanced" if spec.imbalance == "class_weight" else None)),
        ("tfidf_multinomial_nb", MultinomialNB()),
    ]
    per_model = {}
    fusion_used = False
    for name, clf in models:
        vec = _vectorizer(spec)
        X_train_text = vec.fit_transform([texts[i] for i in train_idx])
        X_test_text = vec.transform([texts[i] for i in test_idx])
        X_train, X_test, did_fuse = _maybe_fuse(X_train_text, X_test_text, spec, profile, df_kept, train_idx, test_idx, allow_fusion)
        if did_fuse and isinstance(clf, MultinomialNB):
            X_train = abs(X_train)
            X_test = abs(X_test)
        fusion_used = fusion_used or did_fuse
        clf.fit(X_train, labels[train_idx])
        pred = clf.predict(X_test)
        per_model[name + ("_fused" if did_fuse else "")] = _classification_metrics(labels[test_idx], pred)
    raw = {k: _safe_mean([m[k] for m in per_model.values()]) for k in next(iter(per_model.values())).keys()}
    std = {f"{k}_std": _safe_std([m[k] for m in per_model.values()]) for k in raw}
    return raw, std, per_model, len(set(labels)), len(models), fusion_used


def _evaluate_classification_single(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    raw, std, per_model, n_classes, n_models, fused = _evaluate_single_label(spec, df, profile, task_type, metric_priority, cols["text"], cols["label"], allow_fusion=True)
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "macro_f1")
    summary = f"TF-IDF text classifiers evaluated {n_classes} class(es)."
    if fused:
        summary += " Fused with auxiliary tabular features."
    if note:
        summary += " " + note
    family = "tfidf + tabular fusion classifier" if fused else "tfidf + shallow text classifier"
    return raw, std, per_model, selected, {"model_family": family, "models": list(per_model.keys()), "fusion_used": fused}, "supervised", summary, n_models


def _evaluate_multilabel(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    texts = _preprocess_series(df[cols["text"]], spec)
    if cols.get("labels"):
        y_lists = [_parse_list(v) for v in df[cols["labels"]]]
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(y_lists)
    else:
        binary = cols.get("binary_label_columns") or []
        Y = df[binary].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).to_numpy()
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    valid_cols = [i for i in range(Y.shape[1]) if 0 < int(Y[:, i].sum()) < Y.shape[0]]
    if valid_cols:
        Y = Y[:, valid_cols]
    if Y.shape[1] < 1 or len(texts) < 3:
        raise ValueError("Multi-label evaluation requires at least one non-constant label and three samples.")
    train_idx, test_idx = _split(texts, Y, multilabel=True)
    models = [
        ("tfidf_ovr_logistic_regression", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced" if spec.imbalance == "class_weight" else None))),
        ("tfidf_ovr_linear_svc", OneVsRestClassifier(LinearSVC(class_weight="balanced" if spec.imbalance == "class_weight" else None))),
    ]
    per_model = {}
    fusion_used = False
    for name, clf in models:
        vec = _vectorizer(spec)
        X_train_text = vec.fit_transform([texts[i] for i in train_idx])
        X_test_text = vec.transform([texts[i] for i in test_idx])
        X_train, X_test, did_fuse = _maybe_fuse(X_train_text, X_test_text, spec, profile, df.reset_index(drop=True), train_idx, test_idx, allow_fusion=True)
        fusion_used = fusion_used or did_fuse
        clf.fit(X_train, Y[train_idx])
        pred = clf.predict(X_test)
        per_model[name + ("_fused" if did_fuse else "")] = _classification_metrics(Y[test_idx], pred, multilabel=True)
    raw = {k: _safe_mean([m[k] for m in per_model.values()]) for k in next(iter(per_model.values())).keys()}
    std = {f"{k}_std": _safe_std([m[k] for m in per_model.values()]) for k in raw}
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "micro_f1")
    summary = "TF-IDF binary relevance multi-label classifiers evaluated label assignments."
    if fusion_used:
        summary += " Fused with auxiliary tabular features."
    if note:
        summary += " " + note
    family = "tfidf + tabular fusion multi-label classifier" if fusion_used else "tfidf + one-vs-rest multi-label classifier"
    return raw, std, per_model, selected, {"model_family": family, "models": list(per_model.keys()), "fusion_used": fusion_used}, "supervised", summary, len(models)


def _parse_entities(text: str, value) -> List[Tuple[str, str]]:
    s = str(value).strip()
    ents = []
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        for ent in parsed:
            if isinstance(ent, dict):
                start, end, label = ent.get("start"), ent.get("end"), ent.get("label")
                if start is not None and end is not None and label is not None:
                    ents.append((str(text)[int(start):int(end)].lower(), str(label)))
            elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                start, end, label = ent[0], ent[1], ent[2]
                ents.append((str(text)[int(start):int(end)].lower(), str(label)))
    else:
        tags = s.split()
        words = str(text).split()
        current = []
        current_label = ""
        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                if current and current_label:
                    ents.append((" ".join(current).lower(), current_label))
                current = [word]
                current_label = tag[2:]
            elif tag.startswith("I-") and current and tag[2:] == current_label:
                current.append(word)
            else:
                if current and current_label:
                    ents.append((" ".join(current).lower(), current_label))
                current = []
                current_label = ""
        if current and current_label:
            ents.append((" ".join(current).lower(), current_label))
    return ents


def _evaluate_ner(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    texts = _preprocess_series(df[cols["text"]], spec, preserve_alignment=True)
    gold = [_parse_entities(text, val) for text, val in zip(texts, df[cols["entities"]])]
    train_idx, test_idx = _split(texts, np.asarray([len(g) for g in gold]))
    lexicon = defaultdict(set)
    for i in train_idx:
        for surface, label in gold[i]:
            if surface:
                lexicon[label].add(surface)
    tp = fp = fn = 0
    for i in test_idx:
        pred = set()
        lower = texts[i].lower()
        for label, values in lexicon.items():
            for surface in values:
                if surface and surface in lower:
                    pred.add((surface, label))
        truth = set(gold[i])
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    raw = {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1, "token_f1": f1}
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "entity_f1")
    summary = "Explicit fallback NER evaluation used a train/test entity surface lexicon baseline and preserved text alignment."
    if note:
        summary += " " + note
    return raw, {}, {"entity_lexicon_baseline": raw}, selected, {"model_family": "rule/token baseline fallback", "models": ["entity_lexicon_baseline"], "baselines": ["entity_surface_lexicon"], "fusion_used": False}, "proxy", summary, 1




def _cosine_scores(texts_a: List[str], texts_b: List[str], spec: TextPipelineSpec) -> np.ndarray:
    vec = _vectorizer(spec)
    all_texts = texts_a + texts_b
    X = vec.fit_transform(all_texts)
    A = X[:len(texts_a)]
    B = X[len(texts_a):]
    num = np.asarray(A.multiply(B).sum(axis=1)).reshape(-1)
    den = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).reshape(-1)) * np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).reshape(-1))
    return np.nan_to_num(num / np.maximum(den, 1e-12))


def _evaluate_similarity(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    a = _preprocess_series(df[cols["text_a"]], spec)
    b = _preprocess_series(df[cols["text_b"]], spec)
    y = pd.to_numeric(df[cols["similarity"]], errors="coerce")
    keep = y.notna().to_numpy()
    scores = _cosine_scores([x for x, k in zip(a, keep) if k], [x for x, k in zip(b, keep) if k], spec)
    truth = y[keep].to_numpy(dtype=float)
    if len(truth) < 2:
        raise ValueError("Semantic similarity needs at least two scored pairs.")
    sp = spearmanr(truth, scores).correlation
    pr = pearsonr(truth, scores)[0] if len(set(truth)) > 1 else 0.0
    raw = {"spearman": float(0.0 if np.isnan(sp) else sp), "pearson": float(0.0 if np.isnan(pr) else pr)}
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "spearman")
    summary = "Semantic similarity used TF-IDF cosine similarity over text pairs."
    if note:
        summary += " " + note
    return raw, {}, {"tfidf_cosine_similarity": raw}, selected, {"model_family": "tfidf cosine similarity baseline", "models": ["tfidf_cosine_similarity"], "baselines": ["TFIDF_cosine"], "mode": "scored_pairs", "fusion_used": False}, "supervised", summary, 1


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _rouge_scores(pred: str, ref: str) -> Dict[str, float]:
    pt = pred.lower().split()
    rt = ref.lower().split()
    def f1_n(n):
        p = _ngrams(pt, n)
        r = _ngrams(rt, n)
        overlap = sum((p & r).values())
        prec = overlap / max(sum(p.values()), 1)
        rec = overlap / max(sum(r.values()), 1)
        return 2 * prec * rec / max(prec + rec, 1e-12)
    lcs = SequenceMatcher(None, pt, rt).find_longest_match(0, len(pt), 0, len(rt)).size
    prec_l = lcs / max(len(pt), 1)
    rec_l = lcs / max(len(rt), 1)
    return {"rouge1": f1_n(1), "rouge2": f1_n(2), "rouge_l": 2 * prec_l * rec_l / max(prec_l + rec_l, 1e-12)}




def _evaluate_summarization(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    raws = []
    for source, ref in zip(df[cols["source_text"]].fillna("").astype(str), df[cols["summary"]].fillna("").astype(str)):
        words = clean_text_value(source, spec, preserve_alignment=True).split()
        pred = " ".join(words[:min(75, max(20, len(words) // 4))])
        raws.append(_rouge_scores(pred, ref))
    raw = {k: _safe_mean([x[k] for x in raws]) for k in ["rouge1", "rouge2", "rouge_l"]}
    raw["bertscore"] = raw["rouge_l"]
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "rouge_l")
    summary = "Summarization used an explicit lead-N extractive baseline."
    if note:
        summary += " " + note
    return raw, {}, {"lead_n_extractive_baseline": raw}, selected, {"model_family": "extractive summarization baseline", "models": ["lead_n_extractive_baseline"], "baselines": ["lead_n_extractor", "rouge_overlap"], "fusion_used": False}, "proxy", summary, 1


def _token_f1(pred: str, ref: str) -> float:
    p = Counter(pred.lower().split())
    r = Counter(ref.lower().split())
    overlap = sum((p & r).values())
    precision = overlap / max(sum(p.values()), 1)
    recall = overlap / max(sum(r.values()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _best_sentence(context: str, question: str) -> str:
    sentences = [s.strip() for s in re_split_sentences(context) if s.strip()]
    if not sentences:
        return context
    q = set(question.lower().split())
    return max(sentences, key=lambda s: len(q & set(s.lower().split())))


def re_split_sentences(text: str) -> List[str]:
    import re
    return re.split(r"(?<=[.!?])\s+", str(text))


def _evaluate_qa(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    em, f1s = [], []
    for _, row in df.iterrows():
        context = str(row.get(cols["context"], ""))
        question = str(row.get(cols["question"], ""))
        answer = str(row.get(cols["answer"], ""))
        pred = _best_sentence(context, question)
        em.append(float(pred.strip().lower() == answer.strip().lower()))
        f1s.append(_token_f1(pred, answer))
    raw = {"exact_match": _safe_mean(em), "token_f1": _safe_mean(f1s)}
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "token_f1")
    summary = "Question answering used an explicit lexical overlap span/sentence baseline."
    if note:
        summary += " " + note
    return raw, {}, {"lexical_overlap_span_baseline": raw}, selected, {"model_family": "extractive QA fallback", "models": ["lexical_overlap_span_baseline"], "baselines": ["lexical_overlap_sentence"], "fusion_used": False}, "proxy", summary, 1




def _evaluate_topic_modeling(spec, df, profile, task_type, metric_priority):
    cols = profile.resolved_columns
    texts = _preprocess_series(df[cols["text"]], spec)
    n_topics = min(max(2, int(math.sqrt(max(len(texts), 2)))), 10)
    vec = _vectorizer(spec)
    X = vec.fit_transform(texts)
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("Topic modeling requires at least three non-empty texts with a usable vocabulary.")
    n_components = max(2, min(n_topics, X.shape[1], X.shape[0] - 1))
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=300)
    W = nmf.fit_transform(X)
    labels = np.argmax(W, axis=1)
    terms = np.asarray(vec.get_feature_names_out())
    top_words = []
    for topic in nmf.components_:
        top_words.extend(terms[np.argsort(topic)[::-1][:10]].tolist())
    diversity = len(set(top_words)) / max(len(top_words), 1)
    coherence = float(np.mean(np.max(W, axis=1) / np.maximum(np.sum(W, axis=1), 1e-12)))
    sil = silhouette_score(W, labels) if len(set(labels)) > 1 and len(labels) > len(set(labels)) else 0.0
    raw = {"coherence": _clamp_01(coherence), "topic_diversity": _clamp_01(diversity), "silhouette": float(sil)}
    if cols.get("label") and profile.label_distribution:
        y = df[cols["label"]].fillna("").astype(str).to_numpy()
        raw["nmi"] = normalized_mutual_info_score(y, labels)
        raw["ari"] = adjusted_rand_score(y, labels)
    selected, note = _resolve_metric(task_type, metric_priority, raw.keys(), "coherence")
    summary = "Topic modeling used an NMF topic model over TF-IDF features."
    if note:
        summary += " " + note
    return raw, {}, {"nmf_topic_model": raw}, selected, {"model_family": "NMF topic model", "models": ["nmf_topic_model"], "baselines": ["NMF"], "n_topics": int(nmf.n_components), "fusion_used": False}, "unsupervised", summary, 1


def evaluate_pipeline(spec: TextPipelineSpec, df: pd.DataFrame, profile: TextProfile, task_type: str, metric_priority: str) -> Dict:
    start = time.time()
    task = normalize_task_type(task_type)
    try:
        if task == "classification_single":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_classification_single(spec, df, profile, task, metric_priority)
        elif task == "classification_multi":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_multilabel(spec, df, profile, task, metric_priority)
        elif task == "ner":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_ner(spec, df, profile, task, metric_priority)
        elif task == "semantic_similarity":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_similarity(spec, df, profile, task, metric_priority)
        elif task == "summarization":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_summarization(spec, df, profile, task, metric_priority)
        elif task == "question_answering":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_qa(spec, df, profile, task, metric_priority)
        elif task == "topic_modeling":
            raw, std, model_scores, selected, details, mode, summary, n_models = _evaluate_topic_modeling(spec, df, profile, task, metric_priority)
        else:
            return _failed_result(spec, task, metric_priority, f"Unsupported text task type: {task}", time.time() - start)
        return _make_result(spec, task, metric_priority, selected, raw, model_scores, details, mode, summary, time.time() - start, metrics_std=std, n_splits=1, n_models=n_models)
    except Exception as exc:
        return _failed_result(spec, task, metric_priority, str(exc), time.time() - start)
