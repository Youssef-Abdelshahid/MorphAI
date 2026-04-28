from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

_ALIASES = {
    "text": ["text", "sentence", "content", "document", "body", "review", "message", "utterance"],
    "label": ["label", "class", "target", "category", "sentiment", "intent"],
    "labels": ["labels", "tags", "multi_labels", "multilabel", "categories"],
    "entities": ["entities", "entity_spans", "ner", "bio_tags", "tags"],
    "tokens": ["tokens", "words"],
    "pos_tags": ["pos_tags", "tags", "pos", "upos", "xpos"],
    "entity1": ["entity1", "head", "subject", "source_entity", "arg1"],
    "entity2": ["entity2", "tail", "object", "target_entity", "arg2"],
    "relation": ["relation", "relation_label", "predicate", "rel_label"],
    "text_a": ["text_a", "sentence1", "query", "question", "source_text"],
    "text_b": ["text_b", "sentence2", "document", "candidate", "target_text"],
    "similarity": ["similarity_score", "score", "label", "relevance", "is_similar"],
    "query": ["query", "search_query"],
    "document": ["document", "doc", "passage", "text"],
    "id": ["id", "doc_id", "document_id"],
    "relevance": ["relevance", "relevance_label", "is_relevant", "label"],
    "source_text": ["source_text", "source", "article", "input_text", "document"],
    "summary": ["summary", "target_summary", "reference_summary"],
    "target_text": ["target_text", "target", "translation", "completion", "output_text"],
    "source_language": ["source_language", "src_lang"],
    "target_language": ["target_language", "tgt_lang"],
    "context": ["context", "passage", "document"],
    "question": ["question", "query"],
    "answer": ["answer", "answers", "target_answer"],
    "answer_start": ["answer_start", "start", "answer_offset"],
    "prompt": ["prompt", "input", "instruction"],
    "completion": ["completion", "target_text", "response", "output"],
    "language_label": ["language", "lang", "language_label", "locale"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for alias in aliases:
        n = _norm(alias)
        if n in norm_map:
            return norm_map[n]
    for col in df.columns:
        nc = _norm(col)
        if any(_norm(alias) in nc for alias in aliases):
            return col
    return None


def binary_label_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    excluded = set(exclude or [])
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        nc = _norm(col)
        if not (nc.startswith("label") or nc.startswith("class") or nc.startswith("tag") or nc.startswith("target")):
            continue
        vals = set(str(v).strip().lower() for v in df[col].dropna().unique())
        if vals and vals.issubset({"0", "1", "true", "false", "yes", "no"}):
            cols.append(col)
    return cols


def resolve_columns(df: pd.DataFrame, task_type: str) -> Dict[str, object]:
    task = (task_type or "").strip().lower()
    c = {}
    if task in {"classification_single", "topic_modeling"}:
        c["text"] = find_column(df, _ALIASES["text"])
        c["label"] = find_column(df, _ALIASES["label"])
    elif task == "classification_multi":
        c["text"] = find_column(df, _ALIASES["text"])
        c["labels"] = find_column(df, _ALIASES["labels"])
        c["binary_label_columns"] = binary_label_columns(df, [c.get("text"), c.get("labels")])
    elif task == "ner":
        c["text"] = find_column(df, _ALIASES["text"])
        c["entities"] = find_column(df, _ALIASES["entities"])
    elif task == "pos":
        c["tokens"] = find_column(df, _ALIASES["tokens"])
        c["pos_tags"] = find_column(df, _ALIASES["pos_tags"])
        c["text"] = find_column(df, _ALIASES["text"])
    elif task == "relation_extraction":
        c["text"] = find_column(df, _ALIASES["text"])
        c["entity1"] = find_column(df, _ALIASES["entity1"])
        c["entity2"] = find_column(df, _ALIASES["entity2"])
        c["relation"] = find_column(df, _ALIASES["relation"])
    elif task == "semantic_similarity":
        c["text_a"] = find_column(df, ["text_a", "sentence1", "text1", "left_text"])
        c["text_b"] = find_column(df, ["text_b", "sentence2", "text2", "right_text"])
        c["similarity"] = find_column(df, _ALIASES["similarity"])
        c["query"] = find_column(df, _ALIASES["query"])
        c["document"] = find_column(df, _ALIASES["document"])
        c["id"] = find_column(df, _ALIASES["id"])
        c["relevance"] = find_column(df, _ALIASES["relevance"])
    elif task == "summarization":
        c["source_text"] = find_column(df, _ALIASES["source_text"])
        c["summary"] = find_column(df, _ALIASES["summary"])
    elif task == "machine_translation":
        c["source_text"] = find_column(df, _ALIASES["source_text"])
        c["target_text"] = find_column(df, _ALIASES["target_text"])
        c["source_language"] = find_column(df, _ALIASES["source_language"])
        c["target_language"] = find_column(df, _ALIASES["target_language"])
    elif task == "question_answering":
        c["context"] = find_column(df, _ALIASES["context"])
        c["question"] = find_column(df, _ALIASES["question"])
        c["answer"] = find_column(df, _ALIASES["answer"])
        c["answer_start"] = find_column(df, _ALIASES["answer_start"])
    elif task == "text_generation":
        c["prompt"] = find_column(df, _ALIASES["prompt"])
        c["completion"] = find_column(df, _ALIASES["completion"])
    elif task == "language_detection":
        c["text"] = find_column(df, _ALIASES["text"])
        c["language_label"] = find_column(df, _ALIASES["language_label"])
    return c
