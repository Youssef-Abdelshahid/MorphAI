from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

TEXT_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]
TEXT_MULTILABEL_METRICS = ["micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"]
NER_METRICS = ["entity_f1", "entity_precision", "entity_recall", "token_f1"]
POS_METRICS = ["token_accuracy", "macro_f1", "weighted_f1"]
RELATION_METRICS = ["macro_f1", "micro_f1", "accuracy", "precision", "recall"]
SIMILARITY_PAIR_METRICS = ["spearman", "pearson"]
SUMMARIZATION_METRICS = ["rouge1", "rouge2", "rouge_l", "bertscore"]
QA_METRICS = ["exact_match", "token_f1"]
GENERATION_METRICS = ["rouge_l", "bleu", "inverse_perplexity"]
TOPIC_METRICS = ["coherence", "topic_diversity", "silhouette", "nmi", "ari"]

_TXT_TASK_BACKEND = {
    "Text classification (single-label)": "classification_single",
    "Text classification (multi-label)": "classification_multi",
    "Named entity recognition": "ner",
    "Part-of-speech tagging": "pos",
    "Relation extraction": "relation_extraction",
    "Semantic similarity / search": "semantic_similarity",
    "Text summarization": "summarization",
    "Question answering": "question_answering",
    "Text generation": "text_generation",
    "Topic modeling": "topic_modeling",
}

VALID_TASK_TYPES = list(_TXT_TASK_BACKEND.values())
SUPPORTED_TASK_TYPES = set(VALID_TASK_TYPES)

_TASK_FAMILIES = {
    "classification_single": "classification",
    "classification_multi": "classification",
    "ner": "sequence_labeling",
    "pos": "sequence_labeling",
    "relation_extraction": "information_extraction",
    "semantic_similarity": "retrieval",
    "summarization": "seq2seq",
    "question_answering": "qa",
    "text_generation": "generation",
    "topic_modeling": "topic",
}

_TASK_METRICS = {
    "classification_single": TEXT_CLASSIFICATION_METRICS,
    "classification_multi": TEXT_MULTILABEL_METRICS,
    "ner": NER_METRICS,
    "pos": POS_METRICS,
    "relation_extraction": RELATION_METRICS,
    "semantic_similarity": SIMILARITY_PAIR_METRICS,
    "summarization": SUMMARIZATION_METRICS,
    "question_answering": QA_METRICS,
    "text_generation": GENERATION_METRICS,
    "topic_modeling": TOPIC_METRICS,
}

_DEFAULT_METRICS = {
    "classification_single": "macro_f1",
    "classification_multi": "micro_f1",
    "ner": "entity_f1",
    "pos": "token_accuracy",
    "relation_extraction": "macro_f1",
    "semantic_similarity": "spearman",
    "summarization": "rouge_l",
    "question_answering": "token_f1",
    "text_generation": "rouge_l",
    "topic_modeling": "coherence",
}

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "precision": "Precision",
    "recall": "Recall",
    "micro_f1": "Micro F1",
    "hamming_loss": "Hamming loss",
    "subset_accuracy": "Subset accuracy",
    "entity_precision": "Entity precision",
    "entity_recall": "Entity recall",
    "entity_f1": "Entity F1",
    "token_f1": "Token F1",
    "token_accuracy": "Token accuracy",
    "spearman": "Spearman correlation",
    "pearson": "Pearson correlation",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "bertscore": "BERTScore",
    "bleu": "BLEU",
    "exact_match": "Exact match",
    "inverse_perplexity": "Inverse perplexity",
    "coherence": "Topic coherence",
    "topic_diversity": "Topic diversity",
    "silhouette": "Silhouette",
    "nmi": "NMI",
    "ari": "ARI",
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


SEQ_LABELING_TASKS = {"ner", "pos"}
SEQ2SEQ_TASKS = {"summarization", "question_answering", "text_generation"}
TABULAR_FUSION_COMPATIBLE = {"classification_single", "classification_multi", "relation_extraction", "semantic_similarity", "topic_modeling"}


@dataclass
class TextConfig:
    data_path: Path
    metric: str = ""
    task_type: str = "classification_single"
    domain: str = ""
    constraints: str = ""
    notes: str = ""
    modality: str = "Text"
    input_format: str = ""
    input_format_key: str = ""
    record_path: str = ""
    metadata_path: str = ""
    language: str = ""
    text_source: str = ""
    text_length: str = ""
    col_overrides: Optional[Dict[str, str]] = None
    auxiliary_feature_columns: List[str] = field(default_factory=list)
    multilabel_format: str = "single_column"
    binary_label_columns: List[str] = field(default_factory=list)

    @property
    def supervision(self) -> str:
        return "unsupervised" if normalize_task_type(self.task_type) == "topic_modeling" else "supervised"

    @property
    def task_family(self) -> str:
        return task_family(self.task_type)

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    @property
    def supports_tabular_fusion(self) -> bool:
        return normalize_task_type(self.task_type) in TABULAR_FUSION_COMPATIBLE

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
            "input_format_key": self.input_format_key,
            "record_path": self.record_path,
            "metadata_path": self.metadata_path,
            "language": self.language,
            "text_source": self.text_source,
            "text_length": self.text_length,
            "supervision": self.supervision,
            "auxiliary_feature_columns": list(self.auxiliary_feature_columns or []),
            "multilabel_format": self.multilabel_format,
        }
