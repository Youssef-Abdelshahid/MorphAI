from dataclasses import dataclass
from pathlib import Path

BINARY_CLASSIFICATION_METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
MULTICLASS_CLASSIFICATION_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
]
MULTILABEL_CLASSIFICATION_METRICS = [
    "micro_f1",
    "macro_f1",
    "hamming_loss",
    "subset_accuracy",
]
REGRESSION_METRICS = ["mae", "rmse", "r2"]
ORDINAL_REGRESSION_METRICS = ["quadratic_weighted_kappa", "mae", "accuracy"]
RANKING_METRICS = ["ndcg", "spearman", "kendall_tau"]
TIME_SERIES_METRICS = ["mae", "rmse", "smape", "mape"]
CLUSTERING_METRICS = ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]
ANOMALY_METRICS = [
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "proxy_score",
    "score_separation",
    "stability",
    "contamination_consistency",
]
DIMENSIONALITY_REDUCTION_METRICS = [
    "downstream_score",
    "explained_variance_ratio",
    "trustworthiness",
    "reconstruction_error",
]
ASSOCIATION_RULE_METRICS = ["rule_quality", "support", "confidence", "lift", "coverage", "number_of_rules"]

VALID_TASK_TYPES = [
    "binary",
    "multiclass",
    "multilabel",
    "regression",
    "ordinal",
    "ranking",
    "time_series",
    "clustering",
    "anomaly",
    "dimensionality_reduction",
    "association_rules",
]
SUPPORTED_TASK_TYPES = list(VALID_TASK_TYPES)

_TASK_FAMILIES = {
    "binary": "classification",
    "multiclass": "classification",
    "multilabel": "classification",
    "regression": "regression",
    "ordinal": "ordinal",
    "ranking": "ranking",
    "time_series": "time_series",
    "clustering": "clustering",
    "anomaly": "anomaly",
    "dimensionality_reduction": "dimensionality_reduction",
    "association_rules": "association_rules",
}

_TASK_METRICS = {
    "binary": BINARY_CLASSIFICATION_METRICS,
    "multiclass": MULTICLASS_CLASSIFICATION_METRICS,
    "multilabel": MULTILABEL_CLASSIFICATION_METRICS,
    "regression": REGRESSION_METRICS,
    "ordinal": ORDINAL_REGRESSION_METRICS,
    "ranking": RANKING_METRICS,
    "time_series": TIME_SERIES_METRICS,
    "clustering": CLUSTERING_METRICS,
    "anomaly": ANOMALY_METRICS,
    "dimensionality_reduction": DIMENSIONALITY_REDUCTION_METRICS,
    "association_rules": ASSOCIATION_RULE_METRICS,
}

_DEFAULT_METRICS = {
    "binary": "f1",
    "multiclass": "macro_f1",
    "multilabel": "micro_f1",
    "regression": "r2",
    "ordinal": "quadratic_weighted_kappa",
    "ranking": "ndcg",
    "time_series": "rmse",
    "clustering": "silhouette_score",
    "anomaly": "proxy_score",
    "dimensionality_reduction": "explained_variance_ratio",
    "association_rules": "rule_quality",
}

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "micro_f1": "Micro F1",
    "hamming_loss": "Hamming loss",
    "subset_accuracy": "Subset accuracy",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R2",
    "quadratic_weighted_kappa": "Quadratic weighted kappa",
    "ndcg": "NDCG",
    "spearman": "Spearman correlation",
    "kendall_tau": "Kendall tau",
    "smape": "SMAPE",
    "mape": "MAPE",
    "silhouette_score": "Silhouette score",
    "davies_bouldin_score": "Davies-Bouldin score",
    "calinski_harabasz_score": "Calinski-Harabasz score",
    "proxy_score": "Proxy score",
    "score_separation": "Score separation",
    "stability": "Stability",
    "contamination_consistency": "Contamination consistency",
    "downstream_score": "Downstream score",
    "explained_variance_ratio": "Explained variance ratio",
    "trustworthiness": "Trustworthiness",
    "reconstruction_error": "Reconstruction error",
    "rule_quality": "Rule quality",
    "support": "Support",
    "confidence": "Confidence",
    "lift": "Lift",
    "coverage": "Coverage",
    "number_of_rules": "Number of rules",
}

_FE_BUDGET_NORM = {
    "": "moderate",
    "Minimal (raw features only)": "minimal",
    "Light (basic transforms)": "light",
    "Moderate (interactions + encoding)": "moderate",
    "Heavy (full feature engineering)": "heavy",
}

_DATA_QUALITY_NORM = {
    "": "unknown",
    "Clean / well-curated": "clean",
    "Mostly clean (minor issues)": "mostly_clean",
    "Noisy / real-world collection": "noisy",
    "Mixed quality": "mixed",
    "Unknown": "unknown",
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
class Config:
    data_path: Path
    target: str
    task_type: str
    metric: str = ""
    domain: str = ""
    constraints: str = ""
    notes: str = ""
    modality: str = "CSV / Tabular"
    fe_budget: str = ""
    data_quality: str = ""

    @property
    def supervision(self) -> str:
        return "unsupervised" if self.task_family in {
            "clustering",
            "anomaly",
            "dimensionality_reduction",
            "association_rules",
        } else "supervised"

    @property
    def task_family(self) -> str:
        return task_family(self.task_type)

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    @property
    def fe_budget_norm(self) -> str:
        return _FE_BUDGET_NORM.get(self.fe_budget, "moderate")

    @property
    def data_quality_norm(self) -> str:
        return _DATA_QUALITY_NORM.get(self.data_quality, "unknown")

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
            "fe_budget": self.fe_budget,
            "fe_budget_norm": self.fe_budget_norm,
            "data_quality": self.data_quality,
            "data_quality_norm": self.data_quality_norm,
            "supervision": self.supervision,
        }
