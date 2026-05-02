from dataclasses import dataclass
import re
import string
import unicodedata


@dataclass
class TextPipelineSpec:
    lowercase: bool
    clean_urls_emails_html: bool
    emoji_handling: str
    punctuation_handling: str
    number_normalization: str
    whitespace_normalization: bool
    stopword_removal: bool
    normalization_strategy: str
    tokenization_strategy: str
    max_sequence_length: int
    min_df: int
    representation: str
    imbalance: str = "none"
    fusion_strategy: str = "text_only"
    numeric_imputation: str = "mean"
    numeric_scaling: str = "standard"
    categorical_encoding: str = "onehot"

    def to_dict(self) -> dict:
        return {
            "lowercase": self.lowercase,
            "clean_urls_emails_html": self.clean_urls_emails_html,
            "emoji_handling": self.emoji_handling,
            "punctuation_handling": self.punctuation_handling,
            "number_normalization": self.number_normalization,
            "whitespace_normalization": self.whitespace_normalization,
            "stopword_removal": self.stopword_removal,
            "normalization_strategy": self.normalization_strategy,
            "tokenization_strategy": self.tokenization_strategy,
            "max_sequence_length": self.max_sequence_length,
            "min_df": self.min_df,
            "representation": self.representation,
            "imbalance": self.imbalance,
            "fusion_strategy": self.fusion_strategy,
            "numeric_imputation": self.numeric_imputation,
            "numeric_scaling": self.numeric_scaling,
            "categorical_encoding": self.categorical_encoding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextPipelineSpec":
        return cls(
            lowercase=bool(d.get("lowercase", True)),
            clean_urls_emails_html=bool(d.get("clean_urls_emails_html", True)),
            emoji_handling=d.get("emoji_handling", "remove"),
            punctuation_handling=d.get("punctuation_handling", "keep"),
            number_normalization=d.get("number_normalization", "keep"),
            whitespace_normalization=bool(d.get("whitespace_normalization", True)),
            stopword_removal=bool(d.get("stopword_removal", False)),
            normalization_strategy=d.get("normalization_strategy", "none"),
            tokenization_strategy=d.get("tokenization_strategy", "word"),
            max_sequence_length=int(d.get("max_sequence_length", 256)),
            min_df=int(d.get("min_df", 1)),
            representation=d.get("representation", "tfidf_word"),
            imbalance=d.get("imbalance", "none"),
            fusion_strategy=d.get("fusion_strategy", "text_only"),
            numeric_imputation=d.get("numeric_imputation", "mean"),
            numeric_scaling=d.get("numeric_scaling", "standard"),
            categorical_encoding=d.get("categorical_encoding", "onehot"),
        )

    def name(self) -> str:
        parts = [
            f"case={'lower' if self.lowercase else 'preserve'}",
            f"clean={'yes' if self.clean_urls_emails_html else 'no'}",
            f"tok={self.tokenization_strategy}",
            f"repr={self.representation}",
            f"maxlen={self.max_sequence_length}",
            f"min_df={self.min_df}",
        ]
        if self.punctuation_handling != "keep":
            parts.append(f"punct={self.punctuation_handling}")
        if self.number_normalization != "keep":
            parts.append(f"num={self.number_normalization}")
        if self.stopword_removal:
            parts.append("stopwords")
        if self.normalization_strategy != "none":
            parts.append(f"norm={self.normalization_strategy}")
        if self.emoji_handling != "keep":
            parts.append(f"emoji={self.emoji_handling}")
        if self.imbalance != "none":
            parts.append(f"imb={self.imbalance}")
        if self.fusion_strategy != "text_only":
            parts.append(f"fusion={self.fusion_strategy}")
            parts.append(f"num_imp={self.numeric_imputation}")
            parts.append(f"num_sc={self.numeric_scaling}")
            parts.append(f"cat_enc={self.categorical_encoding}")
        return " | ".join(parts)

    def complexity_score(self) -> int:
        score = 0
        score += int(self.clean_urls_emails_html)
        score += int(self.lowercase)
        score += int(self.punctuation_handling != "keep")
        score += int(self.number_normalization != "keep")
        score += int(self.stopword_removal)
        score += int(self.normalization_strategy != "none")
        score += int(self.representation != "tfidf_word")
        score += int(self.imbalance != "none")
        score += int(self.fusion_strategy != "text_only")
        return score


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
}

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000027BF"
    r"\U0000FE00-\U0000FEFF"
    r"\U00010000-\U0010FFFF]",
    re.UNICODE,
)

_NONBREAKING_SPACE_RE = re.compile(
    "[   -     　]+"
)
_ZEROWIDTH_RE = re.compile(
    "[​‌‍‎‏⁠﻿­]+"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
_EXCESSIVE_PUNCT_RE = re.compile(r"([!?.,;:\-])\1{2,}")

_EMOJI_NAME_NOISE = {"symbol", "sign", "selector", "variation"}
EMOJI_STRATEGIES = {"remove", "preserve", "keep", "translate_to_text", "describe", "limit_then_translate"}
EMOJI_RUN_LIMIT = 2

_EMOJI_RUN_RE = re.compile(
    r"([\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF])\1{" + str(EMOJI_RUN_LIMIT) + r",}"
)


def _emoji_describe(match: "re.Match") -> str:
    name = unicodedata.name(match.group(0), "")
    if not name:
        return " emoji "
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name.lower()) if p and p not in _EMOJI_NAME_NOISE]
    if not parts:
        return " emoji "
    return " " + " ".join(parts) + " "


def _collapse_emoji_runs(text: str) -> str:
    return _EMOJI_RUN_RE.sub(lambda m: m.group(1) * EMOJI_RUN_LIMIT, text)


def _normalize_base(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ZEROWIDTH_RE.sub("", text)
    text = _NONBREAKING_SPACE_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    return text


def clean_text_value(value, spec: TextPipelineSpec, preserve_alignment: bool = False) -> str:
    text = "" if value is None else str(value)
    text = _normalize_base(text)
    if preserve_alignment:
        if spec.whitespace_normalization:
            text = re.sub(r"[ \t]+", " ", text).strip()
        return text
    if spec.clean_urls_emails_html:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
        text = re.sub(r"\b[\w.\-]+@[\w.\-]+\.\w+\b", " EMAIL ", text)
        text = re.sub(r"@\w+", " MENTION ", text)
        text = re.sub(r"#(\w+)", r" \1 ", text)
    text = _EXCESSIVE_PUNCT_RE.sub(r"\1", text)
    strategy = (spec.emoji_handling or "preserve").lower()
    if strategy == "remove":
        text = _EMOJI_RE.sub(" ", text)
    elif strategy in ("describe", "translate_to_text"):
        text = _EMOJI_RE.sub(_emoji_describe, text)
    elif strategy == "limit_then_translate":
        text = _collapse_emoji_runs(text)
        text = _EMOJI_RE.sub(_emoji_describe, text)
    if spec.number_normalization == "replace":
        text = re.sub(r"\d+", " NUMBER ", text)
    elif spec.number_normalization == "remove":
        text = re.sub(r"\d+", " ", text)
    if spec.lowercase:
        text = text.lower()
    if spec.punctuation_handling == "remove":
        text = text.translate(str.maketrans({p: " " for p in string.punctuation}))
    elif spec.punctuation_handling == "space":
        text = text.translate(str.maketrans({p: f" {p} " for p in string.punctuation}))
    if spec.whitespace_normalization:
        text = re.sub(r"\s+", " ", text).strip()
    if spec.stopword_removal:
        tokens = [tok for tok in text.split() if tok.lower() not in _STOPWORDS]
        text = " ".join(tokens)
    if spec.max_sequence_length and spec.max_sequence_length > 0:
        tokens = text.split()
        if len(tokens) > spec.max_sequence_length:
            text = " ".join(tokens[:spec.max_sequence_length])
    return text
