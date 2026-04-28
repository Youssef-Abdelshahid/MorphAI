from dataclasses import dataclass
import re
import string


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
        return score


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
}


def clean_text_value(value, spec: TextPipelineSpec, preserve_alignment: bool = False) -> str:
    text = "" if value is None else str(value)
    if preserve_alignment:
        if spec.whitespace_normalization:
            text = re.sub(r"\s+", " ", text).strip()
        return text
    if spec.clean_urls_emails_html:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " EMAIL ", text)
    if spec.emoji_handling == "remove":
        text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    elif spec.emoji_handling == "describe":
        text = re.sub(r"[\U00010000-\U0010ffff]", " EMOJI ", text)
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
