from dataclasses import dataclass


@dataclass
class AudioPipelineSpec:
    target_sample_rate: int
    mono: bool
    trim_silence: bool
    duration_strategy: str
    loudness_normalization: str
    noise_filter: str
    clipping_handling: str
    feature_representation: str
    augmentation: str
    imbalance: str = "none"

    def to_dict(self) -> dict:
        return {
            "target_sample_rate": self.target_sample_rate,
            "mono": self.mono,
            "trim_silence": self.trim_silence,
            "duration_strategy": self.duration_strategy,
            "loudness_normalization": self.loudness_normalization,
            "noise_filter": self.noise_filter,
            "clipping_handling": self.clipping_handling,
            "feature_representation": self.feature_representation,
            "augmentation": self.augmentation,
            "imbalance": self.imbalance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AudioPipelineSpec":
        return cls(
            target_sample_rate=int(d.get("target_sample_rate", 16000)),
            mono=bool(d.get("mono", True)),
            trim_silence=bool(d.get("trim_silence", False)),
            duration_strategy=d.get("duration_strategy", "preserve"),
            loudness_normalization=d.get("loudness_normalization", "rms"),
            noise_filter=d.get("noise_filter", "none"),
            clipping_handling=d.get("clipping_handling", "none"),
            feature_representation=d.get("feature_representation", "mfcc"),
            augmentation=d.get("augmentation", "none"),
            imbalance=d.get("imbalance", "none"),
        )

    def name(self) -> str:
        parts = [
            f"sr={self.target_sample_rate if self.target_sample_rate else 'native'}",
            f"ch={'mono' if self.mono else 'native'}",
            f"feat={self.feature_representation}",
            f"dur={self.duration_strategy}",
            f"norm={self.loudness_normalization}",
        ]
        if self.trim_silence:
            parts.append("trim")
        if self.noise_filter != "none":
            parts.append(f"noise={self.noise_filter}")
        if self.clipping_handling != "none":
            parts.append(f"clip={self.clipping_handling}")
        if self.augmentation != "none":
            parts.append(f"aug={self.augmentation}")
        if self.imbalance != "none":
            parts.append(f"imb={self.imbalance}")
        return " | ".join(parts)

    def complexity_score(self) -> int:
        score = 0
        if self.target_sample_rate and self.target_sample_rate >= 44100:
            score += 2
        elif self.target_sample_rate:
            score += 1
        if self.trim_silence:
            score += 1
        if self.loudness_normalization != "none":
            score += 1
        if self.noise_filter != "none":
            score += 1
        if self.clipping_handling != "none":
            score += 1
        if self.feature_representation in {"mel_spectrogram", "log_mel_spectrogram"}:
            score += 1
        if self.augmentation != "none":
            score += 1
        if self.imbalance != "none":
            score += 1
        return score
