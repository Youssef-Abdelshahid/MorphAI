from dataclasses import dataclass


@dataclass
class ImagePipelineSpec:
    resize: int
    color_mode: str
    normalization: str
    histogram_eq: bool
    denoise: bool
    sharpen: bool
    augment_h_flip: bool
    augment_v_flip: bool
    augment_rotation: str
    augment_color_jitter: bool
    imbalance: str = "none"

    def to_dict(self) -> dict:
        return {
            "resize":               self.resize,
            "color_mode":           self.color_mode,
            "normalization":        self.normalization,
            "histogram_eq":         self.histogram_eq,
            "denoise":              self.denoise,
            "sharpen":              self.sharpen,
            "augment_h_flip":       self.augment_h_flip,
            "augment_v_flip":       self.augment_v_flip,
            "augment_rotation":     self.augment_rotation,
            "augment_color_jitter": self.augment_color_jitter,
            "imbalance":            self.imbalance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImagePipelineSpec":
        return cls(
            resize=int(d["resize"]),
            color_mode=d["color_mode"],
            normalization=d["normalization"],
            histogram_eq=bool(d["histogram_eq"]),
            denoise=bool(d["denoise"]),
            sharpen=bool(d["sharpen"]),
            augment_h_flip=bool(d["augment_h_flip"]),
            augment_v_flip=bool(d["augment_v_flip"]),
            augment_rotation=d["augment_rotation"],
            augment_color_jitter=bool(d["augment_color_jitter"]),
            imbalance=d.get("imbalance", "none"),
        )

    def name(self) -> str:
        parts = [
            f"sz={self.resize}",
            f"clr={self.color_mode}",
            f"norm={self.normalization}",
        ]
        if self.histogram_eq:
            parts.append("heq")
        if self.denoise:
            parts.append("dns")
        if self.sharpen:
            parts.append("shp")
        if self.augment_h_flip:
            parts.append("hflip")
        if self.augment_v_flip:
            parts.append("vflip")
        if self.augment_rotation != "none":
            parts.append(f"rot={self.augment_rotation}")
        if self.augment_color_jitter:
            parts.append("jitter")
        if self.imbalance != "none":
            parts.append(f"imb={self.imbalance}")
        return " | ".join(parts)

    def complexity_score(self) -> int:
        score = 0
        if self.resize >= 128:
            score += 2
        elif self.resize >= 64:
            score += 1
        if self.histogram_eq:
            score += 1
        if self.denoise:
            score += 1
        if self.sharpen:
            score += 1
        if self.augment_h_flip:
            score += 1
        if self.augment_v_flip:
            score += 1
        if self.augment_rotation != "none":
            score += 1
        if self.augment_color_jitter:
            score += 1
        if self.imbalance == "oversample":
            score += 1
        return score

    @property
    def has_augmentation(self) -> bool:
        return (
            self.augment_h_flip
            or self.augment_v_flip
            or self.augment_rotation != "none"
            or self.augment_color_jitter
        )
