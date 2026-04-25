from typing import Any, Dict, List, Optional, Tuple

from .preprocessing import AudioPipelineSpec
from .profiler import AudioProfile

_MAX_PIPELINES = 12


def _deduplicate(pipelines: List[AudioPipelineSpec]) -> List[AudioPipelineSpec]:
    seen = set()
    unique = []
    for spec in pipelines:
        key = str(sorted(spec.to_dict().items()))
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def _matches_bad_pattern(spec: AudioPipelineSpec, bad_specs: List[AudioPipelineSpec]) -> bool:
    for bad in bad_specs:
        if (
            spec.target_sample_rate == bad.target_sample_rate
            and spec.feature_representation == bad.feature_representation
            and spec.noise_filter == bad.noise_filter
            and spec.loudness_normalization == bad.loudness_normalization
        ):
            return True
    return False


def _apply_constraints(spec: AudioPipelineSpec, constraints: List[str]) -> AudioPipelineSpec:
    d = spec.to_dict()
    if "no_resampling" in constraints or "preserve_sample_rate" in constraints:
        d["target_sample_rate"] = 0
    if "no_normalization" in constraints:
        d["loudness_normalization"] = "none"
    if "no_augmentation" in constraints:
        d["augmentation"] = "none"
    if "preserve_duration" in constraints:
        d["duration_strategy"] = "preserve"
        d["trim_silence"] = False
    if "mono_only" in constraints:
        d["mono"] = True
    if "no_noise_reduction" in constraints:
        d["noise_filter"] = "none"
    if "no_silence_removal" in constraints:
        d["trim_silence"] = False
    return AudioPipelineSpec.from_dict(d)


def generate_pipelines(
    profile: AudioProfile,
    good_cases: Optional[List[Dict[str, Any]]] = None,
    bad_cases: Optional[List[Dict[str, Any]]] = None,
    meta_learner: Any = None,
    task_context: Optional[Dict[str, Any]] = None,
    profile_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[List[AudioPipelineSpec], List[str]]:
    candidates: List[AudioPipelineSpec] = []
    messages: List[str] = []
    tc = task_context or {}
    constraints = tc.get("active_constraints") or []
    task_type = tc.get("task_type", "classification")
    classification_like = task_type in {"classification", "speaker_recognition", "sound_event_detection", "anomaly"}
    speech_like = task_type in {"asr", "speaker_recognition", "speaker_diarization", "vad", "noise_suppression"}
    target_sr = 16000 if speech_like else 22050
    if profile.sample_rate_distribution:
        native_rates = [int(k) for k in profile.sample_rate_distribution if str(k).isdigit()]
        if native_rates and len(set(native_rates)) == 1:
            target_sr = native_rates[0] if native_rates[0] <= 48000 else target_sr
    use_trim = profile.silence_ratio > 0.05 and task_type not in {"asr", "sound_event_detection", "speaker_diarization", "noise_suppression"}
    use_noise = profile.estimated_noise_ratio > 0.25 and task_type not in {"speaker_recognition"}
    use_clip = profile.clipping_ratio > 0.01
    imbalance = "oversample" if classification_like and profile.imbalance_ratio > 1.5 else "none"

    baseline = AudioPipelineSpec(16000, True, False, "preserve", "rms", "none", "none", "mfcc", "none", "none")
    candidates.append(baseline)
    candidates.append(AudioPipelineSpec(target_sr, True, use_trim, "pad_or_trim", "rms", "highpass" if use_noise else "none", "soft_limit" if use_clip else "none", "mfcc", "none", imbalance))
    candidates.append(AudioPipelineSpec(target_sr, True, False, "preserve", "peak", "none", "none", "log_mel_spectrogram", "none", imbalance))
    candidates.append(AudioPipelineSpec(22050, True, use_trim, "pad_or_trim", "rms", "spectral_gate" if use_noise else "none", "soft_limit" if use_clip else "none", "mel_spectrogram", "time_shift", imbalance))
    candidates.append(AudioPipelineSpec(0, False, False, "preserve", "none", "none", "none", "raw_waveform", "none", "none"))
    candidates.append(AudioPipelineSpec(16000, True, False, "preserve", "rms", "highpass" if use_noise else "none", "none", "mfcc", "background_noise", imbalance))
    if task_type == "asr":
        candidates.append(AudioPipelineSpec(16000, True, False, "preserve", "rms", "none", "none", "log_mel_spectrogram", "none", "none"))
    if task_type == "speaker_recognition":
        candidates.append(AudioPipelineSpec(16000, True, False, "preserve", "none", "none", "none", "mfcc", "none", imbalance))
    if task_type == "sound_event_detection":
        candidates.append(AudioPipelineSpec(22050, True, False, "preserve", "rms", "none", "none", "log_mel_spectrogram", "time_shift", imbalance))
    if task_type == "noise_suppression":
        candidates.append(AudioPipelineSpec(16000, True, False, "preserve", "rms", "spectral_gate", "soft_limit", "log_mel_spectrogram", "none", "none"))

    if bad_cases:
        bad_specs = []
        for case in bad_cases:
            d = case.get("best_pipeline")
            if d:
                try:
                    bad_specs.append(AudioPipelineSpec.from_dict(d))
                except Exception:
                    pass
        if bad_specs:
            rest = candidates[1:]
            filtered = [c for c in rest if not _matches_bad_pattern(c, bad_specs)]
            skipped = len(rest) - len(filtered)
            candidates = [baseline] + filtered
            if skipped:
                messages.append(f"Memory (avoidance): skipped {skipped} candidate(s) matching poor past audio pipeline pattern(s).")

    if good_cases:
        injected = 0
        for case in good_cases[:3]:
            d = case.get("best_pipeline")
            if d:
                try:
                    candidates.append(AudioPipelineSpec.from_dict(d))
                    injected += 1
                except Exception:
                    pass
        if injected:
            messages.append(f"Memory (positive): injected {injected} pipeline(s) from good similar audio run(s).")

    if constraints:
        candidates = [_apply_constraints(c, constraints) for c in candidates]
        messages.append("Constraints applied to audio candidates.")

    candidates = _deduplicate(candidates)[:_MAX_PIPELINES]
    if meta_learner is not None and task_context is not None and profile_summary is not None:
        try:
            reordered, ml_msgs = meta_learner.rank_candidates(candidates, task_context, profile_summary)
            if len(reordered) == len(candidates):
                candidates = reordered
                messages.extend(ml_msgs)
        except Exception:
            pass
    return candidates, messages
