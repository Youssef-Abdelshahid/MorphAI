"""
ui/pdf_exporter.py — Professional PDF report for MorphAI Preprocessing Agent.

Complete redesign: dark cover page, KPI summary cards, professional typography,
right-aligned numerics, clean section hierarchy, full-width charts.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

# ── Brand ─────────────────────────────────────────────────────────────────────
AGENT_NAME    = "MorphAI"
AGENT_TAGLINE = "Adaptive Preprocessing Agent"
_LOGO_PATH    = Path(__file__).parent / "logo" / "Morph_AI_LightMode_Logo.png"

# ── PDF colour palette ────────────────────────────────────────────────────────
# Cover / header
_C_STRIPE    = colors.HexColor("#1e3a5f")   # left accent stripe (toned-down blue)

# Section bars
_C_SEC_NAVY  = colors.HexColor("#1e3a5f")   # general sections
_C_SEC_GREEN = colors.HexColor("#14532d")   # best pipeline section
_C_SEC_PURP  = colors.HexColor("#581c87")   # config + rationale

# Body colours
_C_BODY      = colors.HexColor("#1e293b")
_C_MUTED     = colors.HexColor("#64748b")
_C_WHITE     = colors.white

# Table colours
_C_TBL_HDR  = colors.HexColor("#1e3a5f")
_C_TBL_BRD  = colors.HexColor("#cbd5e1")
_C_TBL_ALT  = colors.HexColor("#f8fafc")
_C_TBL_HIGH = colors.HexColor("#f0fdf4")
_C_TBL_TXT  = colors.HexColor("#15803d")

# KPI cards — uniform neutral design (no rainbow)
_C_KPI_BG  = colors.HexColor("#f8fafc")   # all cards same light bg
_C_KPI_VAL = colors.HexColor("#1e3a5f")   # all values: navy
_C_KPI_BRD = colors.HexColor("#e2e8f0")   # all borders: light gray

# Matplotlib palette
_M_BLUE    = "#2563eb"
_M_GREEN   = "#16a34a"
_M_PURPLE  = "#7c3aed"
_M_ORANGE  = "#ea580c"
_M_CYAN    = "#0891b2"
_M_MUTED   = "#94a3b8"
_M_PALETTE = [_M_BLUE, _M_GREEN, _M_PURPLE, _M_ORANGE, _M_CYAN, _M_MUTED]

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


# ── Matplotlib helpers ────────────────────────────────────────────────────────

def _set_rcparams() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "axes.labelsize":    13,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "figure.dpi":        120,
        "savefig.dpi":       220,
        "axes.linewidth":    1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _style_ax(ax, grid: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid, linestyle="--", linewidth=0.5, alpha=0.35, color="#94a3b8")
    ax.tick_params(axis="both", length=5, width=1.0, labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#cbd5e1")


def _save_fig(fig) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _fit_image(path: str, max_w: float, max_h: float) -> RLImage:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return RLImage(path, width=w * scale, height=h * scale)


def _fw_image(path: str, avail_w: float) -> RLImage:
    """Always render at full page width — height set by aspect ratio."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = avail_w / w
    return RLImage(path, width=avail_w, height=h * scale)


def _logo_to_png(logo_path: Path, bg_rgb: tuple = (15, 23, 42),
                 lighten: bool = False) -> Optional[str]:
    """Convert webp logo to PNG composited on bg_rgb colour.

    lighten=True: dark pixels are brightened to light blue-white so the
    logo is visible on dark backgrounds (cover page, dark header).
    """
    if not logo_path.exists():
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(str(logo_path)) as im:
            rgba = im.convert("RGBA")
        if lighten:
            pixels = list(rgba.getdata())
            new_pixels = []
            for rp, gp, bp, ap in pixels:
                if ap > 10:
                    brightness = (int(rp) + int(gp) + int(bp)) / 3
                    if brightness < 140:          # dark pixel → light blue-white
                        new_pixels.append((210, 225, 255, ap))
                    else:
                        new_pixels.append((rp, gp, bp, ap))
                else:
                    new_pixels.append((rp, gp, bp, ap))
            rgba = PILImage.new("RGBA", rgba.size)
            rgba.putdata(new_pixels)
        bg = PILImage.new("RGBA", rgba.size, (*bg_rgb, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        out = bg.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        out.save(tmp.name, format="PNG")
        return tmp.name
    except Exception:
        return None


# ── Chart generators ──────────────────────────────────────────────────────────

def _chart_pipeline_rankings(results: list, metric: str) -> Optional[str]:
    if not results:
        return None
    _set_rcparams()
    names  = [f"#{r['rank']}  {r['pipeline_name']}" for r in results]
    scores = [r["metrics"].get(metric, 0) for r in results]
    stds   = [r.get("metrics_std", {}).get(f"{metric}_std", 0) for r in results]
    # Truncate to 38 chars, then pad to equal length so all labels left-align consistently
    names  = [n if len(n) <= 38 else n[:35] + "..." for n in names]
    max_len = max(len(n) for n in names)
    names  = [n.ljust(max_len) for n in names]
    n      = len(names)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, n * 0.55 + 1.0)))
    bar_colors = [_M_GREEN if scores[i] == max(scores) else _M_BLUE for i in range(n)]
    bars = ax.barh(range(n), scores, color=bar_colors, height=0.42,
                   xerr=stds, error_kw={"ecolor": "#94a3b8", "capsize": 4, "linewidth": 1.0})
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper(), fontsize=12, labelpad=8)
    ax.set_title(f"Candidate Pipeline Rankings  ({metric.upper()})",
                 fontsize=14, fontweight="bold", pad=38)
    ax.set_xlim(0, min(1.0, max(scores) * 1.45))
    _style_ax(ax, grid="x")
    try:
        ax.bar_label(bars, labels=[f"{s:.4f}" for s in scores],
                     padding=4, fontsize=9, fontweight="bold", color="#1e293b")
    except Exception:
        pass
    best_p  = mpatches.Patch(color=_M_GREEN, label="Best pipeline")
    other_p = mpatches.Patch(color=_M_BLUE,  label="Other candidates")
    ax.legend(handles=[best_p, other_p], fontsize=11,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, framealpha=0.9, edgecolor="#cbd5e1")
    fig.tight_layout()
    return _save_fig(fig)


def _chart_metrics_overview(metrics: dict, metrics_std: dict, metric: str) -> Optional[str]:
    if not metrics:
        return None
    _set_rcparams()
    keys   = ["accuracy", "f1", "precision", "recall"]
    labels = ["Accuracy", "F1", "Precision", "Recall"]
    vals   = [metrics.get(k, 0) for k in keys]
    stds   = [metrics_std.get(f"{k}_std", 0) for k in keys]
    clrs   = [_M_GREEN if k == metric else _M_BLUE for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    bars = ax.bar(labels, vals, color=clrs, width=0.45,
                  yerr=stds, error_kw={"ecolor": "#94a3b8", "capsize": 5, "linewidth": 1.0})
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score", fontsize=13, labelpad=8)
    ax.set_title("Best Pipeline — Evaluation Metrics",
                 fontsize=15, fontweight="bold", pad=38)
    _style_ax(ax, grid="y")
    try:
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals],
                     padding=5, fontsize=12, fontweight="bold", color="#1e293b")
    except Exception:
        pass
    primary_p = mpatches.Patch(color=_M_GREEN, label=f"Primary metric ({metric.upper()})")
    other_p   = mpatches.Patch(color=_M_BLUE,  label="Other metrics")
    ax.legend(handles=[primary_p, other_p], fontsize=12,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, framealpha=0.9, edgecolor="#cbd5e1")
    fig.tight_layout()
    return _save_fig(fig)


def _chart_per_model(per_model: dict) -> Optional[str]:
    if not per_model:
        return None
    _set_rcparams()
    model_names   = list(per_model.keys())
    metric_keys   = ["accuracy", "f1", "precision", "recall"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall"]
    x       = np.arange(len(metric_keys))
    width   = 0.20
    n       = len(model_names)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    for i, mn in enumerate(model_names):
        vals = [per_model[mn].get(mk, 0) for mk in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=mn.capitalize(), color=_M_PALETTE[i % len(_M_PALETTE)],
                      alpha=0.9)
        try:
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals],
                         padding=3, fontsize=8)
        except Exception:
            pass
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel("Score", fontsize=13, labelpad=8)
    ax.set_title("Per-Model Breakdown — Best Pipeline", fontsize=15, fontweight="bold", pad=38)
    ax.legend(title=None, fontsize=12, framealpha=0.9, edgecolor="#cbd5e1",
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(model_names))
    _style_ax(ax, grid="y")
    fig.tight_layout()
    return _save_fig(fig)


def _chart_profile_issues(prof: dict) -> Optional[str]:
    _set_rcparams()
    items = [
        ("High-outlier cols",     prof.get("high_outlier_cols_count", 0)),
        ("High-skew cols",        prof.get("high_skew_cols_count", 0)),
        ("High-kurtosis cols",    prof.get("high_kurtosis_cols_count", 0)),
        ("High-cardinality cols", prof.get("high_cardinality_cols_count", 0)),
        ("Constant / near-const", prof.get("constant_cols_count", 0)),
        ("Correlated pairs",      prof.get("n_high_corr_pairs", 0)),
        ("Cols >50% missing",     prof.get("high_missing_cols_count", 0)),
        ("Duplicate rows",        prof.get("n_duplicates", 0)),
        ("Binary numeric cols",   prof.get("binary_num_cols_count", 0)),
    ]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    if max(values) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    clrs = [_M_ORANGE if v > 0 else _M_MUTED for v in values]
    bars = ax.barh(labels, values, color=clrs, height=0.42)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=13, labelpad=8)
    ax.set_title("Dataset Quality Indicators", fontsize=15, fontweight="bold", pad=14)
    _style_ax(ax, grid="x")
    try:
        ax.bar_label(bars, labels=[str(v) for v in values], padding=4, fontsize=9)
    except Exception:
        pass
    ax.margins(x=0.16)
    fig.tight_layout()
    return _save_fig(fig)


def _chart_dataset_composition(prof: dict) -> Optional[str]:
    num   = prof.get("num_cols_count", 0)
    cat   = prof.get("cat_cols_count", 0)
    total = num + cat
    if total == 0:
        return None
    _set_rcparams()
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    ax.barh(["Features"], [num], color=_M_BLUE,   label=f"Numeric  ({num} cols)",   height=0.45)
    ax.barh(["Features"], [cat], left=[num], color=_M_PURPLE,
            label=f"Categorical  ({cat} cols)", height=0.45)
    ax.set_xlim(0, total * 1.05)
    ax.set_xlabel("Number of columns", fontsize=13, labelpad=8)
    ax.set_title(f"Feature Composition  ({total} total features)", fontsize=15, fontweight="bold", pad=38)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=12, framealpha=0.9, edgecolor="#cbd5e1")
    _style_ax(ax, grid="x")
    fig.tight_layout()
    return _save_fig(fig)


def _chart_image_class_distribution(prof: dict) -> Optional[str]:
    class_counts = prof.get("class_counts", {})
    if not class_counts:
        return None
    _set_rcparams()
    labels = sorted(class_counts.keys())
    values = [class_counts[l] for l in labels]
    max_v = max(values)
    clrs = [_M_GREEN if v == max_v else _M_BLUE for v in values]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.5, max(2.5, n * 0.45 + 1.0)))
    bars = ax.barh(range(n), values, color=clrs, height=0.42)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Images", fontsize=13, labelpad=8)
    ax.set_title("Class Distribution", fontsize=15, fontweight="bold", pad=14)
    _style_ax(ax, grid="x")
    try:
        ax.bar_label(bars, labels=[str(v) for v in values], padding=4, fontsize=9)
    except Exception:
        pass
    ax.margins(x=0.16)
    fig.tight_layout()
    return _save_fig(fig)


def _chart_image_quality_flags(prof: dict) -> Optional[str]:
    items = [
        ("Low contrast",            1 if prof.get("has_low_contrast") else 0),
        ("Varied brightness",       1 if prof.get("has_varied_brightness") else 0),
        ("Varied sizes",            1 if prof.get("has_varied_sizes") else 0),
        ("Small images (<32px)",    1 if prof.get("has_small_images") else 0),
        ("Large images (>1024px)",  1 if prof.get("has_large_images") else 0),
        ("Corrupt images",          1 if prof.get("n_corrupt", 0) > 0 else 0),
    ]
    if not any(v for _, v in items):
        return None
    _set_rcparams()
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    clrs = [_M_ORANGE if v else _M_MUTED for v in values]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.barh(labels, values, color=clrs, height=0.42)
    ax.set_xlim(0, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not detected", "Detected"], fontsize=11)
    ax.invert_yaxis()
    ax.set_title("Image Quality Flags", fontsize=15, fontweight="bold", pad=14)
    _style_ax(ax, grid="x")
    ax.margins(x=0.08)
    fig.tight_layout()
    return _save_fig(fig)


def _chart_image_dimensions(prof: dict) -> Optional[str]:
    min_h = prof.get("min_height", 0)
    avg_h = prof.get("avg_height", 0)
    max_h = prof.get("max_height", 0)
    min_w = prof.get("min_width", 0)
    avg_w = prof.get("avg_width", 0)
    max_w = prof.get("max_width", 0)
    if not any([min_h, avg_h, max_h, min_w, avg_w, max_w]):
        return None
    _set_rcparams()
    x = np.arange(2)
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    bars_min = ax.bar(x - width, [min_h, min_w], width, label="Min", color=_M_MUTED, alpha=0.85)
    bars_avg = ax.bar(x,         [avg_h, avg_w], width, label="Avg", color=_M_BLUE,  alpha=0.9)
    bars_max = ax.bar(x + width, [max_h, max_w], width, label="Max", color=_M_PURPLE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Height (px)", "Width (px)"], fontsize=13)
    ax.set_ylabel("Pixels", fontsize=13, labelpad=8)
    ax.set_title("Image Dimension Statistics", fontsize=15, fontweight="bold", pad=38)
    for bars in [bars_min, bars_avg, bars_max]:
        try:
            ax.bar_label(bars, labels=[f"{b.get_height():.0f}" for b in bars],
                         padding=3, fontsize=8)
        except Exception:
            pass
    ax.legend(title=None, fontsize=12, framealpha=0.9, edgecolor="#cbd5e1",
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    _style_ax(ax, grid="y")
    fig.tight_layout()
    return _save_fig(fig)


# ── Document ──────────────────────────────────────────────────────────────────

def _readable_pipeline(name: str) -> str:
    """Convert 'num=median | cat=mode | scale=robust | ...' to human-readable form."""
    label_map = {
        "num":   "Impute",
        "cat":   "Encode",
        "scale": "Scale",
        "enc":   "Encoding",
        "pwr":   "Power Transform",
        "cl":    "Classifier",
    }
    parts   = [p.strip() for p in name.split("|")]
    readable = []
    for p in parts:
        if "=" in p:
            key, val = p.split("=", 1)
            readable.append(f"{label_map.get(key.strip(), key.strip().capitalize())}: {val.strip().capitalize()}")
        elif p:
            readable.append(label_map.get(p, p.capitalize()))
    shown  = readable[:3]
    result = "  |  ".join(shown)
    if len(readable) > 3:
        result += f"  +{len(readable) - 3}"
    return result


def export_report_pdf(report: dict, output_path: Path) -> Path:
    """
    Generate a professional PDF from a MorphAI report dict.
    Page 1 = full dark cover (drawn via canvas).
    Pages 2+ = structured content with thin dark header + footer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Two logo variants: dark bg for cover, light bg for page header
    _LOGO_LIGHT = Path(__file__).parent / "logo" / "Morph_AI_LightMode_Logo.png"
    _LOGO_DARK = Path(__file__).parent / "logo" / "Morph_AI_DarkMode_Logo.png"
    
    logo_png       = _logo_to_png(_LOGO_DARK, bg_rgb=(11, 22, 40))
    logo_light_png = _logo_to_png(_LOGO_LIGHT, bg_rgb=(248, 250, 252))
    tmp_files = [f for f in [logo_png, logo_light_png] if f]

    # ── Extract data ──────────────────────────────────────────────────────
    cfg     = report.get("config", {})
    prof    = report.get("profile_summary", {})
    best    = report.get("best_pipeline", {})
    results = report.get("results", [])
    metric  = cfg.get("metric", "f1")
    m       = best.get("metrics", {})
    std     = best.get("metrics_std", {})
    pmt     = best.get("per_model_metrics", {})
    expl    = report.get("explanation", "")
    tc      = report.get("task_context", {})
    lrn     = report.get("learning_summary", {})

    ds_name = Path(cfg.get("data_path", "—")).name
    ts_raw  = report.get("timestamp", "")
    try:
        ts_fmt = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d  %H:%M:%S")
    except Exception:
        ts_fmt = ts_raw

    best_score   = m.get(metric, 0)
    n_pipelines  = len(results)
    n_rows_prof  = prof.get("n_rows", 0)
    n_cols_prof  = prof.get("n_cols", 0)
    best_name     = best.get("name", "—")
    is_image      = report.get("modality") == "Image"
    n_images_prof = prof.get("n_images", 0)
    n_classes_prof = prof.get("n_classes", 0)
    # ── Document ──────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.7 * cm, bottomMargin=1.5 * cm,
    )
    available_w = PAGE_W - 2 * MARGIN

    # ── Page callbacks ────────────────────────────────────────────────────

    def _draw_cover(canvas, doc_):
        """Page 1: clean white cover with navy accents."""
        canvas.saveState()
        pw, ph = doc_.pagesize

        # White background
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, pw, ph, fill=True, stroke=False)

        # Bottom navy bar
        canvas.setFillColor(colors.HexColor("#1e3a5f"))
        canvas.rect(0, 0, pw, 1.8 * cm, fill=True, stroke=False)

        # Left accent stripe (full height, sits on top of bars)
        canvas.setFillColor(colors.HexColor("#2563eb"))
        canvas.rect(0, 0, 0.55 * cm, ph, fill=True, stroke=False)

        # Logo centered at 64% height — use light-bg variant
        logo_cy = ph * 0.64
        _cover_logo = logo_light_png if logo_light_png and os.path.exists(logo_light_png) else None
        if _cover_logo:
            try:
                ls = 4.5 * cm
                canvas.drawImage(_cover_logo, (pw - ls) / 2, logo_cy - ls / 2 + 1.0 * cm,
                                 width=ls, height=ls,
                                 preserveAspectRatio=True, mask="auto")
            except Exception:
                pass

        # "MorphAI" title — dark navy
        canvas.setFont("Helvetica-Bold", 48)
        canvas.setFillColor(colors.HexColor("#1e3a5f"))
        canvas.drawCentredString(pw / 2, logo_cy - 2.8 * cm, AGENT_NAME)

        # Tagline
        canvas.setFont("Helvetica", 16)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawCentredString(pw / 2, logo_cy - 4.1 * cm, AGENT_TAGLINE)

        # Accent separator line
        sep_y = logo_cy - 5.0 * cm
        canvas.setStrokeColor(colors.HexColor("#2563eb"))
        canvas.setLineWidth(1.5)
        canvas.line(pw / 2 - 3.5 * cm, sep_y, pw / 2 + 3.5 * cm, sep_y)

        # Metadata block
        meta_y = sep_y - 0.8 * cm
        if is_image:
            meta_items = [
                ("Dataset",         ds_name),
                ("Modality",        "Image"),
                ("Priority metric", cfg.get("metric", "—").upper()),
                ("Generated",       ts_fmt),
            ]
        else:
            meta_items = [
                ("Dataset",         ds_name),
                ("Target column",   cfg.get("target", "—")),
                ("Priority metric", cfg.get("metric", "—").upper()),
                ("Generated",       ts_fmt),
            ]
        for i, (label, value) in enumerate(meta_items):
            y = meta_y - i * 0.68 * cm
            canvas.setFont("Helvetica-Bold", 9.5)
            canvas.setFillColor(colors.HexColor("#1e3a5f"))
            canvas.drawRightString(pw / 2 - 0.4 * cm, y, f"{label}:")
            canvas.setFont("Helvetica", 9.5)
            canvas.setFillColor(colors.HexColor("#334155"))
            canvas.drawString(pw / 2 + 0.4 * cm, y, str(value))

        # Bottom tagline (white text on navy bar)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawCentredString(
            pw / 2, 0.65 * cm,
            f"{AGENT_NAME}  —  {AGENT_TAGLINE}  |  Preprocessing Analysis Report",
        )
        canvas.restoreState()

    def _draw_page(canvas, doc_):
        """Pages 2+: clean light header + subtle footer."""
        canvas.saveState()
        pw, ph = doc_.pagesize

        # Header bar — light background
        canvas.setFillColor(colors.HexColor("#f8fafc"))
        canvas.rect(0, ph - 1.2 * cm, pw, 1.2 * cm, fill=True, stroke=False)

        # Header bottom border
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.setLineWidth(0.5)
        canvas.line(0, ph - 1.2 * cm, pw, ph - 1.2 * cm)

        # Left accent stripe in header
        canvas.setFillColor(_C_STRIPE)
        canvas.rect(0, ph - 1.2 * cm, 0.35 * cm, 1.2 * cm, fill=True, stroke=False)

        # Logo (light-bg variant)
        _hdr_logo = logo_light_png if logo_light_png and os.path.exists(logo_light_png) else None
        if _hdr_logo:
            try:
                lh = 0.82 * cm
                canvas.drawImage(_hdr_logo, MARGIN + 0.1 * cm, ph - 1.05 * cm,
                                 width=lh, height=lh,
                                 preserveAspectRatio=True, mask="auto")
            except Exception:
                pass

        # Agent name — dark text on light bg
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#1e293b"))
        canvas.drawString(MARGIN + 1.05 * cm, ph - 0.72 * cm, AGENT_NAME)

        # Page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(pw - MARGIN, ph - 0.72 * cm, f"Page {doc_.page}")

        # Footer separator
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.1 * cm, pw - MARGIN, 1.1 * cm)

        # Footer text
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(MARGIN, 0.65 * cm,
                          f"Generated  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
        canvas.drawCentredString(pw / 2, 0.65 * cm, f"{AGENT_NAME}  —  Analysis Report")
        canvas.drawRightString(pw - MARGIN, 0.65 * cm, ds_name)

        canvas.restoreState()

    # ── Styles ────────────────────────────────────────────────────────────
    ss = getSampleStyleSheet()

    def _sty(name, **kw):
        return ParagraphStyle(name, parent=ss["Normal"], **kw)

    sec_hdr_sty  = _sty("sec_hdr",  fontSize=12, textColor=_C_WHITE,
                         fontName="Helvetica-Bold", leading=16)
    h2_sty       = _sty("h2",       fontSize=11, textColor=_C_SEC_NAVY,
                         fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6, leading=15)
    h3_sty       = _sty("h3",       fontSize=10, textColor=_C_BODY,
                         fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4, leading=14)
    body_sty     = _sty("body",     fontSize=9.5, textColor=_C_BODY, leading=14, spaceAfter=5)
    caption_sty  = _sty("caption",  fontSize=8, textColor=_C_MUTED,
                         alignment=TA_CENTER, spaceAfter=8, leading=11)
    bullet_sty   = _sty("bullet_s", fontSize=9.5, textColor=_C_BODY,
                         leading=14, leftIndent=18, spaceAfter=5)
    kpi_lbl_sty  = _sty("kpi_lbl",  fontSize=7.5, textColor=_C_MUTED,
                         fontName="Helvetica-Bold", leading=10)
    kpi_sub_sty  = _sty("kpi_sub",  fontSize=8, textColor=_C_MUTED, leading=11)
    pipe_nm_sty  = _sty("pipe_nm",  fontSize=7.5, textColor=_C_BODY, leading=10)
    pipe_hdr_sty = _sty("pipe_hdr", fontSize=7.5, textColor=_C_WHITE,
                         fontName="Helvetica-Bold", leading=10)

    def _p(text, style=body_sty):
        return Paragraph(str(text) if text is not None else "", style)

    def _kpi_val_sty(col, size=20):
        return _sty(f"kv_{col}_{size}", fontSize=size, textColor=col,
                    fontName="Helvetica-Bold", leading=int(size * 1.2))

    def _section_bar(num: str, title: str, color=_C_SEC_NAVY) -> Table:
        """Section header: left blue accent stripe + coloured band."""
        stripe_w = 5  # pts
        bar = Table(
            [[Paragraph("", _sty("_e")),
              Paragraph(f"<b>{num}</b>  {title}", sec_hdr_sty)]],
            colWidths=[stripe_w, doc.width - stripe_w],
        )
        bar.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#60a5fa")),
            ("BACKGROUND",    (1, 0), (1, -1), color),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (0, -1), 0),
            ("RIGHTPADDING",  (0, 0), (0, -1), 0),
            ("LEFTPADDING",   (1, 0), (1, -1), 14),
            ("RIGHTPADDING",  (1, 0), (1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ]))
        return bar

    def _kpi_card(label: str, value: str, subtitle: str,
                  bg: colors.Color, val_col: colors.Color,
                  brd: colors.Color, val_size: int = 20,
                  width: Optional[float] = None) -> Table:
        iw = width if width is not None else (available_w - 0.9 * cm) / 4
        t  = Table([
            [_p(label.upper(), kpi_lbl_sty)],
            [_p(value, _kpi_val_sty(val_col, val_size))],
            [_p(subtitle, kpi_sub_sty)],
        ], colWidths=[iw])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (0,  0),  10),
            ("TOPPADDING",    (0, 1), (-1, -1),  4),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  10),
            ("BOX",           (0, 0), (-1, -1),  1, brd),
            ("ROUNDEDCORNERS", [6]),
        ]))
        return t

    def _tbl(data, col_widths, extra=None, compact=False):
        """Standard table with navy header + alternating rows."""
        t     = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=True)
        fsize = 7.5 if compact else 8.5
        pad   = 4   if compact else 6
        cmds  = [
            ("BACKGROUND",    (0, 0), (-1,  0), _C_TBL_HDR),
            ("TEXTCOLOR",     (0, 0), (-1,  0), _C_WHITE),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), fsize),
            ("VALIGN",        (0, 0), (-1, -1), "TOP" if compact else "MIDDLE"),
            ("GRID",          (0, 0), (-1, -1), 0.3, _C_TBL_BRD),
            ("LINEBELOW",     (0, 0), (-1,  0), 1.0, _C_TBL_BRD),
            ("LEFTPADDING",   (0, 0), (-1, -1), pad),
            ("RIGHTPADDING",  (0, 0), (-1, -1), pad),
            ("TOPPADDING",    (0, 0), (-1, -1), pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_C_WHITE, _C_TBL_ALT]),
        ]
        if extra:
            cmds.extend(extra)
        t.setStyle(TableStyle(cmds))
        return t

    # ── Generate charts ───────────────────────────────────────────────────
    chart_pipeline  = _chart_pipeline_rankings(results, metric)
    chart_metrics   = _chart_metrics_overview(m, std, metric)
    chart_per_model = _chart_per_model(pmt)
    if is_image:
        chart_class_dist = _chart_image_class_distribution(prof)
        chart_quality    = _chart_image_quality_flags(prof)
        chart_dims       = _chart_image_dimensions(prof)
        chart_profile    = None
        chart_comp       = None
    else:
        chart_class_dist = None
        chart_quality    = None
        chart_dims       = None
        chart_profile    = _chart_profile_issues(prof)
        chart_comp       = _chart_dataset_composition(prof)
    for c in [chart_pipeline, chart_metrics, chart_per_model, chart_profile, chart_comp,
              chart_class_dist, chart_quality, chart_dims]:
        if c:
            tmp_files.append(c)

    # ── Build story ───────────────────────────────────────────────────────
    story: list = []

    # ─── Page 1: Cover (all drawn by _draw_cover canvas callback) ────────
    story.append(PageBreak())

    # ─── Page 2: Executive Summary ───────────────────────────────────────
    story.append(_section_bar("01", "Executive Summary"))
    story.append(Spacer(1, 0.55 * cm))

    # KPI cards — 3 equal cards (Score / Candidates / Dataset)
    kw3 = (available_w - 0.6 * cm) / 3
    kpi_tbl = Table([[
        _kpi_card("Best Score", f"{best_score:.4f}", metric.upper(),
                  _C_KPI_BG, _C_KPI_VAL, _C_KPI_BRD, width=kw3),
        _kpi_card("Candidates", str(n_pipelines), "pipelines evaluated",
                  _C_KPI_BG, _C_KPI_VAL, _C_KPI_BRD, width=kw3),
        _kpi_card("Dataset",
                  f"{n_images_prof:,} images" if is_image else f"{n_rows_prof:,} rows",
                  f"{n_classes_prof} classes" if is_image else f"{n_cols_prof} features",
                  _C_KPI_BG, _C_KPI_VAL, _C_KPI_BRD, width=kw3),
    ]], colWidths=[kw3] * 3)
    kpi_tbl.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # Best Pipeline — KPI-card-consistent step grid (same width as KPI section)
    _step_lm = {
        "num":    "Numeric Imputation",
        "cat":    "Categorical Imputation",
        "scale":  "Feature Scaling",
        "enc":    "Encoding",
        "pwr":    "Power Transform",
        "cl":     "Classifier",
        "imb":    "Imbalance Handling",
        "dedup":  "Deduplication",
        "clip":   "Outlier Clipping",
        "sz":     "Resize",
        "clr":    "Color Mode",
        "norm":   "Normalization",
        "heq":    "Histogram Equalization",
        "dns":    "Denoise",
        "shp":    "Sharpen",
        "hflip":  "H-Flip Augmentation",
        "vflip":  "V-Flip Augmentation",
        "rot":    "Rotation Augmentation",
        "jitter": "Color Jitter",
    }
    _kpi_val10_sty = _sty("kv10", fontSize=10, textColor=_C_KPI_VAL,
                           fontName="Helvetica-Bold", leading=13)
    pipe_tbl_w = available_w - 0.6 * cm
    col_w = pipe_tbl_w / 2
    _step_cells = []
    for _seg in best_name.split("|"):
        _seg = _seg.strip()
        if "=" in _seg:
            _k, _v = _seg.split("=", 1)
            _step_cells.append([
                _p(_step_lm.get(_k.strip(), _k.strip().capitalize()).upper(), kpi_lbl_sty),
                _p(_v.strip().capitalize(), _kpi_val10_sty),
            ])
        elif _seg:
            _step_cells.append([
                _p(_step_lm.get(_seg, _seg.capitalize()).upper(), kpi_lbl_sty),
                _p("Applied", _kpi_val10_sty),
            ])
    if _step_cells:
        _pipe_grid = Table(_step_cells, colWidths=[col_w, col_w])
        _pipe_grid.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _C_KPI_BG),
            ("GRID",          (0, 0), (-1, -1), 0.5, _C_KPI_BRD),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(_pipe_grid)
    story.append(Spacer(1, 0.55 * cm))

    # Run overview table
    story.append(_p("Run Overview", h2_sty))
    story.append(Spacer(1, 0.2 * cm))
    ov_rows = [
        [Paragraph("Field", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)],
        ["Dataset",           ds_name],
    ]
    if is_image:
        ov_rows.append(["Modality", "Image"])
    else:
        ov_rows.append(["Target column", cfg.get("target", "—")])
    ov_rows += [
        ["Priority metric",   cfg.get("metric", "—").upper()],
        ["Pipelines tested",  str(report.get("pipelines_tested", "—"))],
        ["Models / pipeline", str(report.get("n_models", "—"))],
        ["Timestamp",         ts_fmt],
    ]
    story.append(_tbl(ov_rows, [5.5 * cm, available_w - 5.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    # Task & Problem Context (shown when any field is non-empty)
    tc_fields = [
        ("task_type",           "Task type"),
        ("domain",              "Domain / use case"),
        ("problem_description", "Problem description"),
        ("data_meaning",        "Data meaning"),
        ("constraints",         "Constraints"),
    ]
    tc_rows_data = [(label, str(tc.get(key, "")))
                    for key, label in tc_fields if tc.get(key)]
    if tc_rows_data:
        story.append(_p("Task & Problem Context", h2_sty))
        story.append(Spacer(1, 0.2 * cm))
        tc_hdr = [Paragraph("Field", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)]
        story.append(_tbl(
            [tc_hdr] + tc_rows_data,
            [5.5 * cm, available_w - 5.5 * cm],
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ─── Page 3: Dataset Profile ──────────────────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar("02", "Dataset Profile"))
    story.append(Spacer(1, 0.45 * cm))

    if is_image:
        ch_map = {1: "Grayscale", 3: "RGB", 4: "RGBA"}
        ch_label = ch_map.get(prof.get("dominant_color_channels", 3), "RGB")
        img_prof_rows = [
            [Paragraph("Metric", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty),
             Paragraph("Metric", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)],
            ["Images",           f"{n_images_prof:,}",
             "Classes",          f"{prof.get('n_classes','?')}  "
                                 f"(imbalance {prof.get('imbalance_ratio',1):.1f}x)"],
            ["Avg dimensions",   f"{prof.get('avg_height',0):.0f} x {prof.get('avg_width',0):.0f} px",
             "Dimension range",  f"[{prof.get('min_height','?')}x{prof.get('min_width','?')}] to "
                                 f"[{prof.get('max_height','?')}x{prof.get('max_width','?')}]"],
            ["Size uniformity",  "Uniform" if prof.get("is_uniform_size") else
                                 f"Varied  (h_std={prof.get('height_std',0):.1f})",
             "Dominant color",   f"{ch_label}  (gray={prof.get('grayscale_ratio',0):.0%})"],
            ["Avg brightness",   f"{prof.get('avg_brightness',0):.3f}  (std={prof.get('brightness_std',0):.3f})",
             "Avg contrast",     f"{prof.get('avg_contrast',0):.3f}  (std={prof.get('contrast_std',0):.3f})"],
            ["Avg file size",    f"{prof.get('avg_file_size_kb',0):.1f} KB",
             "Corrupt images",   str(prof.get("n_corrupt", 0))],
            ["Low contrast",     "Yes" if prof.get("has_low_contrast") else "No",
             "Varied brightness","Yes" if prof.get("has_varied_brightness") else "No"],
            ["Small images",     "Yes" if prof.get("has_small_images") else "No",
             "Large images",     "Yes" if prof.get("has_large_images") else "No"],
        ]
        hw = available_w / 4
        img_prof_extra = [("ALIGN", (1, 1), (1, -1), "RIGHT"), ("ALIGN", (3, 1), (3, -1), "RIGHT")]
        story.append(_tbl(img_prof_rows, [hw*1.15, hw*0.85, hw*1.15, hw*0.85], extra=img_prof_extra))
        story.append(Spacer(1, 0.5 * cm))
        if chart_class_dist:
            story.append(_fw_image(chart_class_dist, available_w))
            story.append(Spacer(1, 0.4 * cm))
        if chart_quality:
            story.append(_fw_image(chart_quality, available_w))
            story.append(Spacer(1, 0.4 * cm))
        if chart_dims:
            story.append(_fw_image(chart_dims, available_w))
            story.append(Spacer(1, 0.3 * cm))
    else:
        num_c = prof.get("num_cols_count", 0)
        cat_c = prof.get("cat_cols_count", 0)
        prof_rows = [
            [Paragraph("Metric", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty),
             Paragraph("Metric", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)],
            ["Rows",                f"{n_rows_prof:,}",
             "Feature columns",     f"{n_cols_prof}  (num={num_c}, cat={cat_c})"],
            ["Missing ratio",       f"{prof.get('total_missing_ratio',0)*100:.1f}%",
             "Cols >50% missing",   str(prof.get("high_missing_cols_count", 0))],
            ["Duplicate rows",      str(prof.get("n_duplicates", 0)),
             "Classes",             f"{prof.get('n_classes','?')}  "
                                    f"(imbalance {prof.get('imbalance_ratio',1):.1f}x)"],
            ["High-outlier cols",   str(prof.get("high_outlier_cols_count", 0)),
             "High-skew cols",      str(prof.get("high_skew_cols_count", 0))],
            ["High-kurtosis cols",  str(prof.get("high_kurtosis_cols_count", 0)),
             "High-cardinality",    str(prof.get("high_cardinality_cols_count", 0))],
            ["Binary numeric",      str(prof.get("binary_num_cols_count", 0)),
             "Corr. pairs |r|>0.85",str(prof.get("n_high_corr_pairs", 0))],
            ["Constant/near-const", str(prof.get("constant_cols_count", 0)),
             "Sparse features",     "Yes" if prof.get("has_sparse_features") else "No"],
            ["Multicollinearity",   "Yes" if prof.get("has_multicollinearity") else "No",
             "Min class size",      str(prof.get("min_class_size", "?"))],
        ]
        hw = available_w / 4
        prof_extra = [("ALIGN", (1, 1), (1, -1), "RIGHT"), ("ALIGN", (3, 1), (3, -1), "RIGHT")]
        story.append(_tbl(prof_rows, [hw*1.15, hw*0.85, hw*1.15, hw*0.85], extra=prof_extra))
        story.append(Spacer(1, 0.5 * cm))
        if chart_comp:
            story.append(_fw_image(chart_comp,    available_w))
            story.append(Spacer(1, 0.4 * cm))
        if chart_profile:
            story.append(_fw_image(chart_profile, available_w))
            story.append(Spacer(1, 0.3 * cm))

    # ─── Page 4: Pipeline Rankings ────────────────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar("03", "Candidate Pipeline Rankings"))
    story.append(Spacer(1, 0.45 * cm))

    if chart_pipeline:
        n_p = len(results)
        story.append(_fw_image(chart_pipeline, available_w))
        story.append(_p(
            f"All {n_p} candidate pipelines ranked by {metric.upper()}  "
            f"(green = best  |  error bars = inter-model std deviation)", caption_sty))
        story.append(Spacer(1, 0.5 * cm))

    _all_met   = ["accuracy", "f1", "precision", "recall"]
    _met_label = {"accuracy": "Acc", "f1": "F1", "precision": "Prec", "recall": "Rec"}
    _other_met = [mk for mk in _all_met if mk != metric]
    cw_rank = 0.55 * cm
    cw_time = 1.55 * cm
    cw_met  = 1.35 * cm
    cw_name = available_w - cw_rank - cw_met * (1 + len(_other_met)) - cw_time

    rank_hdr = ([Paragraph("#", pipe_hdr_sty), Paragraph("Pipeline", pipe_hdr_sty),
                 Paragraph(metric.upper(), pipe_hdr_sty)]
                + [Paragraph(_met_label[mk], pipe_hdr_sty) for mk in _other_met]
                + [Paragraph("Time (s)", pipe_hdr_sty)])
    rank_rows = [rank_hdr]
    for r in results:
        rm = r.get("metrics", {})
        rank_rows.append(
            [str(r.get("rank", "?")),
             Paragraph(r.get("pipeline_name", ""), pipe_nm_sty),
             f"{rm.get(metric, 0):.4f}"]
            + [f"{rm.get(mk, 0):.4f}" for mk in _other_met]
            + [f"{r.get('elapsed_sec', 0):.2f}"]
        )
    rank_extra = [
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ]
    for i, r in enumerate(results, 1):
        if r.get("rank") == 1:
            rank_extra += [
                ("BACKGROUND", (0, i), (-1, i), _C_TBL_HIGH),
                ("TEXTCOLOR",  (0, i), (-1, i), _C_TBL_TXT),
                ("FONTNAME",   (0, i), (-1, i), "Helvetica-Bold"),
            ]
    story.append(_tbl(
        rank_rows,
        [cw_rank, cw_name, cw_met] + [cw_met]*len(_other_met) + [cw_time],
        extra=rank_extra, compact=True,
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ─── Page 5: Best Pipeline Results ────────────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar("04", "Best Pipeline — Results", color=_C_SEC_GREEN))
    story.append(Spacer(1, 0.45 * cm))

    sum_rows = [
        [Paragraph("Pipeline",        pipe_hdr_sty),
         Paragraph(best_name,         pipe_hdr_sty)],
        ["CV folds",          str(best.get("n_splits", "?"))],
        ["Models evaluated",  str(best.get("n_models", "?"))],
        ["Elapsed",           f"{best.get('elapsed_sec', 0):.2f} s"],
    ]
    story.append(_tbl(sum_rows, [5.5 * cm, available_w - 5.5 * cm],
                      extra=[("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f0fdf4")),
                              ("TEXTCOLOR",  (1, 0), (1, 0), _C_TBL_TXT),
                              ("FONTNAME",   (1, 0), (1, 0), "Helvetica-Bold")]))
    story.append(Spacer(1, 0.55 * cm))

    if chart_metrics:
        story.append(_fw_image(chart_metrics, available_w))
        story.append(_p(
            "Evaluation metrics for the best pipeline  "
            "(mean across models × folds;  error bars = inter-model std deviation)",
            caption_sty))
        story.append(Spacer(1, 0.45 * cm))

    story.append(_p("Metric Scores  (mean across models × folds)", h2_sty))
    met_rows = [[Paragraph("Metric", pipe_hdr_sty),
                 Paragraph("Mean", pipe_hdr_sty),
                 Paragraph("± Std", pipe_hdr_sty),
                 Paragraph("Status", pipe_hdr_sty)]]
    met_extra = [("ALIGN", (1, 1), (2, -1), "RIGHT")]
    for i, mk in enumerate(["accuracy", "f1", "precision", "recall"], 1):
        is_p = (mk == metric)
        met_rows.append([
            f"{'★  ' if is_p else ''}{mk.upper() if is_p else mk.capitalize()}",
            f"{m.get(mk, 0):.4f}",
            f"± {std.get(mk+'_std', 0):.4f}",
            "★  Primary" if is_p else "—",
        ])
        if is_p:
            met_extra += [("BACKGROUND", (0, i), (-1, i), _C_TBL_HIGH),
                          ("TEXTCOLOR",  (0, i), (-1, i), _C_TBL_TXT),
                          ("FONTNAME",   (0, i), (-1, i), "Helvetica-Bold")]
    story.append(_tbl(met_rows, [available_w / 4] * 4, extra=met_extra))
    story.append(Spacer(1, 0.55 * cm))

    if pmt:
        _pm_elems = [_p("Per-Model Breakdown", h2_sty)]
        if chart_per_model:
            _pm_elems.append(_fw_image(chart_per_model, available_w))
            _pm_elems.append(_p("Score per model type across all 4 evaluation metrics", caption_sty))
            _pm_elems.append(Spacer(1, 0.4 * cm))
        story.append(KeepTogether(_pm_elems))
        mn_list  = list(pmt.keys())
        pm_extra = [("ALIGN", (1, 1), (-1, -1), "RIGHT")]
        pm_rows  = [[Paragraph("Metric", pipe_hdr_sty)]
                    + [Paragraph(mn.capitalize(), pipe_hdr_sty) for mn in mn_list]]
        for mk in ["accuracy", "f1", "precision", "recall"]:
            pm_rows.append([mk.capitalize()] + [f"{pmt[mn].get(mk, 0):.4f}" for mn in mn_list])
        cw_pm = available_w / (len(mn_list) + 1)
        story.append(_tbl(pm_rows, [cw_pm] * (len(mn_list) + 1), extra=pm_extra))
        story.append(Spacer(1, 0.3 * cm))

    # ─── Page 6: Configuration & Decision Rationale ───────────────────────
    story.append(PageBreak())
    story.append(_section_bar("05", "Configuration & Decision Rationale",
                               color=_C_SEC_PURP))
    story.append(Spacer(1, 0.45 * cm))

    bp_cfg = best.get("config", {})
    if bp_cfg:
        story.append(_p("Best Pipeline Configuration", h2_sty))
        story.append(Spacer(1, 0.2 * cm))
        items = list(bp_cfg.items())
        half  = (len(items) + 1) // 2

        def _cfg_val(v) -> str:
            return ("Yes" if v else "No") if isinstance(v, bool) else str(v)

        cfg_hdr = [Paragraph("Parameter", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty),
                   Paragraph("Parameter", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)]
        paired  = [cfg_hdr]
        for i in range(half):
            lk, lv = items[i]
            rk, rv = (items[half + i] if half + i < len(items) else ("", ""))
            paired.append([
                lk.replace("_", " ").title(), _cfg_val(lv),
                rk.replace("_", " ").title() if rk else "", _cfg_val(rv) if rk else "",
            ])
        hw2 = available_w / 4
        story.append(_tbl(paired, [hw2*1.15, hw2*0.85, hw2*1.15, hw2*0.85]))
        story.append(Spacer(1, 0.65 * cm))

    if expl:
        story.append(_p("Decision Rationale", h2_sty))
        story.append(Spacer(1, 0.2 * cm))
        for line in expl.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 0.15 * cm))
            elif stripped.startswith("•"):
                text = stripped[1:].strip()
                if ":" in text:
                    key, rest = text.split(":", 1)
                    text = f"<b>{key.strip()}</b>:{rest}"
                story.append(Paragraph(f"&bull;  {text}", bullet_sty))
            else:
                story.append(Paragraph(f"<b>{stripped}</b>", h3_sty))
    story.append(Spacer(1, 0.4 * cm))

    # ─── Learning & Memory Update ─────────────────────────────────────────
    lrn_rows: list = []
    mem_update = lrn.get("memory_update", "")
    if mem_update:
        lrn_rows.append(["Memory update", str(mem_update)])
    mem_inf = lrn.get("memory_influence", {})
    if mem_inf:
        lrn_rows.append(["Good runs injected",   str(mem_inf.get("good_injections", 0))])
        lrn_rows.append(["Bad patterns avoided",  str(mem_inf.get("bad_avoidances", 0))])
    ml = lrn.get("meta_learner", {})
    if ml:
        is_mature = ml.get("is_mature", False)
        n_train   = ml.get("n_train", 0)
        weight    = ml.get("weight", 0.0)
        min_use   = ml.get("min_to_use", 5)
        if is_mature:
            ml_val = f"ACTIVE — {n_train} training samples, advisory weight = {weight:.2f}"
        else:
            ml_val = f"LEARNING — {n_train} / {min_use} samples before activation"
        lrn_rows.append(["Meta-learner status", ml_val])

    if lrn_rows:
        story.append(PageBreak())
        story.append(_section_bar("06", "Learning & Memory Update", color=_C_SEC_NAVY))
        story.append(Spacer(1, 0.45 * cm))
        lrn_hdr = [Paragraph("Field", pipe_hdr_sty), Paragraph("Value", pipe_hdr_sty)]
        story.append(_tbl(
            [lrn_hdr] + lrn_rows,
            [5.5 * cm, available_w - 5.5 * cm],
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ── Build ─────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_draw_cover, onLaterPages=_draw_page)

    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

    return output_path
