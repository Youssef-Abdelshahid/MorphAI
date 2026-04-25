from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import customtkinter as ctk

from src.image.config import (
    metric_label as image_metric_label,
    valid_metrics_for_task as image_valid_metrics_for_task,
)
from src.audio.config import (
    metric_label as audio_metric_label,
    valid_metrics_for_task as audio_valid_metrics_for_task,
)
from src.tabular.config import (
    metric_label as tabular_metric_label,
    valid_metrics_for_task as tabular_valid_metrics_for_task,
)
from ui.constants import (
    _ROOT,
    BG_WIN, BG_BAR, BG_SIDEBAR, BG_INPUT,
    ACCENT, ACCENT_H, BORDER,
    TXT, TXT_MUTED,
    FONT_FAMILY,
)
from ui.helpers import _card, _load_json, _open_file


class HistoryViewMixin:
    def _build_history_view(self) -> None:
        view = ctk.CTkFrame(self._content, fg_color=BG_WIN, corner_radius=0)
        self._views["history"] = view

        bar = ctk.CTkFrame(view, height=44, fg_color=BG_BAR, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Run History",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TXT).pack(side="left", padx=16)
        ctk.CTkButton(
            bar, text="Refresh", width=72, height=28,
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=TXT_MUTED, hover_color=BG_BAR,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._refresh_history,
        ).pack(side="right", padx=10, pady=8)

        body = ctk.CTkFrame(view, fg_color="transparent")
        body.pack(fill="both", expand=True)

        list_panel = ctk.CTkFrame(body, width=290, fg_color=BG_SIDEBAR, corner_radius=0)
        list_panel.pack(side="left", fill="y")
        list_panel.pack_propagate(False)
        ctk.CTkLabel(list_panel, text="PAST RUNS",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                     text_color=TXT_MUTED).pack(anchor="w", padx=16, pady=(14, 6))
        self._hist_list = ctk.CTkScrollableFrame(
            list_panel, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        self._hist_list.pack(fill="both", expand=True)

        ctk.CTkFrame(body, width=1, fg_color=BORDER, corner_radius=0).pack(
            side="left", fill="y"
        )

        self._hist_detail = ctk.CTkScrollableFrame(
            body, fg_color=BG_WIN, scrollbar_button_color=BORDER,
        )
        self._hist_detail.pack(side="left", fill="both", expand=True)

        self._hist_empty = ctk.CTkLabel(
            self._hist_detail,
            text="Select a run from the list to view its details.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=15), text_color=TXT_MUTED,
        )
        self._hist_empty.pack(expand=True, pady=80)

    def _refresh_history(self) -> None:
        for w in self._hist_list.winfo_children():
            w.destroy()

        reports_dir = _ROOT / "reports"
        files: List[Path] = []
        if reports_dir.exists():
            files = sorted(
                reports_dir.glob("report_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

        if not files:
            ctk.CTkLabel(
                self._hist_list, text="No past runs found.",
                font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED,
            ).pack(padx=12, pady=12)
            return

        for jf in files:
            report = _load_json(jf)
            if report is None:
                continue
            cfg = report.get("config", {})
            is_image = report.get("modality") == "Image"
            is_audio = report.get("modality") == "Audio"
            metric_label_fn = audio_metric_label if is_audio else image_metric_label if is_image else tabular_metric_label
            ts_raw = report.get("timestamp", "")
            try:
                ts_fmt = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d  %H:%M")
            except Exception:
                ts_fmt = jf.stem[-15:]
            ds_name = Path(cfg.get("data_path", "?")).name
            best_pipe = report.get("best_pipeline", {})
            metric = best_pipe.get("selected_metric", cfg.get("metric", "?"))
            sc = best_pipe.get("raw_metrics", best_pipe.get("metrics", {})).get(metric)
            score = f"{sc:.4f}" if sc is not None else "?"
            label = f"{ts_fmt}  |  {ds_name}\n{metric_label_fn(metric)} {score}"

            btn = ctk.CTkButton(
                self._hist_list, text=label,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12), height=52,
                anchor="w", corner_radius=6,
                fg_color="transparent", hover_color=BG_BAR, text_color=TXT,
                command=lambda r=report, p=jf: self._show_history_detail(r, p),
            )
            btn.pack(fill="x", padx=6, pady=2)

    def _show_history_detail(self, report: dict, report_path: Path) -> None:
        for w in self._hist_detail.winfo_children():
            w.destroy()

        best = report.get("best_pipeline", {})
        cfg = report.get("config", {})
        is_image = report.get("modality") == "Image"
        is_audio = report.get("modality") == "Audio"
        metric_label_fn = audio_metric_label if is_audio else image_metric_label if is_image else tabular_metric_label
        valid_metrics_fn = audio_valid_metrics_for_task if is_audio else image_valid_metrics_for_task if is_image else tabular_valid_metrics_for_task
        metric = best.get("selected_metric", cfg.get("metric", "f1"))
        m = best.get("raw_metrics", best.get("metrics", {}))
        task_type = report.get("task_context", {}).get("task_type", "")
        metric_names = valid_metrics_fn(task_type) or list(m.keys())
        P = 20

        ts_raw = report.get("timestamp", "")
        try:
            ts_fmt = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d  %H:%M:%S")
        except Exception:
            ts_fmt = ts_raw

        ctk.CTkLabel(
            self._hist_detail, text="Run Details",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"), text_color=TXT,
        ).pack(anchor="w", padx=P, pady=(16, 8))

        c = _card(self._hist_detail)
        c.pack(fill="x", padx=P, pady=(0, 10))

        def kv(k: str, v: str) -> None:
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=k,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED, width=140, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=v,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                         anchor="w", wraplength=420).pack(side="left", fill="x", expand=True)

        kv("Timestamp", ts_fmt)
        kv("Dataset", Path(cfg.get("data_path", "—")).name)
        if report.get("modality") in {"Image", "Audio"}:
            kv("Modality", report.get("modality"))
        else:
            kv("Target", cfg.get("target", "—"))
        kv("Priority metric", metric_label_fn(metric))
        kv("Best pipeline", best.get("name", "—"))
        score_text = f"{m.get(metric, 0):.4f}"
        if "final_score" in best:
            score_text += f"  |  normalized={best.get('normalized_score', best.get('final_score', 0)):.4f}"
        score_text += f"  ({'; '.join(f'{metric_label_fn(mk)}={m.get(mk, 0):.4f}' for mk in metric_names)})"
        kv("Score", score_text)
        kv("Pipelines tested", str(report.get("pipelines_tested", "?")))
        kv("Report file", report_path.name)

        ctk.CTkFrame(c, fg_color="transparent", height=6).pack()

        btn_row = ctk.CTkFrame(self._hist_detail, fg_color="transparent")
        btn_row.pack(fill="x", padx=P, pady=(0, 12))

        ctk.CTkButton(
            btn_row, text="View Full Report", width=150, height=32,
            fg_color=ACCENT, hover_color=ACCENT_H, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=lambda r=report: self._load_history_report(r),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="Export PDF", width=110, height=32,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TXT,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=lambda r=report: self._export_pdf_for(r),
        ).pack(side="left", padx=(0, 8))

        proc_dir = _ROOT / "processed"
        ds_stem = Path(cfg.get("data_path", "")).stem
        is_image_run = report.get("modality") == "Image"
        is_audio_run = report.get("modality") == "Audio"
        artifact_files: List[Path] = []
        if proc_dir.exists() and ds_stem:
            if is_image_run or is_audio_run:
                artifact_files = sorted(
                    proc_dir.glob(f"{ds_stem}_*_processed.zip"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            else:
                artifact_files = sorted(
                    proc_dir.glob(f"{ds_stem}_*_cleaned.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
        if artifact_files:
            btn_text = "Open Processed ZIP" if (is_image_run or is_audio_run) else "Open Cleaned CSV"
            ctk.CTkButton(
                btn_row, text=btn_text, width=150, height=32,
                fg_color=BG_INPUT, hover_color=BORDER,
                border_width=1, border_color=BORDER, text_color=TXT,
                corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                command=lambda p=artifact_files[0]: _open_file(p),
            ).pack(side="left")

        prof = report.get("profile_summary", {})
        if prof:
            ctk.CTkLabel(
                self._hist_detail, text="DATASET PROFILE",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                text_color=TXT_MUTED,
            ).pack(anchor="w", padx=P, pady=(8, 4))
            pc = _card(self._hist_detail)
            pc.pack(fill="x", padx=P, pady=(0, 20))
            if report.get("modality") == "Image":
                ch_map = {1: "Grayscale", 3: "RGB", 4: "RGBA"}
                for k, v in [
                    ("Images", f"{prof.get('n_images', 0):,}"),
                    ("Classes", f"{prof.get('n_classes','?')}  (imbalance {prof.get('imbalance_ratio',1):.1f}x)"),
                    ("Avg size", f"{prof.get('avg_height', 0):.0f} x {prof.get('avg_width', 0):.0f} px"),
                    ("Color mode", ch_map.get(prof.get('dominant_color_channels', 3), "RGB")),
                    ("Avg brightness", f"{prof.get('avg_brightness', 0):.3f}"),
                    ("Corrupt", str(prof.get('n_corrupt', 0))),
                ]:
                    kv(k, v)
            elif report.get("modality") == "Audio":
                for k, v in [
                    ("Audio files", f"{prof.get('n_audio_files', 0):,}"),
                    ("Classes", f"{prof.get('n_classes','?')}  (imbalance {prof.get('imbalance_ratio',1):.1f}x)"),
                    ("Avg duration", f"{prof.get('avg_duration_sec', 0):.2f}s"),
                    ("Sample rates", str(prof.get("sample_rate_distribution", {}))),
                    ("Quality", f"corrupt={prof.get('n_corrupt', 0)}, silent={prof.get('n_silent', 0)}, clipped={prof.get('n_clipped', 0)}"),
                    ("Noise proxy", f"{prof.get('estimated_noise_ratio', 0):.4f}"),
                ]:
                    kv(k, v)
            else:
                for k, v in [
                    ("Rows", f"{prof.get('n_rows', 0):,}"),
                    ("Columns", f"{prof.get('n_cols','?')}  (num={prof.get('num_cols_count','?')}, cat={prof.get('cat_cols_count','?')})"),
                    ("Missing", f"{prof.get('total_missing_ratio',0)*100:.1f}%"),
                    ("Classes", f"{prof.get('n_classes','?')}  (imbalance {prof.get('imbalance_ratio',1):.1f}x)"),
                    ("Outlier cols", str(prof.get("high_outlier_cols_count", 0))),
                    ("Skew cols", str(prof.get("high_skew_cols_count", 0))),
                ]:
                    kv(k, v)

    def _load_history_report(self, report: dict) -> None:
        self._report_data = report
        self._render_report(report)
        self._switch_view("report")
