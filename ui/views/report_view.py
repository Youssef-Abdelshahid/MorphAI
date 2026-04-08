"""
ui/views/report_view.py — Report viewer mixin.

Provides _build_report_view and _render_report for the App class.
"""
from __future__ import annotations

import os
from datetime import datetime

import customtkinter as ctk

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    PILImage = None
    _PIL_OK = False

from ui.constants import (
    BG_WIN, BG_BAR, BG_CARD,
    ACCENT, ACCENT_H, SUCCESS, BORDER,
    TXT, TXT_MUTED,
    NAV_W,
    FONT_FAMILY,
)
from ui.helpers import _card, _sec_label


class ReportViewMixin:
    """Mixin that adds the Report viewer to App."""

    def _build_report_view(self) -> None:
        view = ctk.CTkFrame(self._content, fg_color=BG_WIN, corner_radius=0)
        self._views["report"] = view

        # Top bar
        bar = ctk.CTkFrame(view, height=44, fg_color=BG_BAR, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Report Viewer",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TXT).pack(side="left", padx=16)
        ctk.CTkButton(
            bar, text="Export PDF", width=110, height=28,
            fg_color=ACCENT, hover_color=ACCENT_H,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._on_export_pdf,
        ).pack(side="right", padx=10, pady=8)

        # Placeholder (visible when no report loaded)
        self._report_placeholder = ctk.CTkLabel(
            view,
            text="No report loaded.\nRun the agent or select a run from History.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=15), text_color=TXT_MUTED,
            justify="center",
        )
        self._report_placeholder.pack(expand=True)

        # Scrollable content (initially hidden)
        self._report_scroll = ctk.CTkScrollableFrame(
            view, fg_color=BG_WIN, scrollbar_button_color=BORDER,
        )

    def _render_report(self, report: dict) -> None:
        """Render a report dict into the report view's scrollable frame."""
        self._report_placeholder.pack_forget()
        self._report_scroll.pack_forget()

        scroll = self._report_scroll
        for w in scroll.winfo_children():
            w.destroy()

        cfg     = report.get("config", {})
        prof    = report.get("profile_summary", {})
        best    = report.get("best_pipeline", {})
        results = report.get("results", [])
        metric  = cfg.get("metric", "f1")
        P       = 22   # padx for all sections

        ts_raw = report.get("timestamp", "")
        try:
            ts_fmt = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d  %H:%M:%S")
        except Exception:
            ts_fmt = ts_raw

        # ── Pre-generate charts for inline display ─────────────────────────
        self._report_chart_refs: list = []
        self._chart_containers: dict = {}

        def _add_chart(chart_id: str, caption: str = "") -> None:
            container = ctk.CTkFrame(scroll, fg_color="transparent")
            container.pack(fill="x", padx=P, pady=(8, 10))
            lbl = ctk.CTkLabel(container, text="Generating chart...", font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED)
            lbl.pack(pady=20)
            self._chart_containers[chart_id] = (container, lbl, caption)

        def _inject_chart(chart_id: str, img) -> None:
            if chart_id not in self._chart_containers: return
            container, lbl, caption = self._chart_containers[chart_id]
            if not img:
                if lbl.winfo_exists(): lbl.configure(text="Chart unavailable")
                return
            if lbl.winfo_exists(): lbl.destroy()
            try:
                win_w = scroll.winfo_width()
                disp_w = max(400, win_w - NAV_W - 2 * P - 55) if win_w > NAV_W + 300 else max(400, win_w - 80)
                disp_h = int(img.height * disp_w / img.width)
                if disp_h > 500:
                    disp_h = 500
                    disp_w = int(img.width * 500 / img.height)
                ci = ctk.CTkImage(light_image=img, dark_image=img, size=(disp_w, disp_h))
                self._report_chart_refs.append((ci, img))
                ctk.CTkLabel(container, image=ci, text="", fg_color="transparent").pack(anchor="w")
                if caption:
                    ctk.CTkLabel(container, text=caption, font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED).pack(pady=(2, 0))
            except Exception: pass

        _resize_state = {"job": None, "last_w": 0}

        def _do_resize():
            _resize_state["job"] = None
            try:
                win_w = scroll.winfo_width()
                if win_w == _resize_state["last_w"]: return
                _resize_state["last_w"] = win_w
                disp_w = max(400, win_w - NAV_W - 2 * P - 55) if win_w > NAV_W + 300 else max(400, win_w - 80)
                for ci, orig_img in self._report_chart_refs:
                    disp_h = int(orig_img.height * disp_w / orig_img.width)
                    if disp_h > 500:
                        disp_h = 500
                        disp_w = int(orig_img.width * 500 / orig_img.height)
                    ci.configure(size=(disp_w, disp_h))
            except Exception: pass

        def _on_resize_all(event=None):
            if _resize_state["job"] is not None:
                self.after_cancel(_resize_state["job"])
            _resize_state["job"] = self.after(150, _do_resize)

        scroll.bind("<Configure>", _on_resize_all, add="+")

        if _PIL_OK:
            def _bg_generate():
                try:
                    from ui.pdf_exporter import (
                        _chart_dataset_composition, _chart_profile_issues,
                        _chart_metrics_overview, _chart_per_model,
                        _chart_pipeline_rankings,
                    )
                    import matplotlib
                    matplotlib.use("Agg")
                    import os
                    
                    def _lc(fn, *a):
                        try:
                            p = fn(*a)
                            if p and os.path.exists(p):
                                img = PILImage.open(p).copy()
                                os.remove(p)
                                return img
                        except Exception: pass
                        return None

                    _bm = best.get("metrics", {})
                    _bs = best.get("metrics_std", {})
                    _bp = best.get("per_model_metrics", {})
                    
                    c1 = _lc(_chart_dataset_composition, prof)
                    self.after(0, lambda: _inject_chart("_c_comp", c1))
                    
                    c2 = _lc(_chart_profile_issues, prof)
                    self.after(0, lambda: _inject_chart("_c_qual", c2))
                    
                    c3 = _lc(_chart_metrics_overview, _bm, _bs, metric)
                    self.after(0, lambda: _inject_chart("_c_met", c3))
                    
                    c4 = _lc(_chart_per_model, _bp)
                    self.after(0, lambda: _inject_chart("_c_pmd", c4))
                    
                    c5 = _lc(_chart_pipeline_rankings, results, metric)
                    self.after(0, lambda: _inject_chart("_c_rank", c5))
                except Exception: pass
            import threading
            threading.Thread(target=_bg_generate, daemon=True).start()

        # ── Helper: coloured section header strip ──────────────────────────
        def sec_hdr(title: str, color: str = ACCENT) -> None:
            h = ctk.CTkFrame(scroll, fg_color=color, corner_radius=6, height=32)
            h.pack(fill="x", padx=P, pady=(16, 6))
            h.pack_propagate(False)
            ctk.CTkLabel(h, text=title,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color="#ffffff").pack(anchor="w", padx=14, pady=6)

        # ── Helper: key-value row inside a card ────────────────────────────
        def kv_row(parent, key: str, value: str, highlight: bool = False) -> None:
            r = ctk.CTkFrame(parent, fg_color="transparent")
            r.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(r, text=key,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED, width=200, anchor="w").pack(side="left")
            ctk.CTkLabel(r, text=str(value),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                         text_color=ACCENT if highlight else TXT,
                         anchor="w", wraplength=720,
                         justify="left").pack(side="left", fill="x", expand=True)

        tc      = report.get("task_context", {})
        lrn     = report.get("learning_summary", {})

        # ─────────────────────────────────────────────────────────────────
        # Section 1: Overview
        sec_hdr("1   Run Overview")
        c = _card(scroll)
        c.pack(fill="x", padx=P, pady=(0, 4))
        kv_row(c, "Run timestamp",     ts_fmt)
        kv_row(c, "Dataset",           cfg.get("data_path", "—"))
        kv_row(c, "Target column",     cfg.get("target", "—"))
        kv_row(c, "Priority metric",   cfg.get("metric", "—").upper())
        kv_row(c, "Pipelines tested",  str(report.get("pipelines_tested", "—")))
        kv_row(c, "Models / pipeline", str(report.get("n_models", "—")))

        # Section 1b: Task & Problem Context (only when any field is set)
        if any(v for v in tc.values()):
            sec_hdr("1b  Task & Problem Context")
            c = _card(scroll)
            c.pack(fill="x", padx=P, pady=(0, 4))
            for field_key, label in [
                ("task_type",           "Task type"),
                ("domain",              "Domain / use case"),
                ("problem_description", "Problem description"),
                ("data_meaning",        "Data meaning"),
                ("constraints",         "Constraints"),
            ]:
                val = tc.get(field_key, "")
                if val:
                    kv_row(c, label, val)

        # Section 2: Dataset Profile
        sec_hdr("2   Dataset Profile")
        c = _card(scroll)
        c.pack(fill="x", padx=P, pady=(0, 4))
        kv_row(c, "Rows",
               f"{prof.get('n_rows', '?'):,}")
        kv_row(c, "Feature columns",
               f"{prof.get('n_cols','?')}  (numeric={prof.get('num_cols_count','?')}, "
               f"categorical={prof.get('cat_cols_count','?')})")
        kv_row(c, "Missing ratio",
               f"{prof.get('total_missing_ratio',0)*100:.1f}%  "
               f"({prof.get('high_missing_cols_count',0)} cols >50% missing)")
        kv_row(c, "Duplicate rows",        str(prof.get("n_duplicates", 0)))
        kv_row(c, "Classes",
               f"{prof.get('n_classes','?')}  "
               f"(imbalance ratio = {prof.get('imbalance_ratio',1):.1f}x, "
               f"min class = {prof.get('min_class_size','?')} samples)")
        kv_row(c, "High-outlier columns",  str(prof.get("high_outlier_cols_count", 0)))
        kv_row(c, "High-skew columns",     str(prof.get("high_skew_cols_count", 0)))
        kv_row(c, "High-kurtosis columns", str(prof.get("high_kurtosis_cols_count", 0)))
        kv_row(c, "High-cardinality cols", str(prof.get("high_cardinality_cols_count", 0)))
        kv_row(c, "Binary numeric cols",   str(prof.get("binary_num_cols_count", 0)))
        kv_row(c, "Corr. pairs |r|>0.85", str(prof.get("n_high_corr_pairs", 0)))
        kv_row(c, "Sparse features",
               "Yes" if prof.get("has_sparse_features") else "No")
        kv_row(c, "Multicollinearity",
               "Yes" if prof.get("has_multicollinearity") else "No")

        _add_chart("_c_comp", "Feature composition")
        _add_chart("_c_qual", "Dataset quality indicators")

        # Section 3: Best Pipeline
        sec_hdr("3   Selected Best Pipeline", color=SUCCESS)
        c = _card(scroll)
        c.pack(fill="x", padx=P, pady=(0, 4))
        kv_row(c, "Pipeline",  best.get("name", "—"), highlight=True)
        kv_row(c, "CV folds",  str(best.get("n_splits", "?")))
        kv_row(c, "Models",    str(best.get("n_models", "?")))
        kv_row(c, "Elapsed",   f"{best.get('elapsed_sec', 0):.2f} s")

        # Metrics table
        m   = best.get("metrics", {})
        std = best.get("metrics_std", {})

        mhdr = ctk.CTkFrame(scroll, fg_color=BG_BAR, corner_radius=0, height=28)
        mhdr.pack(fill="x", padx=P, pady=(6, 0))
        mhdr.pack_propagate(False)
        for label, w in [("Metric", 170), ("Mean", 110), ("+/- Std", 110), ("", 0)]:
            ctk.CTkLabel(mhdr, text=label,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED, width=w, anchor="w").pack(side="left", padx=10)

        for mk in ["accuracy", "f1", "precision", "recall"]:
            is_p  = (mk == metric)
            row_c = "#1a2540" if is_p else BG_CARD
            mrow  = ctk.CTkFrame(scroll, fg_color=row_c, corner_radius=0, height=26)
            mrow.pack(fill="x", padx=P, pady=1)
            mrow.pack_propagate(False)
            ctk.CTkLabel(mrow,
                         text=mk.upper() if is_p else mk.capitalize(),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED,
                         width=170, anchor="w").pack(side="left", padx=10)
            ctk.CTkLabel(mrow, text=f"{m.get(mk,0):.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT,
                         width=110, anchor="w").pack(side="left")
            ctk.CTkLabel(mrow, text=f"{std.get(mk+'_std',0):.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                         text_color=TXT_MUTED, width=110, anchor="w").pack(side="left")

        # Per-model breakdown
        pmt = best.get("per_model_metrics", {})
        if pmt:
            ctk.CTkLabel(scroll, text="Per-model breakdown",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED).pack(anchor="w", padx=P + 10, pady=(8, 2))
            model_names = list(pmt.keys())
            pmhdr = ctk.CTkFrame(scroll, fg_color=BG_BAR, corner_radius=0, height=26)
            pmhdr.pack(fill="x", padx=P, pady=(0, 0))
            pmhdr.pack_propagate(False)
            ctk.CTkLabel(pmhdr, text="Metric",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED, width=130, anchor="w").pack(side="left", padx=10)
            for mn in model_names:
                ctk.CTkLabel(pmhdr, text=mn.capitalize(),
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                             text_color=TXT_MUTED, width=90, anchor="w").pack(side="left")
            for mk in ["accuracy", "f1", "precision", "recall"]:
                pmrow = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=0, height=24)
                pmrow.pack(fill="x", padx=P, pady=1)
                pmrow.pack_propagate(False)
                ctk.CTkLabel(pmrow, text=mk.capitalize(),
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED,
                             width=130, anchor="w").pack(side="left", padx=10)
                for mn in model_names:
                    ctk.CTkLabel(pmrow, text=f"{pmt[mn].get(mk,0):.4f}",
                                 font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                                 width=90, anchor="w").pack(side="left")

        _add_chart("_c_met",  "Evaluation metrics for the best pipeline")
        _add_chart("_c_pmd",  "Per-model breakdown — best pipeline")

        # Section 4: Candidate Rankings
        sec_hdr("4   Candidate Pipeline Rankings")
        rhdr = ctk.CTkFrame(scroll, fg_color=BG_BAR, corner_radius=0, height=28)
        rhdr.pack(fill="x", padx=P, pady=(0, 0))
        rhdr.pack_propagate(False)
        for label, w in [("#", 28), ("Pipeline", 0), (metric.upper(), 80), ("Time", 72)]:
            ctk.CTkLabel(rhdr, text=label,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=TXT_MUTED,
                         width=w if w else 0, anchor="w").pack(
                side="left" if w else "left",
                fill="x" if not w else None,
                expand=(w == 0),
                padx=6,
            )
        for r in results:
            is_best = (r.get("rank") == 1)
            row_c   = "#1a2540" if is_best else BG_CARD
            rrow    = ctk.CTkFrame(scroll, fg_color=row_c, corner_radius=0, height=24)
            rrow.pack(fill="x", padx=P, pady=1)
            rrow.pack_propagate(False)
            ctk.CTkLabel(rrow, text=str(r.get("rank", "?")),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                         text_color=TXT_MUTED, width=28, anchor="w").pack(side="left", padx=6)
            name = r.get("pipeline_name", "")
            ctk.CTkLabel(rrow, text=name,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_best else "normal"),
                         text_color=ACCENT if is_best else TXT,
                         anchor="w").pack(side="left", fill="x", expand=True, padx=4)
            score = r.get("metrics", {}).get(metric, 0)
            ctk.CTkLabel(rrow, text=f"{score:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_best else "normal"),
                         text_color=ACCENT if is_best else TXT,
                         width=80, anchor="w").pack(side="right", padx=4)
            ctk.CTkLabel(rrow, text=f"{r.get('elapsed_sec',0):.2f}s",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED,
                         width=72, anchor="w").pack(side="right")

        _add_chart("_c_rank", f"Candidate pipeline rankings by {metric.upper()}")

        # Section 5: Decision Rationale
        sec_hdr("5   Decision Rationale")
        expl_card = _card(scroll)
        expl_card.pack(fill="x", padx=P, pady=(0, 4))
        explanation = report.get("explanation", "No explanation available.")
        for line in explanation.split("\n"):
            stripped = line.strip()
            if not stripped:
                ctk.CTkLabel(expl_card, text="", height=6,
                             fg_color="transparent").pack()
                continue
            is_bullet = stripped.startswith("•")
            ctk.CTkLabel(
                expl_card, text=stripped,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                 weight="normal" if is_bullet else "bold"),
                text_color=TXT if is_bullet else TXT_MUTED,
                anchor="w", wraplength=900, justify="left",
            ).pack(anchor="w", padx=14, pady=(5, 2) if is_bullet else (8, 2))

        # Section 6: Learning & Memory Update
        sec_hdr("6   Learning & Memory Update")
        c = _card(scroll)
        c.pack(fill="x", padx=P, pady=(0, 24))

        mem_update = lrn.get("memory_update", "")
        if mem_update:
            kv_row(c, "Memory update", mem_update)

        mem_inf = lrn.get("memory_influence", {})
        if mem_inf:
            kv_row(c, "Good runs injected",  str(mem_inf.get("good_injections", 0)))
            kv_row(c, "Bad patterns avoided", str(mem_inf.get("bad_avoidances", 0)))

        ml = lrn.get("meta_learner", {})
        if ml:
            is_mature = ml.get("is_mature", False)
            n_train   = ml.get("n_train", 0)
            weight    = ml.get("weight", 0.0)
            min_use   = ml.get("min_to_use", 5)
            min_full  = ml.get("min_full_wt", 20)
            if is_mature:
                kv_row(c, "Meta-learner status",
                       f"ACTIVE — {n_train} training samples, "
                       f"advisory weight = {weight:.2f}  "
                       f"(max at {min_full} samples)", highlight=True)
            else:
                kv_row(c, "Meta-learner status",
                       f"LEARNING — {n_train} / {min_use} samples before activation")

        if not any([mem_update, mem_inf, ml]):
            kv_row(c, "No learning data", "Learning summary not available for this run.")

        self._report_scroll.pack(fill="both", expand=True)
