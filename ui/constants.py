"""
ui/constants.py — Shared colour palette, layout constants, and project root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# ── Colour palette  (light, dark) tuples — CTk picks the right one automatically
# Backgrounds
BG_WIN      = ("#f1f5f9", "#0f172a")   # main content area
BG_SIDEBAR  = ("#ffffff", "#020617")   # sidebar — distinct panel
BG_BAR      = ("#f8fafc", "#1e293b")   # top bars / hover
BG_INPUT    = ("#ffffff", "#0f172a")   # input fields
BG_CARD     = ("#ffffff", "#1e293b")   # card surfaces
BORDER      = ("#cbd5e1", "#334155")   # card borders
ROW_HIGHLIGHT = ("#dbeafe", "#1a2540") # highlighted table row (selected metric / best rank)

# Accent / semantic — desaturated for dark mode to avoid eye-strain
ACCENT      = ("#3b82f6", "#3b82f6")   # primary blue
ACCENT_H    = ("#2563eb", "#60a5fa")   # hover state
ACCENT2     = ("#8b5cf6", "#a78bfa")   # purple
SUCCESS     = ("#10b981", "#34d399")   # green
ERROR       = ("#ef4444", "#f87171")
WARN        = ("#f59e0b", "#fbbf24")

# Text
TXT         = ("#0f172a", "#f8fafc")
TXT_MUTED   = ("#64748b", "#94a3b8")

# Typography
FONT_FAMILY = "Inter"

# Console widget — always dark (tk.Text doesn't accept tuples)
BG_CONSOLE  = "#080b10"
TXT_CONSOLE = "#9fb3c8"

# Plain-string dark values for tkinter widgets (tk.Text / tk.Frame / tk.Scrollbar)
_TK_TXT     = "#f8fafc"
_TK_MUTED   = "#94a3b8"
_TK_ACCENT  = "#3b82f6"
_TK_SIDEBAR = "#020617"

# Layout
NAV_W   = 220
POLL_MS = 80

# Console tag colours
CTAG: Dict[str, str] = {
    "SEP":    "#2b3348",
    "STEP":   "#4a7fcb",
    "INFO":   "#9fb3c8",
    "OK":     "#3dba74",
    "WARN":   "#f0a04b",
    "ERROR":  "#e05c5c",
    "METRIC": "#7ec8e3",
    "BEST":   "#c9b1ff",
    "MUTED":  "#5c6780",
}
