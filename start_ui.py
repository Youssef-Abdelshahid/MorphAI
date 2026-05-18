"""
start_ui.py — Launch the desktop assistant UI for the Preprocessing Agent.

Usage:
    python start_ui.py

Requirements:
    pip install customtkinter
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of where Python is invoked.
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from ui.app import main

    main()