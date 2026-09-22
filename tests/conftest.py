"""Pytest bootstrap — put src/ on sys.path.

Tests must NOT import mediapipe / cv2 / pynput / pyautogui: the pure-logic
modules under test are hardware-free by design.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
