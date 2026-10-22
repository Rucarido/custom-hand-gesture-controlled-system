"""Sanity check: src/gesture_classifier.py on real mini-subset images.

Run under .venv (has torch+cu126): proves the full chain game mode will
use — best.pt load -> preprocessing -> GPU inference -> BlazeHand pose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image

from gesture_classifier import GestureClassifier


def load_bgr(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb[..., ::-1].copy()  # -> BGR like OpenCV camera frames


def main():
    gc = GestureClassifier("models/blazehand-mini/best.pt")
    assert gc.available, "classifier failed to load"
    root = Path("data/hagrid-mini")
    hits = {}
    for cls in ("one", "fist", "peace", "four", "no_gesture"):
        img = sorted((root / cls).glob("*.jpeg"))[0]
        frame = load_bgr(img)
        pose, conf, raw = gc.classify(frame)
        ok = raw == cls
        hits[cls] = ok
        print(f"{cls:12s} -> raw={raw:12s} conf={conf:.3f} "
              f"pose={pose} {'OK' if ok else 'MISS'}")
    # latency probe (20 frames)
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        gc.classify(frame)
    print(f"latency: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/frame "
          f"on {gc.device.type}")
    print("PASS" if all(hits.values()) else "PARTIAL (see MISS lines)")


if __name__ == "__main__":
    main()
