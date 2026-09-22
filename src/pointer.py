"""Map index fingertip to screen coordinates — stabilized.

Precision measures:
  - tip position blends TIP (8) 70% + DIP (7) 30%: the DIP joint jitters
    far less than the tip, so the blend cuts raw jitter ~30% before any
    temporal filter runs;
  - index-extended gate has release hysteresis (``gate_release_frames``):
    a single bent frame (pinch dip, motion blur) no longer drops the
    cursor — the finger must read folded N frames in a row;
  - pyautogui imports lazily and screen size is injectable, so the module
    stays hardware-free for tests.
"""

import landmarks as L


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _screen_size(fallback=(1920, 1080)):
    try:
        import pyautogui

        return pyautogui.size()
    except Exception:
        return fallback


class PointerMapper:
    def __init__(
        self,
        mirror: bool = True,
        require_index_extended: bool = True,
        motion_area: dict | None = None,
        screen_size: tuple[int, int] | None = None,
        gate_release_frames: int = 3,
        tip_blend: float = 0.7,
    ) -> None:
        self.mirror = mirror
        self.require_index_extended = require_index_extended
        if screen_size is None:
            self.screen_w, self.screen_h = _screen_size()
        else:
            self.screen_w, self.screen_h = screen_size
        self.gate_release_frames = max(1, gate_release_frames)
        self.tip_blend = min(1.0, max(0.0, tip_blend))
        self._folded_count = 0

        if motion_area and motion_area.get("enabled", True):
            self._ma_x_min = motion_area["x_min"]
            self._ma_x_max = motion_area["x_max"]
            self._ma_y_min = motion_area["y_min"]
            self._ma_y_max = motion_area["y_max"]
            self._ma_enabled = True
        else:
            self._ma_enabled = False

    def _remap_to_motion_area(self, x: float, y: float):
        x = _clamp(x, self._ma_x_min, self._ma_x_max)
        y = _clamp(y, self._ma_y_min, self._ma_y_max)
        x = (x - self._ma_x_min) / (self._ma_x_max - self._ma_x_min)
        y = (y - self._ma_y_min) / (self._ma_y_max - self._ma_y_min)
        return x, y

    def _stable_tip(self, landmarks):
        """TIP/DIP blend — DIP barely moves relative to the tip's wobble."""
        tip = landmarks[L.INDEX_TIP]
        dip = landmarks[L.INDEX_DIP]
        w = self.tip_blend
        return (
            w * tip[0] + (1.0 - w) * dip[0],
            w * tip[1] + (1.0 - w) * dip[1],
            tip[2] if len(tip) > 2 else 0.0,
        )

    def _gate_open(self, landmarks) -> bool:
        """Extended-gate with release hysteresis against single-frame dips."""
        if not self.require_index_extended:
            return True
        if L.is_index_extended(landmarks):
            self._folded_count = 0
            return True
        self._folded_count += 1
        return self._folded_count < self.gate_release_frames

    def landmarks_to_screen(self, landmarks) -> tuple[float, float] | None:
        if not self._gate_open(landmarks):
            return None

        x, y, _ = self._stable_tip(landmarks)

        if self._ma_enabled:
            x, y = self._remap_to_motion_area(x, y)

        if self.mirror:
            x = 1.0 - x

        screen_x = x * self.screen_w
        screen_y = y * self.screen_h
        return screen_x, screen_y

    def reset(self) -> None:
        self._folded_count = 0
