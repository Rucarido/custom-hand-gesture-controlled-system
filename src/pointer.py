"""Map index fingertip (landmark 8) to screen coordinates."""

import pyautogui

from tracker import HandTracker


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class PointerMapper:
    def __init__(
        self,
        mirror: bool = True,
        require_index_extended: bool = True,
        motion_area: dict | None = None,
    ) -> None:
        self.mirror = mirror
        self.require_index_extended = require_index_extended
        self.screen_w, self.screen_h = pyautogui.size()

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

    def landmarks_to_screen(self, landmarks) -> tuple[float, float] | None:
        if self.require_index_extended:
            if not HandTracker.is_index_extended(landmarks):
                return None

        x, y, _ = landmarks[HandTracker.INDEX_TIP]

        if self._ma_enabled:
            x, y = self._remap_to_motion_area(x, y)

        if self.mirror:
            x = 1.0 - x

        screen_x = x * self.screen_w
        screen_y = y * self.screen_h
        return screen_x, screen_y
