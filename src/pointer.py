"""Map index fingertip (landmark 8) to screen coordinates."""

import pyautogui

from tracker import HandTracker


class PointerMapper:
    def __init__(
        self,
        mirror: bool = True,
        require_index_extended: bool = True,
    ) -> None:
        self.mirror = mirror
        self.require_index_extended = require_index_extended
        self.screen_w, self.screen_h = pyautogui.size()

    def landmarks_to_screen(self, landmarks) -> tuple[float, float] | None:
        if self.require_index_extended:
            if not HandTracker.is_index_extended(landmarks):
                return None

        x, y, _ = landmarks[HandTracker.INDEX_TIP]
        if self.mirror:
            x = 1.0 - x

        screen_x = x * self.screen_w
        screen_y = y * self.screen_h
        return screen_x, screen_y
