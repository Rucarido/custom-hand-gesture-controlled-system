"""Pinch click — thumb tip (4) + index tip (8) distance with hysteresis."""

import math

from tracker import HandTracker


class PinchClickDetector:
    """
    States:
      open   -> pinch detected -> holding -> click fired -> cooldown (open re-arm)
    Hysteresis: pinch_on < pinch_off prevents flutter at the threshold.
    """

    OPEN = "open"
    PINCHING = "pinching"
    FIRED = "fired"

    def __init__(
        self,
        pinch_on: float = 0.04,
        pinch_off: float = 0.06,
        stable_frames: int = 3,
    ) -> None:
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self.stable_frames = stable_frames
        self.state = self.OPEN
        self._stable_count = 0
        self.last_distance = None

    def _distance(self, landmarks) -> float:
        thumb = landmarks[HandTracker.THUMB_TIP]
        index = landmarks[HandTracker.INDEX_TIP]
        return math.dist(thumb[:2], index[:2])

    def update(self, landmarks) -> str | None:
        """
        Returns 'click' once when pinch confirmed, else None.
        """
        d = self._distance(landmarks)
        self.last_distance = d

        if self.state == self.OPEN:
            if d < self.pinch_on:
                self._stable_count += 1
                if self._stable_count >= self.stable_frames:
                    self.state = self.FIRED
                    self._stable_count = 0
                    return "click"
            else:
                self._stable_count = 0

        elif self.state == self.FIRED:
            # Re-arm only after fingers open past pinch_off
            if d > self.pinch_off:
                self.state = self.OPEN
                self._stable_count = 0

        return None

    def reset(self) -> None:
        self.state = self.OPEN
        self._stable_count = 0
