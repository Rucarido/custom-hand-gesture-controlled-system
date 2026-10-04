"""Pinch click — thumb tip (4) + index tip (8), scale-invariant ratio.

Metric: pinch_ratio = dist(thumb_tip, index_tip) / hand_size (landmarks.py),
so thresholds hold whether the hand fills the frame or sits a metre back.
Hysteresis: pinch_on < pinch_off prevents flutter at the threshold.
Signal conditioning: an asymmetric EMA (fast release / slow engage,
see filters.AsymmetricSmoother) absorbs ±0.05 frame jitter without the
extra stable_frames delay the old code needed — engagement still needs
``stable_frames`` consecutive closed frames, but one noisy frame in the
middle no longer restarts the count from zero.
"""

import landmarks as L
from filters import AsymmetricSmoother


class PinchClickDetector:
    """
    States:
      open -> pinching (ratio < pinch_on, counting stable frames)
        -> click fired -> FIRED until ratio > pinch_off re-arms to open.
    """

    OPEN = "open"
    PINCHING = "pinching"
    FIRED = "fired"

    # Legacy absolute-distance thresholds (< 0.15, pre-ratio code) are scaled
    # to ratio units so old configs keep working instead of never firing.
    _LEGACY_SCALE = 5.0

    def __init__(
        self,
        pinch_on: float = 0.25,
        pinch_off: float = 0.40,
        stable_frames: int = 2,
        smooth_alpha_up: float = 0.85,
        smooth_alpha_down: float = 0.5,
    ) -> None:
        if pinch_on < 0.15 and pinch_off < 0.15:
            pinch_on *= self._LEGACY_SCALE
            pinch_off *= self._LEGACY_SCALE
        if pinch_off <= pinch_on:
            pinch_off = pinch_on + 0.05
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self.stable_frames = max(1, stable_frames)
        self._ratio_filter = AsymmetricSmoother(
            alpha_up=smooth_alpha_up, alpha_down=smooth_alpha_down
        )
        self.state = self.OPEN
        self._stable_count = 0
        self.last_distance = None  # raw tip distance (HUD/debug, normalized units)
        self.last_ratio = None  # smoothed pinch metric driving decisions

    def _distance(self, landmarks) -> float:
        return L.tip_distance(landmarks)

    def _ratio(self, landmarks) -> float:
        return L.pinch_ratio(landmarks)

    def update(self, landmarks) -> str | None:
        """
        Returns 'click' once when pinch confirmed, else None.
        """
        self.last_distance = self._distance(landmarks)
        r = self._ratio_filter.update(self._ratio(landmarks))
        self.last_ratio = r

        if self.state == self.OPEN:
            if r < self.pinch_on:
                self.state = self.PINCHING
                self._stable_count = 1
                if self._stable_count >= self.stable_frames:
                    self.state = self.FIRED
                    self._stable_count = 0
                    return "click"
            # else: stay OPEN, count stays 0

        elif self.state == self.PINCHING:
            if r < self.pinch_on:
                self._stable_count += 1
                if self._stable_count >= self.stable_frames:
                    self.state = self.FIRED
                    self._stable_count = 0
                    return "click"
            elif r > self.pinch_off:
                # released before confirming — back to open
                self.state = self.OPEN
                self._stable_count = 0
            # else: in the hysteresis band — hold count, wait

        elif self.state == self.FIRED and r > self.pinch_off:
            # Re-arm only after fingers open past pinch_off
            self.state = self.OPEN
            self._stable_count = 0

        return None

    def reset(self) -> None:
        self.state = self.OPEN
        self._stable_count = 0
        self._ratio_filter.reset()
