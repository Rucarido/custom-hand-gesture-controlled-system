"""Signal filters — smooth pointer and gate noisy actions."""

import time


class EMAFilter:
    """Exponential moving average — reduces jitter while staying responsive."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._value = None

    def update(self, x: float, y: float):
        if self._value is None:
            self._value = [x, y]
        else:
            self._value[0] = self.alpha * x + (1 - self.alpha) * self._value[0]
            self._value[1] = self.alpha * y + (1 - self.alpha) * self._value[1]
        return self._value[0], self._value[1]

    def reset(self) -> None:
        self._value = None


class DeadZone:
    """Ignore movement smaller than threshold (pixels)."""

    def __init__(self, threshold_px: float = 3.0) -> None:
        self.threshold_px = threshold_px
        self._last = None

    def apply(self, x: float, y: float):
        if self._last is None:
            self._last = (x, y)
            return x, y, True

        dx = x - self._last[0]
        dy = y - self._last[1]
        if (dx * dx + dy * dy) ** 0.5 < self.threshold_px:
            return self._last[0], self._last[1], False

        self._last = (x, y)
        return x, y, True


class JumpGuard:
    """Drop frames where cursor teleports (tracking glitch)."""

    def __init__(self, max_jump_ratio: float, screen_w: int, screen_h: int) -> None:
        self.max_dist = max_jump_ratio * ((screen_w**2 + screen_h**2) ** 0.5)
        self._last = None

    def accept(self, x: float, y: float) -> bool:
        if self._last is None:
            self._last = (x, y)
            return True

        dx = x - self._last[0]
        dy = y - self._last[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_dist:
            return False

        self._last = (x, y)
        return True

    def reset(self) -> None:
        self._last = None


class Cooldown:
    """Block repeated actions for cooldown_ms after each fire."""

    def __init__(self, cooldown_ms: float = 400) -> None:
        self.cooldown_ms = cooldown_ms
        self._last_fire = 0.0

    def ready(self) -> bool:
        elapsed = (time.perf_counter() - self._last_fire) * 1000
        return elapsed >= self.cooldown_ms

    def fire(self) -> None:
        self._last_fire = time.perf_counter()

    @property
    def remaining_ms(self) -> float:
        elapsed = (time.perf_counter() - self._last_fire) * 1000
        return max(0.0, self.cooldown_ms - elapsed)
