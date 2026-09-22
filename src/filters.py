"""Signal filters — smooth pointer and gate noisy actions."""

import math
import time


class _LowPass:
    """First-order low-pass, alpha derived from cutoff Hz + sample freq."""

    def __init__(self) -> None:
        self._hat = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        te = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value: float, cutoff: float, freq: float) -> float:
        a = self._alpha(cutoff, freq)
        if self._hat is None:
            self._hat = value
        else:
            self._hat = a * value + (1.0 - a) * self._hat
        return self._hat

    def reset(self) -> None:
        self._hat = None


class OneEuroFilter:
    """1€ adaptive filter (Casiez et al. 2012) — one axis.

    Low cutoff at rest kills micro-jitter; cutoff rises with speed
    (beta * |velocity|) so fast motion stays snappy with little lag.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last = None
        self._last_t = None

    def update(self, value: float, t: float | None = None) -> float:
        now = time.perf_counter() if t is None else t
        if self._last_t is not None and now > self._last_t:
            # Clamped: glitch/duplicate stamps must never push alpha to 1
            # (which would silently disable all filtering).
            self.freq = min(120.0, max(1.0, 1.0 / (now - self._last_t)))
        self._last_t = now
        d = 0.0 if self._last is None else (value - self._last) * self.freq
        d_hat = self._dx.filter(d, self.d_cutoff, self.freq)
        cutoff = self.min_cutoff + self.beta * abs(d_hat)
        out = self._x.filter(value, cutoff, self.freq)
        self._last = value
        return out

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last = None
        self._last_t = None


class OneEuroXY:
    """2D 1€ filter for screen-pointer smoothing (drop-in EMA replacement)."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)

    def update(self, x: float, y: float, t: float | None = None):
        return self._fx.update(x, t), self._fy.update(y, t)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()


def make_pointer_filter(cfg: dict):
    """Build the pointer smoother honoring pointer/filter settings.

    cfg: the ``pointer`` section of settings.yaml — ``filter: one_euro|ema``.
    Falls back to EMA on unknown names so a typo never kills the loop.
    """
    if cfg.get("filter", "ema") == "one_euro":
        params = cfg.get("one_euro", {}) or {}
        return OneEuroXY(
            min_cutoff=params.get("min_cutoff", 1.2),
            beta=params.get("beta", 0.008),
            d_cutoff=params.get("d_cutoff", 1.0),
        )
    return EMAFilter(alpha=cfg.get("ema_alpha", 0.35))


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
    """Drop frames where cursor teleports (tracking glitch).

    Reseeds after ``max_rejects`` consecutive drops: a sustained teleport
    means the hand genuinely moved there, so the cursor must recover
    instead of freezing at the stale position forever.
    """

    def __init__(
        self,
        max_jump_ratio: float,
        screen_w: int,
        screen_h: int,
        max_rejects: int = 5,
    ) -> None:
        self.max_dist = max_jump_ratio * ((screen_w**2 + screen_h**2) ** 0.5)
        self.max_rejects = max(1, max_rejects)
        self._last = None
        self._rejects = 0

    def accept(self, x: float, y: float) -> bool:
        if self._last is None:
            self._last = (x, y)
            return True

        dx = x - self._last[0]
        dy = y - self._last[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_dist:
            self._rejects += 1
            if self._rejects >= self.max_rejects:
                # genuine move, not a glitch — reseed here
                self._last = (x, y)
                self._rejects = 0
                return True
            return False

        self._last = (x, y)
        self._rejects = 0
        return True

    def reset(self) -> None:
        self._last = None
        self._rejects = 0


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
