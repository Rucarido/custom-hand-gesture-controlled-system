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

    Responsive-cursor tuning: min_cutoff ~1.5-2.0 keeps rest steady,
    beta ~0.05-0.10 lets the cutoff open up quickly once the hand really
    moves (the old beta=0.008 barely adapted, so fast flicks lagged).
    Always feed the real sample timestamp ``t`` so the cutoff tracks the
    true frame rate instead of a guessed 30 Hz.
    """

    def __init__(
        self,
        min_cutoff: float = 1.7,
        beta: float = 0.07,
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
        min_cutoff: float = 1.7,
        beta: float = 0.07,
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


class MotionPredictor:
    """Constant-velocity predictor — hides pipeline latency.

    The camera -> MediaPipe -> OS-cursor chain costs ~30-60 ms, so the
    cursor visibly trails fast hands. Assuming near-constant velocity over
    that horizon (standard in VR/pointer pipelines), extrapolate:

        predicted = smoothed + velocity * lead_time

    Velocity itself is EMA-smoothed (``velocity_alpha``) so tracker noise
    doesn't get amplified into overshoot. ``lead_s`` ~0.02-0.03 matches a
    30 fps pipeline; 0 disables prediction. Output is clamped to a small
    overshoot radius so a sudden stop never flings the cursor.
    """

    def __init__(
        self,
        lead_s: float = 0.024,
        velocity_alpha: float = 0.4,
        max_lead_px: float = 48.0,
    ) -> None:
        self.lead_s = max(0.0, lead_s)
        self.velocity_alpha = min(1.0, max(0.0, velocity_alpha))
        self.max_lead_px = max(0.0, max_lead_px)
        self._last = None
        self._last_t = None
        self._vx = 0.0
        self._vy = 0.0

    def update(self, x: float, y: float, t: float | None = None):
        now = time.perf_counter() if t is None else t
        if self.lead_s <= 0.0 or self._last is None or self._last_t is None:
            self._last = (x, y)
            self._last_t = now
            return x, y
        dt = now - self._last_t
        if dt <= 1e-6 or dt > 0.5:
            # Duplicate stamp or long gap (dropout) — no velocity info.
            self._last = (x, y)
            self._last_t = now
            return x, y
        inst_vx = (x - self._last[0]) / dt
        inst_vy = (y - self._last[1]) / dt
        a = self.velocity_alpha
        self._vx = a * inst_vx + (1.0 - a) * self._vx
        self._vy = a * inst_vy + (1.0 - a) * self._vy
        px = x + self._vx * self.lead_s
        py = y + self._vy * self.lead_s
        # Clamp overshoot: prediction may lead, never leap.
        dx, dy = px - x, py - y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_lead_px and dist > 1e-9:
            s = self.max_lead_px / dist
            px, py = x + dx * s, y + dy * s
        self._last = (x, y)
        self._last_t = now
        return px, py

    def reset(self) -> None:
        self._last = None
        self._last_t = None
        self._vx = 0.0
        self._vy = 0.0


class AsymmetricSmoother:
    """1D EMA with fast-release / slow-engage rates (gesture signals).

    Pinch ratio jitters ±0.05 frame-to-frame. A symmetric EMA either lags
    engagement or delays release. Splitting the rate — ``alpha_up`` fast
    when the signal opens (release snaps back instantly), ``alpha_down``
    slower when it closes (a single noisy frame can't fake a pinch) —
    gives stability without the extra ``stable_frames`` delay.
    """

    def __init__(self, alpha_up: float = 0.85, alpha_down: float = 0.5) -> None:
        self.alpha_up = min(1.0, max(0.0, alpha_up))
        self.alpha_down = min(1.0, max(0.0, alpha_down))
        self._value = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
            return raw
        a = self.alpha_up if raw > self._value else self.alpha_down
        self._value = a * raw + (1.0 - a) * self._value
        return self._value

    @property
    def value(self):
        return self._value

    def reset(self) -> None:
        self._value = None


def make_pointer_filter(cfg: dict):
    """Build the pointer smoother honoring pointer/filter settings.

    cfg: the ``pointer`` section of settings.yaml — ``filter: one_euro|ema``.
    Falls back to EMA on unknown names so a typo never kills the loop.
    """
    if cfg.get("filter", "ema") == "one_euro":
        params = cfg.get("one_euro", {}) or {}
        return OneEuroXY(
            min_cutoff=params.get("min_cutoff", 1.7),
            beta=params.get("beta", 0.07),
            d_cutoff=params.get("d_cutoff", 1.0),
        )
    return EMAFilter(alpha=cfg.get("ema_alpha", 0.35))


class EMAFilter:
    """Exponential moving average — reduces jitter while staying responsive."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._value = None

    def update(self, x: float, y: float, t: float | None = None):
        if self._value is None:
            self._value = [x, y]
        else:
            self._value[0] = self.alpha * x + (1 - self.alpha) * self._value[0]
            self._value[1] = self.alpha * y + (1 - self.alpha) * self._value[1]
        return self._value[0], self._value[1]

    def reset(self) -> None:
        self._value = None


class DeadZone:
    """Ignore movement smaller than threshold (pixels).

    A hard dead-zone makes motion steppy (frames are swallowed, then the
    cursor jumps). Prefer threshold 0 and let the 1€ filter remove rest
    jitter continuously — the cursor then glides instead of ticking.
    """

    def __init__(self, threshold_px: float = 0.0) -> None:
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

    def reset(self) -> None:
        self._last = None


class JumpGuard:
    """Drop frames where cursor teleports (tracking glitch).

    Reseeds after ``max_rejects`` consecutive drops: a sustained teleport
    means the hand genuinely moved there, so the cursor must recover
    instead of freezing at the stale position forever.

    Velocity gating: a jump that continues the current motion direction
    (fast intentional flick) is accepted immediately instead of being
    swallowed for ``max_rejects`` frames — that swallowing is what made
    fast moves feel laggy. Random-teleport glitches point anywhere, so
    they still get rejected.
    """

    def __init__(
        self,
        max_jump_ratio: float,
        screen_w: int,
        screen_h: int,
        max_rejects: int = 5,
        align_cos: float = 0.6,
    ) -> None:
        self.max_dist = max_jump_ratio * ((screen_w**2 + screen_h**2) ** 0.5)
        self.max_rejects = max(1, max_rejects)
        self.align_cos = min(1.0, max(-1.0, align_cos))
        self._last = None
        self._rejects = 0
        self._vx = 0.0
        self._vy = 0.0

    def accept(self, x: float, y: float) -> bool:
        if self._last is None:
            self._last = (x, y)
            return True

        dx = x - self._last[0]
        dy = y - self._last[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_dist:
            # Fast flick continuing current direction? Let it through.
            vmag = (self._vx**2 + self._vy**2) ** 0.5
            if vmag > 1e-9 and dist > 1e-9:
                cos = (dx * self._vx + dy * self._vy) / (dist * vmag)
                if cos >= self.align_cos and vmag * 0.05 > self.max_dist * 0.5:
                    self._last = (x, y)
                    self._vx = 0.9 * self._vx + 0.1 * (dx * 30.0)
                    self._vy = 0.9 * self._vy + 0.1 * (dy * 30.0)
                    self._rejects = 0
                    return True
            self._rejects += 1
            if self._rejects >= self.max_rejects:
                # genuine move, not a glitch — reseed here
                self._last = (x, y)
                self._rejects = 0
                return True
            return False

        self._last = (x, y)
        self._rejects = 0
        # Track motion direction at ~30 fps for the flick-through check.
        self._vx = 0.8 * self._vx + 0.2 * (dx * 30.0)
        self._vy = 0.8 * self._vy + 0.2 * (dy * 30.0)
        return True

    def reset(self) -> None:
        self._last = None
        self._rejects = 0
        self._vx = 0.0
        self._vy = 0.0


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
