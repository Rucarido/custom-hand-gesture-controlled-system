"""Click-lock: freeze the cursor while the hand holds still to click.

Recognition design (whole-hand, not single-point):

* Stillness is judged on the index tip AND thumb tip AND wrist together —
  the cursor only freezes when the *whole hand* is deliberately held.
  Tracking one point caused false locks while shaping a pinch (index
  steady, thumb moving) or drifting the arm (tip steady in frame, wrist
  travelling). All three must read still.
* Each point runs its own velocity EMA, so one noisy frame neither locks
  nor resets the count; the hold needs ~1.1 s of sustained stillness
  (``stillness_frames``), not a brief aiming pause.
* Unlock needs *confirmed* motion: displacement past ``unlock_threshold``
  for ``unlock_confirm_frames`` in a row, so a single jitter spike can't
  break the lock but a deliberate move releases within ~60 ms. The
  lateral shake gesture (either axis) remains as a fallback.
* After an unlock, re-locking is suppressed for
  ``relock_cooldown_frames`` so the cursor doesn't snap frozen again.

``StillLock`` (below) is the battle-cursor variant with slower,
deliberate timing: lock after ~1 s of stillness, unlock only after ~2 s
of *sustained* motion. Pinch exemption is caller-side: game_mode simply
does not call ``update()`` while a pinch is in progress, so shaping or
holding a pinch can never break (or make) the lock — the click still
fires at the frozen position.
"""

import landmarks as L


class ClickLock:
    UNLOCKED = "unlocked"
    LOCKED = "locked"

    # Points that must all agree the hand is still: cursor tip, pinch
    # finger, and arm anchor. Indices into the 21-landmark list.
    _STILL_POINTS = (L.INDEX_TIP, L.THUMB_TIP, L.WRIST)

    def __init__(
        self,
        stillness_threshold: float = 0.003,
        stillness_frames: int = 32,
        shake_buffer: int = 8,
        shake_min_changes: int = 3,
        unlock_threshold: float = 0.012,
        unlock_confirm_frames: int = 2,
        relock_cooldown_frames: int = 20,
        velocity_alpha: float = 0.35,
    ) -> None:
        self.stillness_threshold = stillness_threshold
        self.stillness_frames = max(1, stillness_frames)
        self.shake_buffer = max(2, shake_buffer)
        self.shake_min_changes = max(1, shake_min_changes)
        self.unlock_threshold = max(0.0, unlock_threshold)
        self.unlock_confirm_frames = max(1, unlock_confirm_frames)
        self.relock_cooldown_frames = max(0, relock_cooldown_frames)
        self.velocity_alpha = min(1.0, max(0.0, velocity_alpha))

        self.state = self.UNLOCKED
        self._still_count = 0
        self._prev = None  # previous (x, y) per tracked point
        self._vel = [0.0] * len(self._STILL_POINTS)  # per-point speed EMA
        self._dir_buffer: list[tuple[int, int]] = []
        self._lock_pos = None
        self._unlock_votes = 0
        self._cooldown_left = 0

    @classmethod
    def _points(cls, landmarks):
        return [(landmarks[i][0], landmarks[i][1]) for i in cls._STILL_POINTS]

    def update(self, landmarks) -> str | None:
        pts = self._points(landmarks)

        if self._prev is None:
            self._prev = pts
            return None

        speeds = []
        for (x, y), (px, py) in zip(pts, self._prev):
            dx, dy = x - px, y - py
            speeds.append((dx * dx + dy * dy) ** 0.5)
        # Per-point velocity EMA: one jitter spike can't lock or reset.
        self._vel = [
            self.velocity_alpha * s + (1.0 - self.velocity_alpha) * v
            for s, v in zip(speeds, self._vel)
        ]
        self._prev = pts

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if self.state == self.UNLOCKED:
            if self._cooldown_left > 0:
                self._still_count = 0
                return None
            if max(self._vel) < self.stillness_threshold:
                self._still_count += 1
                if self._still_count >= self.stillness_frames:
                    self.state = self.LOCKED
                    self._still_count = 0
                    self._lock_pos = pts[0]  # cursor anchor = index tip
                    self._dir_buffer.clear()
                    self._unlock_votes = 0
                    return "lock"
            else:
                self._still_count = 0
            return None

        # LOCKED — release on confirmed deliberate motion; a lone spike is
        # absorbed by the vote counter. Shake (either axis) is the fallback.
        if self._lock_pos is not None:
            lx = pts[0][0] - self._lock_pos[0]
            ly = pts[0][1] - self._lock_pos[1]
            if (lx * lx + ly * ly) ** 0.5 >= self.unlock_threshold:
                self._unlock_votes += 1
                if self._unlock_votes >= self.unlock_confirm_frames:
                    return self._unlock(pts)
            else:
                self._unlock_votes = 0

        # Shake uses raw per-frame deltas on both axes (index tip motion).
        direction = self._shake_dir(pts)
        self._dir_buffer.append(direction)
        if len(self._dir_buffer) > self.shake_buffer:
            self._dir_buffer.pop(0)

        if self._shake_changes() >= self.shake_min_changes:
            return self._unlock(pts)
        return None

    def _shake_dir(self, pts) -> tuple[int, int]:
        # Direction of the latest index-tip step; needs the pre-update
        # position, tracked separately from the stillness state.
        cur = pts[0]
        prev = getattr(self, "_shake_prev", None)
        self._shake_prev = cur
        if prev is None:
            return (0, 0)

        def axis(d: float) -> int:
            if d > 0.001:
                return 1
            if d < -0.001:
                return -1
            return 0

        return (axis(cur[0] - prev[0]), axis(cur[1] - prev[1]))

    def _shake_changes(self) -> int:
        changes = 0
        for i in range(1, len(self._dir_buffer)):
            a = self._dir_buffer[i - 1]
            b = self._dir_buffer[i]
            for ca, cb in zip(a, b):
                if ca != 0 and cb != 0 and ca != cb:
                    changes += 1
                    break
        return changes

    def _unlock(self, pts) -> str:
        self.state = self.UNLOCKED
        self._dir_buffer.clear()
        self._still_count = 0
        self._vel = [0.0] * len(self._STILL_POINTS)
        self._lock_pos = None
        self._unlock_votes = 0
        self._cooldown_left = self.relock_cooldown_frames
        self._prev = pts
        self._shake_prev = pts[0]
        return "unlock"

    def is_locked(self) -> bool:
        return self.state == self.LOCKED

    def reset(self) -> None:
        self.state = self.UNLOCKED
        self._still_count = 0
        self._prev = None
        self._vel = [0.0] * len(self._STILL_POINTS)
        self._dir_buffer.clear()
        self._lock_pos = None
        self._unlock_votes = 0
        self._cooldown_left = 0
        if hasattr(self, "_shake_prev"):
            del self._shake_prev


class StillLock:
    """Auto-lock the battle cursor on stillness; slow deliberate unlock.

    Same whole-hand signal as ClickLock (index tip + thumb tip + wrist,
    per-point velocity EMA in normalized landmark units per frame), but
    timing tuned for "park the cursor, then click in peace":

    * UNLOCKED: all three points' EMA speed below ``stillness_threshold``
      for ``still_frames`` in a row (~30 @30fps = 1 s) -> "lock".
    * LOCKED: frames whose EMA speed reaches ``motion_threshold`` add +1
      to a motion meter; still frames drain it by ``unlock_decay``. Meter
      reaching ``unlock_frames`` (~60 = 2 s of good movement) -> "unlock".
      A lone jitter spike contributes at most +1 and drains away, so it
      can never unlock alone.
    * After unlock, ``relock_cooldown_frames`` suppress re-locking so the
      tail of the releasing motion doesn't snap frozen again.
    * Pinch exemption is caller-side: do not call ``update()`` while a
      pinch is held/confirmed. Pinch shaping moves thumb+index a lot and
      would otherwise feed the motion meter; skipping keeps pinch purely
      a click action at the frozen cursor.
    """

    UNLOCKED = "unlocked"
    LOCKED = "locked"

    _STILL_POINTS = (L.INDEX_TIP, L.THUMB_TIP, L.WRIST)

    def __init__(
        self,
        stillness_threshold: float = 0.003,
        still_frames: int = 30,
        motion_threshold: float = 0.008,
        unlock_frames: int = 60,
        unlock_decay: int = 3,
        relock_cooldown_frames: int = 15,
        velocity_alpha: float = 0.35,
    ) -> None:
        self.stillness_threshold = stillness_threshold
        self.still_frames = max(1, still_frames)
        self.motion_threshold = max(0.0, motion_threshold)
        self.unlock_frames = max(1, unlock_frames)
        self.unlock_decay = max(1, unlock_decay)
        self.relock_cooldown_frames = max(0, relock_cooldown_frames)
        self.velocity_alpha = min(1.0, max(0.0, velocity_alpha))

        self.state = self.UNLOCKED
        self._still_count = 0
        self._move_meter = 0
        self._prev = None
        self._vel = [0.0] * len(self._STILL_POINTS)
        self._cooldown_left = 0

    @classmethod
    def _points(cls, landmarks):
        return [(landmarks[i][0], landmarks[i][1]) for i in cls._STILL_POINTS]

    def update(self, landmarks) -> str | None:
        pts = self._points(landmarks)

        if self._prev is None:
            self._prev = pts
            return None

        speeds = []
        for (x, y), (px, py) in zip(pts, self._prev):
            dx, dy = x - px, y - py
            speeds.append((dx * dx + dy * dy) ** 0.5)
        # Per-point velocity EMA: one jitter spike neither locks nor unlocks.
        self._vel = [
            self.velocity_alpha * s + (1.0 - self.velocity_alpha) * v
            for s, v in zip(speeds, self._vel)
        ]
        self._prev = pts

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if self.state == self.UNLOCKED:
            if self._cooldown_left > 0:
                self._still_count = 0
                return None
            if max(self._vel) < self.stillness_threshold:
                self._still_count += 1
                if self._still_count >= self.still_frames:
                    self.state = self.LOCKED
                    self._still_count = 0
                    self._move_meter = 0
                    return "lock"
            else:
                self._still_count = 0
            return None

        # LOCKED — sustained motion only. Still frames drain the meter, so
        # move-and-hold or lone spikes stall well short of unlock_frames.
        if max(self._vel) >= self.motion_threshold:
            self._move_meter += 1
            if self._move_meter >= self.unlock_frames:
                self.state = self.UNLOCKED
                self._move_meter = 0
                self._still_count = 0
                self._vel = [0.0] * len(self._STILL_POINTS)
                self._cooldown_left = self.relock_cooldown_frames
                return "unlock"
        else:
            self._move_meter = max(0, self._move_meter - self.unlock_decay)
        return None

    def is_locked(self) -> bool:
        return self.state == self.LOCKED

    @property
    def move_progress(self) -> float:
        """0..1 unlock-meter fill for the overlay (meaningful when locked)."""
        return min(1.0, self._move_meter / self.unlock_frames)

    def reset(self) -> None:
        self.state = self.UNLOCKED
        self._still_count = 0
        self._move_meter = 0
        self._prev = None
        self._vel = [0.0] * len(self._STILL_POINTS)
        self._cooldown_left = 0
