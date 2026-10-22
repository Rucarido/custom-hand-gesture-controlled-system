"""StillLock tests — hardware-free (no cv2/mediapipe/pynput).

Battle-cursor auto-lock contract: ~1 s whole-hand stillness -> "lock";
then TWO release paths — (a) displacement: index tip drifts
``unlock_displacement`` from the lock anchor (ANY speed — the old
wave-meter-only design stalled forever on slow repositioning), confirmed
``unlock_confirm_frames`` running so spikes never confirm; (b) waving:
sustained motion with a gentle x1 drain (the old x3 drain canceled all
progress on any pause, so the meter needed 75% moving frames and simply
never fired for many users). Pinch exemption is caller-side (game_mode
skips updates until the pinch channel is fully idle).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from click_lock import StillLock  # noqa: E402


def _lm(x, y, tx=None, ty=None, wx=None, wy=None):
    """21-landmark stub; index(8)/thumb(4)/wrist(0) placed, rest centered."""
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    lm[8] = [x, y, 0.0]
    lm[4] = [x if tx is None else tx, y if ty is None else ty, 0.0]
    lm[0] = [x if wx is None else wx, y if wy is None else wy, 0.0]
    return lm


def _still(x=0.5, y=0.5):
    return _lm(x, y)


def _new(**over):
    kw = dict(
        stillness_threshold=0.003,
        still_frames=5,
        motion_threshold=0.008,
        unlock_frames=50,      # wave path ~off unless a test opts in
        unlock_decay=1,
        unlock_displacement=0.06,
        unlock_confirm_frames=3,
        relock_cooldown_frames=15,
        velocity_alpha=0.35,
    )
    kw.update(over)
    return StillLock(**kw)


def _lock_it(lock, x=0.5, y=0.5, frames=10):
    for _ in range(frames):
        lock.update(_still(x, y))
    assert lock.is_locked()


# --- locking -----------------------------------------------------------------


def test_locks_after_sustained_stillness():
    lock = _new()
    events = [lock.update(_still()) for _ in range(3)]
    assert events == [None] * 3
    assert not lock.is_locked()
    events = [lock.update(_still()) for _ in range(8)]
    assert "lock" in events
    assert lock.is_locked()


def test_motion_prevents_lock():
    lock = _new()
    for i in range(30):
        x = 0.5 + (0.02 if i % 2 == 0 else 0.0)
        assert lock.update(_still(x)) is None
    assert not lock.is_locked()


def test_thumb_motion_alone_blocks_lock():
    # Shaping a pinch (thumb travelling, rest steady) must NOT lock.
    lock = _new()
    events = []
    for i in range(20):
        t = 0.5 + (0.01 if i % 2 == 0 else -0.01)
        events.append(lock.update(_lm(0.5, 0.5, tx=t, ty=0.5)))
    assert "lock" not in events
    assert not lock.is_locked()


# --- unlock path (a): displacement, speed-independent -------------------------


def test_slow_repositioning_unlocks():
    """THE regression: drift slower than motion_threshold must still unlock.

    Old wave-meter-only design: EMA speed 0.005 < 0.008 -> zero motion
    votes -> meter stayed 0 forever -> cursor parked while the hand
    travelled away and pinch-clicked the WRONG spot.
    """
    lock = _new()
    _lock_it(lock, x=0.5)
    x = 0.5
    events = []
    for _ in range(20):
        x += 0.005  # deliberate slow travel, below the wave threshold
        events.append(lock.update(_still(x)))
    assert "unlock" in events
    assert not lock.is_locked()


def test_far_move_and_hold_unlocks():
    # Reposition to a new spot and park there: unlock (that's the point —
    # the hand is clearly somewhere else; the cursor must follow again).
    lock = _new()
    _lock_it(lock, x=0.5)
    x = 0.5
    events = []
    for _ in range(3):
        x += 0.02
        events.append(lock.update(_still(x)))   # d reaches 0.06
    for _ in range(5):
        events.append(lock.update(_still(x)))   # hold far -> confirm
    assert "unlock" in events
    assert not lock.is_locked()


def test_displacement_spike_does_not_unlock():
    # Out-and-back jumps never keep 3 confirmation votes in a row.
    lock = _new()
    _lock_it(lock, x=0.5)
    events = []
    for _ in range(5):  # 2 frames far (d=0.08), 2 frames back home
        events.append(lock.update(_still(0.58)))
        events.append(lock.update(_still(0.58)))
        events.append(lock.update(_still(0.5)))
        events.append(lock.update(_still(0.5)))
    assert "unlock" not in events
    assert lock.is_locked()


# --- unlock path (b): sustained waving ----------------------------------------


def test_sustained_waving_unlocks():
    # Oscillation IN PLACE (no net displacement): wave meter must fire.
    lock = _new(unlock_frames=5)
    _lock_it(lock)
    events = []
    x = 0.5
    for i in range(15):
        x = 0.5 + (0.02 if i % 2 == 0 else -0.02)
        events.append(lock.update(_still(x)))
    assert "unlock" in events
    assert not lock.is_locked()


def test_single_spike_does_not_unlock():
    lock = _new()
    _lock_it(lock)
    assert lock.update(_still(0.53)) is None  # lone jump out...
    assert lock.update(_still(0.5)) is None  # ...and back
    for _ in range(10):
        assert lock.update(_still(0.5)) is None
    assert lock.is_locked()


def test_still_while_locked_stays_locked():
    lock = _new()
    _lock_it(lock)
    for _ in range(20):
        assert lock.update(_still()) is None
    assert lock.is_locked()
    assert lock.move_progress == 0.0


# --- cooldown / reset ----------------------------------------------------------


def test_relock_cooldown_then_relock():
    lock = _new(unlock_frames=5)
    _lock_it(lock)
    x = 0.5
    unlocked_at = None
    for i in range(20):  # wave in place until the unlock fires
        x = 0.5 + (0.02 if i % 2 == 0 else -0.02)
        if lock.update(_still(x)) == "unlock":
            unlocked_at = i
            break
    assert unlocked_at is not None
    # ... hand stops dead: cooldown (15) suppresses snap re-lock ...
    for _ in range(14):
        assert lock.update(_still(x)) is None
    assert not lock.is_locked()
    # ... then normal stillness locks again at the new spot.
    relock = [lock.update(_still(x)) for _ in range(8)]
    assert "lock" in relock
    assert lock.is_locked()


def test_reset_clears_everything():
    lock = _new()
    _lock_it(lock)
    lock.reset()
    assert not lock.is_locked()
    assert lock.move_progress == 0.0
    assert lock.update(_still()) is None
