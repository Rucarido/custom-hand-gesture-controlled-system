"""StillLock tests — hardware-free (no cv2/mediapipe/pynput).

Battle-cursor auto-lock: ~1 s whole-hand stillness -> "lock", ~2 s of
sustained motion -> "unlock". Pinch exemption is caller-side (game_mode
skips update() while pinching), so these tests cover the class contract:
lock timing, motion-meter unlock, spike/drain behavior, cooldown.
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
        unlock_frames=6,
        unlock_decay=3,
        relock_cooldown_frames=15,
        velocity_alpha=0.35,
    )
    kw.update(over)
    return StillLock(**kw)


def _lock_it(lock, x=0.5, y=0.5, frames=10):
    for _ in range(frames):
        lock.update(_still(x, y))
    assert lock.is_locked()


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
        # drift every other frame: still counter never completes
        x = 0.5 + (0.02 if i % 2 == 0 else 0.0)
        assert lock.update(_still(x)) is None
    assert not lock.is_locked()


def test_thumb_motion_alone_blocks_lock():
    # Shaping a pinch (thumb travelling, rest steady) must NOT lock.
    # Extra class-level safety on top of game_mode skipping pinch frames.
    lock = _new()
    events = []
    for i in range(20):
        t = 0.5 + (0.01 if i % 2 == 0 else -0.01)
        events.append(lock.update(_lm(0.5, 0.5, tx=t, ty=0.5)))
    assert "lock" not in events
    assert not lock.is_locked()


def test_sustained_motion_unlocks():
    lock = _new()
    _lock_it(lock)
    events = []
    x = 0.5
    for _ in range(12):
        x += 0.02  # deliberate continuous travel
        events.append(lock.update(_still(x)))
    assert "unlock" in events
    assert not lock.is_locked()


def test_single_spike_does_not_unlock():
    lock = _new()
    _lock_it(lock)
    # one out-and-back jump, then dead still: meter peaks well under 6
    assert lock.update(_still(0.53)) is None
    assert lock.update(_still(0.5)) is None
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


def test_move_and_hold_does_not_unlock():
    # 3 motion frames (~2 votes) then still drains the meter; repeating
    # that pattern must never accumulate to unlock_frames=6.
    lock = _new()
    _lock_it(lock)
    x = 0.5
    for _ in range(6):
        for _ in range(3):
            x += 0.02
            assert lock.update(_still(x)) is None
        for _ in range(6):
            assert lock.update(_still(x)) is None
    assert lock.is_locked()


def test_relock_cooldown_then_relock():
    lock = _new()
    _lock_it(lock)
    x = 0.5
    for _ in range(12):
        x += 0.02
        lock.update(_still(x))
        if not lock.is_locked():
            break
    assert not lock.is_locked()  # released by motion; cooldown now full
    # ... hand stops dead: cooldown suppresses snap re-lock ...
    for _ in range(14):
        assert lock.update(_still(x)) is None
    assert not lock.is_locked()
    # ... then normal stillness locks again.
    events = [lock.update(_still(x)) for _ in range(8)]
    assert "lock" in events
    assert lock.is_locked()


def test_reset_clears_everything():
    lock = _new()
    _lock_it(lock)
    lock.reset()
    assert not lock.is_locked()
    assert lock.move_progress == 0.0
    # fresh instance behavior: first frame seeds, no crash, no lock
    assert lock.update(_still()) is None
