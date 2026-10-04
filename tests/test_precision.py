"""Precision tests — hardware-free (no cv2/mediapipe/pyautogui).

Hardware-facing modules (tracker/camera/main) are excluded by design — see
tests/conftest.py. Everything imported here is pure logic.

Covers the sharp-tracking overhaul:
  - scale-invariant pinch ratio (near/far hand, same pose -> same ratio)
  - pinch hysteresis (flutter at threshold -> exactly one click)
  - curl-based index gate (bent finger rejected where wrist-distance passes)
  - OneEuro adaptivity (jitter killed at rest, steps tracked fast)
  - JumpGuard reseed (glitch storm doesn't freeze cursor forever)
"""

import math
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import landmarks as L  # noqa: E402  (src/ on path via bootstrap above)
from filters import JumpGuard, OneEuroXY  # noqa: E402
from gestures_click import PinchClickDetector  # noqa: E402


def make_hand(*, wrist=(0.5, 0.5), scale=0.2, thumb_gap=0.05, index_curl="open"):
    """Synthetic 21-landmark hand in normalized coords.

    index_curl: "open" (straight up) | "bent" (curled, tip far laterally).
    thumb_gap: thumb-tip distance from index-tip as fraction of scale.
    """
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    wx, wy = wrist
    lm[L.WRIST] = [wx, wy, 0.0]
    # middle MCP defines hand size: straight above wrist
    lm[L.MIDDLE_MCP] = [wx, wy - scale, 0.0]

    ix = wx + 0.25 * scale
    if index_curl == "open":
        lm[L.INDEX_MCP] = [ix, wy - 0.45 * scale, 0.0]
        lm[L.INDEX_PIP] = [ix, wy - 0.70 * scale, 0.0]
        lm[L.INDEX_DIP] = [ix, wy - 0.90 * scale, 0.0]
        lm[L.INDEX_TIP] = [ix, wy - 1.10 * scale, 0.0]
    else:  # bent: tip curled sideways, still far from wrist (old check passes)
        lm[L.INDEX_MCP] = [ix, wy - 0.45 * scale, 0.0]
        lm[L.INDEX_PIP] = [ix + 0.10 * scale, wy - 0.62 * scale, 0.0]
        lm[L.INDEX_DIP] = [ix + 0.32 * scale, wy - 0.58 * scale, 0.0]
        lm[L.INDEX_TIP] = [ix + 0.38 * scale, wy - 0.38 * scale, 0.0]

    tip = lm[L.INDEX_TIP]
    lm[L.THUMB_TIP] = [tip[0] + thumb_gap * scale, tip[1], 0.0]
    lm[L.THUMB_IP] = [tip[0] + 0.3 * scale, tip[1] + 0.3 * scale, 0.0]
    lm[L.THUMB_MCP] = [wx - 0.2 * scale, wy - 0.1 * scale, 0.0]
    lm[L.THUMB_CMC] = [wx - 0.3 * scale, wy, 0.0]
    return lm


# --- pinch ratio: scale invariance ----------------------------------------


def test_pinch_ratio_scale_invariant():
    near = make_hand(scale=0.15, thumb_gap=0.1)
    far = make_hand(scale=0.30, thumb_gap=0.1)
    assert L.pinch_ratio(near) == pytest.approx(L.pinch_ratio(far), rel=1e-6)
    # ... while the raw pixel distance differs ~2x (the old failure mode)
    assert L.tip_distance(near) == pytest.approx(L.tip_distance(far) / 2, rel=0.05)


def test_pinch_ratio_touching_vs_open():
    touching = make_hand(thumb_gap=0.05)
    open_hand = make_hand(thumb_gap=0.8)
    assert L.pinch_ratio(touching) < 0.25
    assert L.pinch_ratio(open_hand) > 0.40


# --- pinch hysteresis ------------------------------------------------------


def test_pinch_flutter_fires_once():
    det = PinchClickDetector(pinch_on=0.25, pinch_off=0.40, stable_frames=3,
                             smooth_alpha_up=1.0, smooth_alpha_down=1.0)
    # hold a solid pinch -> exactly one click
    fires = [det.update(make_hand(thumb_gap=0.05)) for _ in range(6)]
    assert fires.count("click") == 1
    # flutter just under/around the on-threshold must NOT re-fire ...
    for _ in range(6):
        assert det.update(make_hand(thumb_gap=0.9)) is None  # wide open re-arms
    fires2 = [det.update(make_hand(thumb_gap=0.05)) for _ in range(6)]
    assert fires2.count("click") == 1


def test_pinch_needs_stable_frames():
    det = PinchClickDetector(pinch_on=0.25, pinch_off=0.40, stable_frames=3,
                             smooth_alpha_up=1.0, smooth_alpha_down=1.0)
    assert det.update(make_hand(thumb_gap=0.05)) is None
    assert det.update(make_hand(thumb_gap=0.05)) is None
    assert det.update(make_hand(thumb_gap=0.9)) is None  # blip resets count
    assert det.update(make_hand(thumb_gap=0.05)) is None
    assert det.update(make_hand(thumb_gap=0.05)) is None
    assert det.update(make_hand(thumb_gap=0.05)) == "click"


# --- index gate: curl beats wrist-distance ---------------------------------


def test_bent_finger_rejected_where_old_check_passes():
    bent = make_hand(index_curl="bent")
    wrist, tip, pip = bent[L.WRIST], bent[L.INDEX_TIP], bent[L.INDEX_PIP]

    def old_check():
        td = (tip[0] - wrist[0]) ** 2 + (tip[1] - wrist[1]) ** 2
        pd = (pip[0] - wrist[0]) ** 2 + (pip[1] - wrist[1]) ** 2
        return td > pd

    assert old_check() is True  # old logic fooled ...
    assert L.is_index_extended(bent) is False  # ... new logic is not
    assert L.is_index_extended(make_hand(index_curl="open")) is True


# --- OneEuro: still jitter down, motion lag low ----------------------------


def test_one_euro_kills_rest_jitter_but_tracks_steps():
    import random

    random.seed(7)
    f = OneEuroXY(min_cutoff=1.2, beta=0.008, d_cutoff=1.0, freq=30.0)
    rest = [(500 + random.uniform(-1.5, 1.5), 300 + random.uniform(-1.5, 1.5))
            for _ in range(60)]
    # Explicit 30 fps stamps — wall clock is meaningless in a tight loop.
    outs = [f.update(x, y, t=i / 30.0) for i, (x, y) in enumerate(rest)]
    tail = outs[30:]
    resid = [math.dist(p, (500, 300)) for p in tail]
    assert sum(resid) / len(resid) < 0.6  # jitter crushed, sub-pixel-ish

    f2 = OneEuroXY(min_cutoff=1.2, beta=0.5, d_cutoff=1.0, freq=30.0)
    for i in range(30):
        f2.update(100.0, 100.0, t=i / 30.0)
    px, py = f2.update(600.0, 400.0, t=30 / 30.0)
    assert math.dist((px, py), (100, 100)) > 50  # moves promptly, no heavy lag


# --- JumpGuard reseed -------------------------------------------------------


def test_jump_guard_reseeds_after_glitch_storm():
    jg = JumpGuard(max_jump_ratio=0.10, screen_w=1920, screen_h=1080)
    assert jg.accept(100.0, 100.0) is True
    for _ in range(4):  # teleport storm (tracking glitch)
        assert jg.accept(1800.0, 900.0) is False
    # ... then the hand genuinely stays there: cursor must recover
    assert jg.accept(1800.0, 900.0) is True  # 5th strike reseeds
    assert jg.accept(1800.0, 900.0) is True


# --- hand selection: preference + largest-fallback ---------------------------


def test_choose_hand_prefers_and_falls_back():
    small = make_hand(scale=0.12)
    big = make_hand(scale=0.28)
    assert L.choose_hand_index([small, big], ["Left", "Right"], "Right") == 1
    assert L.choose_hand_index([small, big], ["Left", "Right"], "Left") == 0
    assert L.choose_hand_index([small, big], ["Left", "Right"], "") == 1
    assert L.choose_hand_index([], [], "Right") == -1


# --- pointer: gate hysteresis + stabilized tip --------------------------------


def test_pointer_gate_rides_out_single_bent_frame():
    from pointer import PointerMapper

    pm = PointerMapper(
        mirror=False,
        motion_area={"enabled": False},
        screen_size=(1000, 1000),
        gate_release_frames=3,
        coast_frames=0,  # isolate the gate: no coast hold in this test
    )
    open_hand = make_hand(index_curl="open")
    bent_hand = make_hand(index_curl="bent")
    assert pm.landmarks_to_screen(open_hand) is not None
    assert pm.landmarks_to_screen(bent_hand) is not None  # 1-frame dip rides
    assert pm.landmarks_to_screen(bent_hand) is not None  # 2-frame dip rides
    assert pm.landmarks_to_screen(bent_hand) is None  # genuinely folded
    assert pm.landmarks_to_screen(open_hand) is not None  # reopens instantly


def test_pointer_tip_blend_sits_between_tip_and_dip():
    from pointer import PointerMapper

    pm = PointerMapper(
        mirror=False,
        require_index_extended=False,
        motion_area={"enabled": False},
        screen_size=(1000, 1000),
    )
    hand = make_hand(index_curl="open")
    pos = pm.landmarks_to_screen(hand)
    assert pos is not None
    sx, sy = pos
    tip = hand[L.INDEX_TIP]
    dip = hand[L.INDEX_DIP]
    assert min(tip[0], dip[0]) * 1000 - 1 <= sx <= max(tip[0], dip[0]) * 1000 + 1
    assert sy < dip[1] * 1000  # blend pulls slightly toward the dip (down)


# --- camera backend map --------------------------------------------------------


def test_resolve_backend_names():
    from camera import _BACKENDS, resolve_backend

    assert resolve_backend("dshow") == _BACKENDS["dshow"]
    assert resolve_backend("v4l2") == _BACKENDS["v4l2"]
    assert resolve_backend("avfoundation") == _BACKENDS["avfoundation"]
    assert resolve_backend("bogus") == _BACKENDS["any"]
    assert resolve_backend("auto") in tuple(_BACKENDS.values())


# --- pointer coast: blink-length gate loss holds position ----------------------


def test_pointer_coast_holds_through_brief_fold():
    from pointer import PointerMapper

    pm = PointerMapper(
        mirror=False,
        motion_area={"enabled": False},
        screen_size=(1000, 1000),
        gate_release_frames=2,
        coast_frames=5,
    )
    open_hand = make_hand(index_curl="open")
    bent_hand = make_hand(index_curl="bent")
    first = pm.landmarks_to_screen(open_hand)
    assert first is not None
    # gate still open on the 1st folded frame: live (bent) position
    second = pm.landmarks_to_screen(bent_hand)
    assert second is not None
    # gate closed on the 2nd folded frame, but coast holds last live pos
    held = pm.landmarks_to_screen(bent_hand)
    assert held == second
    # ... and reopening resumes live tracking instantly
    assert pm.landmarks_to_screen(open_hand) is not None


# --- predictor: leads motion, never leaps on stop ------------------------------


def test_predictor_leads_and_clamps():
    from filters import MotionPredictor

    p = MotionPredictor(lead_s=0.024, velocity_alpha=1.0, max_lead_px=48.0)
    t = 1000.0
    x, y = 100.0, 100.0
    # steady 1000 px/s diagonal motion at 30 fps
    for i in range(10):
        t += 1 / 30.0
        x, y = x + 1000 / 30.0, y + 1000 / 30.0
        px, py = p.update(x, y, t=t)
    # prediction leads along motion, bounded by max_lead_px
    assert px > x and py > y
    assert math.dist((px, py), (x, y)) <= 48.0 + 1e-6
    # sudden stop: next update at same spot predicts (near) no leap
    px2, py2 = p.update(x, y, t=t + 1 / 30.0)
    assert math.dist((px2, py2), (x, y)) <= 48.0 + 1e-6


# --- asymmetric smoother: slow to engage, fast to release ----------------------


def test_asymmetric_smoother_direction_rates():
    from filters import AsymmetricSmoother

    s = AsymmetricSmoother(alpha_up=0.85, alpha_down=0.5)
    assert s.update(1.0) == 1.0
    # falling edge uses alpha_down: one step moves only halfway
    assert s.update(0.0) == pytest.approx(0.5)
    # rising edge uses alpha_up: snaps back most of the way
    assert s.update(1.0) == pytest.approx(0.5 + 0.85 * 0.5)


# --- pinch smoothing: borderline jitter still clicks, once -----------------------


def test_pinch_smoothing_rides_out_borderline_jitter():
    from gestures_click import PinchClickDetector

    det = PinchClickDetector(pinch_on=0.25, pinch_off=0.40, stable_frames=2)
    seq = [
        make_hand(thumb_gap=0.05),  # solid pinch
        make_hand(thumb_gap=0.30),  # jitter into the hysteresis band
        make_hand(thumb_gap=0.05),
        make_hand(thumb_gap=0.05),
        make_hand(thumb_gap=0.05),
        make_hand(thumb_gap=0.05),
    ]
    fires = [det.update(h) for h in seq]
    assert fires.count("click") == 1
    assert fires.index("click") <= 3  # no extra delay from the noisy frame


# --- click lock: deliberate hold locks, any motion unlocks ---------------------


def _lock_landmarks(ix, iy, tx=None, ty=None, wx=None, wy=None):
    """21-landmark stub with index/thumb/wrist placed; rest centered."""
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    lm[8] = [ix, iy, 0.0]
    lm[4] = [ix if tx is None else tx, iy if ty is None else ty, 0.0]
    lm[0] = [ix if wx is None else wx, iy if wy is None else wy, 0.0]
    return lm


def _new_lock(**over):
    from click_lock import ClickLock

    kw = dict(
        stillness_threshold=0.003,
        stillness_frames=32,
        unlock_threshold=0.012,
        unlock_confirm_frames=2,
        relock_cooldown_frames=20,
    )
    kw.update(over)
    return ClickLock(**kw)


def test_click_lock_needs_sustained_stillness_then_easy_unlock():
    lock = _new_lock()
    # brief aiming pause (15 frames) must NOT lock
    events = [lock.update(_lock_landmarks(0.5, 0.5)) for _ in range(15)]
    assert events == [None] * 15
    assert not lock.is_locked()
    # sustained deliberate hold locks (~1.1 s)
    events = [lock.update(_lock_landmarks(0.5, 0.5)) for _ in range(30)]
    assert "lock" in events
    assert lock.is_locked()
    # confirmed deliberate motion unlocks (2 frames running)
    assert lock.update(_lock_landmarks(0.5 + 0.02, 0.5)) is None  # 1st vote
    assert lock.update(_lock_landmarks(0.5 + 0.02, 0.5)) == "unlock"  # 2nd
    assert not lock.is_locked()
    # ... and it must not snap locked again right away
    for _ in range(10):
        assert lock.update(_lock_landmarks(0.52, 0.5)) is None
    assert not lock.is_locked()


def test_click_lock_ignores_moving_thumb_while_aiming():
    # shaping a pinch (thumb travelling, index steady) must NOT lock
    lock = _new_lock()
    events = []
    for i in range(45):
        t = 0.5 + (0.01 if i % 2 == 0 else -0.01)
        events.append(lock.update(_lock_landmarks(0.5, 0.5, tx=t, ty=0.5)))
    assert "lock" not in events
    assert not lock.is_locked()


def test_click_lock_single_spike_does_not_break_lock():
    lock = _new_lock()
    for _ in range(45):
        lock.update(_lock_landmarks(0.5, 0.5))
    assert lock.is_locked()
    # lone jitter spike, then back on station — lock holds
    assert lock.update(_lock_landmarks(0.5 + 0.02, 0.5)) is None
    assert lock.is_locked()
    assert lock.update(_lock_landmarks(0.5, 0.5)) is None
    assert lock.is_locked()


def test_click_lock_shake_unlock_still_works():
    lock = _new_lock()
    for _ in range(45):
        lock.update(_lock_landmarks(0.5, 0.5))
    assert lock.is_locked()
    # small lateral shake, inside the vote radius — shake path unlocks
    result = None
    for i in range(12):
        x = 0.5 + (0.005 if i % 2 == 0 else -0.005)
        result = lock.update(_lock_landmarks(x, 0.5))
        if result == "unlock":
            break
    assert result == "unlock"
    assert not lock.is_locked()
