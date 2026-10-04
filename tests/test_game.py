"""Game-mode tests — hardware-free (no cv2/mediapipe/pynput).

Covers the PBO overlay inputs:
  - static pose classification (fist / open_palm / peace / index_only)
  - palm anchor stability (knuckle-based, not fingertip)
  - virtual joystick deadzone + cardinal collapse + hold/release hysteresis
  - action debouncer (stable-frames fire-once + cooldown + blip tolerance)
  - key controller diffing (no spam, no stuck keys, disabled-safe)
"""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import landmarks as L  # noqa: E402
from game_keys import GameKeyController  # noqa: E402 (lazy backend: no pynput)
from gestures_game import (  # noqa: E402
    ActionDebouncer,
    FingerCountDriver,
    FingerStateFilter,
    VirtualJoystick,
    classify_static,
    count_extended,
    finger_bar,
    finger_states,
    palm_center,
    pose_to_direction,
    robust_classify_static,
    robust_finger_states,
)


# --- synthetic hands -------------------------------------------------------

def _set_finger(lm, tip, dip, pip, mcp, x, wy, s, kind):
    if kind == "open":
        lm[mcp] = [x, wy - 0.45 * s, 0.0]
        lm[pip] = [x, wy - 0.70 * s, 0.0]
        lm[dip] = [x, wy - 0.90 * s, 0.0]
        lm[tip] = [x, wy - 1.10 * s, 0.0]
    else:  # curled sideways (old wrist-distance check still passes)
        lm[mcp] = [x, wy - 0.45 * s, 0.0]
        lm[pip] = [x + 0.10 * s, wy - 0.62 * s, 0.0]
        lm[dip] = [x + 0.32 * s, wy - 0.58 * s, 0.0]
        lm[tip] = [x + 0.38 * s, wy - 0.38 * s, 0.0]


def make_game_hand(pose="index_only", wx=0.5, wy=0.5, s=0.2):
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    lm[L.WRIST] = [wx, wy, 0.0]
    lm[L.MIDDLE_MCP] = [wx, wy - s, 0.0]
    spec = {
        "fist": ("bent", "bent", "bent", "bent", "tucked"),
        "open_palm": ("open", "open", "open", "open", "open"),
        "peace": ("open", "open", "bent", "bent", "tucked"),
        "index_only": ("open", "bent", "bent", "bent", "tucked"),
        "three": ("open", "open", "open", "bent", "tucked"),
    }[pose]
    ix, mx, rx, px = wx + 0.25 * s, wx + 0.05 * s, wx - 0.15 * s, wx - 0.30 * s
    _set_finger(lm, L.INDEX_TIP, L.INDEX_DIP, L.INDEX_PIP, L.INDEX_MCP,
                ix, wy, s, spec[0])
    _set_finger(lm, L.MIDDLE_TIP, L.MIDDLE_DIP, L.MIDDLE_PIP, L.MIDDLE_MCP,
                mx, wy, s, spec[1])
    _set_finger(lm, L.RING_TIP, L.RING_DIP, L.RING_PIP, L.RING_MCP,
                rx, wy, s, spec[2])
    _set_finger(lm, L.PINKY_TIP, L.PINKY_DIP, L.PINKY_PIP, L.PINKY_MCP,
                px, wy, s, spec[3])
    if spec[4] == "open":  # thumb spread wide
        lm[L.THUMB_TIP] = [wx - 0.55 * s, wy - 0.55 * s, 0.0]
        lm[L.THUMB_IP] = [wx - 0.40 * s, wy - 0.30 * s, 0.0]
    else:  # thumb tucked over curled fingers
        imcp = lm[L.INDEX_MCP]
        lm[L.THUMB_TIP] = [imcp[0] + 0.05 * s, imcp[1] + 0.05 * s, 0.0]
        lm[L.THUMB_IP] = [imcp[0] + 0.15 * s, imcp[1] + 0.15 * s, 0.0]
    lm[L.THUMB_MCP] = [wx - 0.2 * s, wy - 0.1 * s, 0.0]
    lm[L.THUMB_CMC] = [wx - 0.3 * s, wy, 0.0]
    return lm


# --- static classification --------------------------------------------------

def test_classify_static_poses():
    assert classify_static(make_game_hand("fist")) == "fist"
    assert classify_static(make_game_hand("open_palm")) == "open_palm"
    assert classify_static(make_game_hand("peace")) == "peace"
    assert classify_static(make_game_hand("index_only")) == "index_only"
    assert classify_static(make_game_hand("three")) == "three"


def test_finger_states_counts():
    assert count_extended(finger_states(make_game_hand("fist"))) == 0
    assert count_extended(finger_states(make_game_hand("open_palm"))) == 4
    assert count_extended(finger_states(make_game_hand("peace"))) == 2
    assert finger_states(make_game_hand("open_palm"))["thumb"] is True
    assert finger_states(make_game_hand("fist"))["thumb"] is False


def test_palm_center_ignores_fingertips():
    a = make_game_hand("fist", wx=0.5, wy=0.5)
    b = make_game_hand("open_palm", wx=0.5, wy=0.5)
    # same palm base -> same anchor despite wildly different fingertips
    ax, ay = palm_center(a)
    bx, by = palm_center(b)
    assert abs(ax - bx) < 0.03
    assert abs(ay - by) < 0.05
    # ... and it tracks real translation
    c = make_game_hand("fist", wx=0.7, wy=0.5)
    assert palm_center(c)[0] > ax + 0.1


# --- joystick ----------------------------------------------------------------

def test_joystick_deadzone_and_cardinal():
    j = VirtualJoystick(deadzone=0.12, hold_frames=1, release_frames=1,
                        mirror_x=False)
    assert j.update(0.5, 0.5) == "stop"          # center
    assert j.update(0.55, 0.5) == "stop"          # inside deadzone
    assert j.update(0.8, 0.5) == "right"          # clear east
    assert j.update(0.2, 0.5) == "left"
    assert j.update(0.5, 0.2) == "up"             # image y up = game up
    assert j.update(0.5, 0.8) == "down"


def test_joystick_diagonal_collapses_to_dominant_axis():
    j = VirtualJoystick(deadzone=0.05, hold_frames=1, release_frames=1,
                        mirror_x=False)
    assert j.update(0.9, 0.6) == "right"   # |dx|>|dy|
    j.reset()
    assert j.update(0.6, 0.9) == "down"


def test_joystick_mirror_flips_x_only():
    j = VirtualJoystick(deadzone=0.05, hold_frames=1, release_frames=1,
                        mirror_x=True)
    assert j.update(0.8, 0.5) == "left"    # screen-right reads as hand-right
    j.reset()
    jm = VirtualJoystick(deadzone=0.05, hold_frames=1, release_frames=1,
                         mirror_x=False)
    assert jm.update(0.8, 0.5) == "right"


def test_joystick_hold_release_hysteresis():
    j = VirtualJoystick(deadzone=0.05, hold_frames=3, release_frames=3,
                        mirror_x=False)
    assert j.update(0.9, 0.5) == "stop"   # 1/3
    assert j.update(0.9, 0.5) == "stop"   # 2/3
    assert j.update(0.9, 0.5) == "right"  # 3/3 commits
    assert j.update(0.5, 0.5) == "right"  # 1-frame blip does not drop
    assert j.update(0.5, 0.5) == "right"
    assert j.update(0.5, 0.5) == "stop"   # 3/3 stop commits


# --- debouncer ---------------------------------------------------------------

def test_action_debouncer_fires_once_per_hold():
    d = ActionDebouncer(stable_frames=3, cooldown_ms=10_000)
    assert d.update("fist", 0.0) is None
    assert d.update("fist", 0.1) is None
    assert d.update("fist", 0.2) == "fist"   # 3rd stable frame fires
    assert d.update("fist", 0.3) is None     # same hold: must release first
    assert d.update("other", 0.4) is None    # release breaks streak
    assert d.update("fist", 20.0) is None
    assert d.update("fist", 20.1) is None
    assert d.update("fist", 20.2) == "fist"  # cooldown elapsed -> fires again


def test_action_debouncer_tolerates_tracking_blip():
    d = ActionDebouncer(stable_frames=3, cooldown_ms=0, tolerant=("unknown",))
    assert d.update("fist", 0.0) is None
    assert d.update("unknown", 0.1) is None  # blip: streak held, not reset
    assert d.update("fist", 0.2) is None
    assert d.update("fist", 0.3) == "fist"


# --- key controller ------------------------------------------------------------

class _FakeBackend:
    def __init__(self):
        self.log = []

    def press(self, k):
        self.log.append(("press", k))

    def release(self, k):
        self.log.append(("release", k))


def test_key_controller_diffs_no_spam_no_stuck():
    fb = _FakeBackend()
    kc = GameKeyController(fb.press, fb.release, move_keys="arrows")
    kc.set_direction("up")
    assert ("press", "up") in fb.log
    n = len(fb.log)
    kc.set_direction("up")  # same -> silent
    assert len(fb.log) == n
    kc.set_direction("left")  # switch -> release up, press left
    assert ("release", "up") in fb.log
    assert ("press", "left") in fb.log
    kc.set_direction("stop")
    assert ("release", "left") in fb.log
    assert kc.held == set()
    kc.set_direction("up")
    kc.release_all()  # failsafe clears everything
    assert kc.held == set()
    assert ("release", "up") in fb.log


def test_key_controller_both_holds_pair_and_respects_enabled():
    fb = _FakeBackend()
    kc = GameKeyController(fb.press, fb.release, move_keys="both")
    kc.set_direction("up")
    assert "up" in kc.held and "w" in kc.held  # arrow + WASD pair
    kc.enabled = False
    kc.set_direction("stop")  # internal state clears even while muted
    assert kc.held == set()
    fb2 = _FakeBackend()
    kc2 = GameKeyController(fb2.press, fb2.release)
    kc2.enabled = False
    kc2.tap("space")
    assert fb2.log == []  # muted tap sends nothing


# --- robust finger-count recognition ----------------------------------------

def test_robust_classifier_matches_all_finger_counts():
    assert robust_classify_static(make_game_hand("fist")) == "fist"
    assert robust_classify_static(make_game_hand("index_only")) == "index_only"
    assert robust_classify_static(make_game_hand("peace")) == "peace"
    assert robust_classify_static(make_game_hand("three")) == "three"
    assert robust_classify_static(make_game_hand("open_palm")) == "open_palm"


def test_robust_states_agree_with_legacy_on_clean_hands():
    for pose in ("fist", "index_only", "peace", "three", "open_palm"):
        hand = make_game_hand(pose)
        assert robust_finger_states(hand) == finger_states(hand)


def test_pose_to_direction_finger_map():
    assert pose_to_direction("fist") == "stop"
    assert pose_to_direction("index_only") == "up"      # 1 = forward
    assert pose_to_direction("peace") == "down"         # 2 = backward
    assert pose_to_direction("three") == "left"        # 3 = left
    assert pose_to_direction("open_palm") == "right"   # 4 = right
    assert pose_to_direction("unknown") is None        # hold, don't stutter
    assert finger_bar(
        {"index": True, "middle": True, "ring": False, "pinky": False}
    ) == "12.."


def test_finger_filter_kills_single_frame_flicker():
    f = FingerStateFilter(flip_frames=3)
    one = {"index": True, "middle": False, "ring": False, "pinky": False}
    two = {"index": True, "middle": True, "ring": False, "pinky": False}
    assert f.update(one) == one
    # single + double blips never commit
    assert f.update(two) == one
    assert f.update(one) == one
    assert f.update(two) == one
    assert f.update(one) == one
    # ... but a real change held 3 in a row commits on the 3rd
    assert f.update(two) == one   # 1/3
    assert f.update(two) == one   # 2/3
    assert f.update(two) == two   # 3/3 commits


def test_driver_needs_hold_for_go_fast_for_stop():
    d = FingerCountDriver(hold_frames=4, stop_frames=2, max_unknown_hold=10)
    for _ in range(3):  # 1 finger, only 3/4 -> still stopped
        assert d.update("index_only") == "stop"
    assert d.update("index_only") == "up"  # 4/4 commits
    assert d.update("index_only") == "up"
    assert d.update("fist") == "up"    # fist 1/2: safety waits one more
    assert d.update("fist") == "stop"  # 2/2 stops fast


def test_driver_holds_through_unknown_blips_then_gives_up():
    d = FingerCountDriver(hold_frames=2, stop_frames=2, max_unknown_hold=3)
    d.update("index_only")
    assert d.update("index_only") == "up"
    assert d.update("unknown") == "up"   # blip 1: hold
    assert d.update("unknown") == "up"   # blip 2: hold
    assert d.update("unknown") == "stop"  # blip 3: hand really gone


def test_end_to_end_noisy_two_to_three_transition():
    """2 held, ring flickers open/closed, driver must not strobe down/left."""
    f = FingerStateFilter(flip_frames=2)
    d = FingerCountDriver(hold_frames=3, stop_frames=2, max_unknown_hold=10)
    two = {"index": True, "middle": True, "ring": False, "pinky": False}
    three = {"index": True, "middle": True, "ring": True, "pinky": False}
    # settle on 2 = down
    for _ in range(4):
        d.update("peace" if f.update(two) == two else "unknown")
    assert d.active == "down"
    # ring noisy for two frames: filter holds 2, driver holds down
    assert d.update("peace" if f.update(three) == two else "unknown") == "down"
    assert d.update("peace" if f.update(two) == two else "unknown") == "down"
    # genuine 3 held: filter flips on 2nd, driver takes 3 to switch
    seen_left = False
    for _ in range(6):
        stable = f.update(three)
        pose = "three" if stable == three else "unknown"
        if d.update(pose) == "left":
            seen_left = True
    assert seen_left
