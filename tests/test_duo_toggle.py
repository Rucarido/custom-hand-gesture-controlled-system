"""Two-hand duo toggle tests — hardware-free (no cv2/mediapipe/pynput).

Covers the 1+1 gesture that flips movement <-> battle-cursor:
  - is_duo_one_one: lenient (any two "1" hands) vs strict (Left+Right)
  - pick_cursor_hand: preferred hand wins, fallback, none
  - DuoToggle: hold-to-fire, latch-until-release, cooldown, gap tolerance
"""

import gestures_game as GG
from gestures_game import DuoToggle, is_duo_one_one, pick_cursor_hand


def _h(pose, handedness="Unknown", index=True):
    return {"pose": pose, "handedness": handedness, "index_extended": index}


# --- duo detection ------------------------------------------------------------


def test_duo_lenient_any_two_ones():
    assert is_duo_one_one([_h("index_only"), _h("index_only")]) is True
    assert is_duo_one_one(
        [_h("index_only", "Left"), _h("index_only", "Left")]
    ) is True
    assert is_duo_one_one([_h("index_only")]) is False
    assert is_duo_one_one([]) is False


def test_duo_lenient_rejects_non_one():
    assert is_duo_one_one([_h("index_only"), _h("peace")]) is False
    assert is_duo_one_one([_h("index_only"), _h("fist")]) is False
    assert is_duo_one_one([_h("fist"), _h("fist")]) is False
    assert is_duo_one_one([_h("index_only"), _h("unknown")]) is False


def test_duo_strict_needs_left_and_right():
    both = [_h("index_only", "Left"), _h("index_only", "Right")]
    assert is_duo_one_one(both, strict_handedness=True) is True
    assert is_duo_one_one(
        [_h("index_only", "Left"), _h("index_only", "Left")],
        strict_handedness=True,
    ) is False
    assert is_duo_one_one(
        [_h("index_only", "Left"), _h("peace", "Right")],
        strict_handedness=True,
    ) is False


def test_duo_three_hands_still_counts():
    hands = [_h("index_only", "Left"), _h("fist"), _h("index_only", "Right")]
    assert is_duo_one_one(hands) is True


# --- cursor hand picking --------------------------------------------------------


def test_cursor_prefers_right():
    hands = [_h("index_only", "Left"), _h("index_only", "Right")]
    assert pick_cursor_hand(hands, prefer="Right") == 1
    assert pick_cursor_hand(hands, prefer="Left") == 0


def test_cursor_falls_back_to_any_pointing_hand():
    hands = [_h("fist", "Right", index=False), _h("peace", "Left")]
    assert pick_cursor_hand(hands, prefer="Right") == 1


def test_cursor_none_when_no_index():
    hands = [_h("fist", "Left", index=False), _h("fist", "Right", index=False)]
    assert pick_cursor_hand(hands) == -1
    assert pick_cursor_hand([]) == -1


# --- DuoToggle timing ------------------------------------------------------------


def test_toggle_fires_after_hold_then_latches():
    d = DuoToggle(hold_frames=4, cooldown_s=10.0)
    assert d.update(True, 0.0) is False
    assert d.update(True, 0.1) is False
    assert d.update(True, 0.2) is False
    assert d.update(True, 0.3) is True  # toggle frame
    assert d.update(True, 0.4) is False  # latched: same hold can't re-fire
    assert d.update(True, 5.0) is False  # still latched even after cooldown
    assert d.update(False, 5.1) is False  # release -> re-armed
    assert d.latched is False


def test_toggle_cooldown_blocks_fresh_hold():
    d = DuoToggle(hold_frames=2, cooldown_s=1.5)
    assert d.update(True, 0.0) is False
    assert d.update(True, 0.1) is True
    assert d.update(False, 0.2) is False  # re-arm...
    assert d.update(True, 0.3) is False  # ...but cooldown blocks counting
    assert d.update(True, 0.4) is False
    assert d.update(True, 2.0) is False  # cooldown over, 1/2
    assert d.update(True, 2.1) is True  # 2/2 fires again


def test_toggle_tolerates_single_miss():
    d = DuoToggle(hold_frames=5, miss_decay=2, cooldown_s=0.0)
    for i in range(4):
        assert d.update(True, i * 0.05) is False  # 4/5
    assert d.update(False, 0.2) is False  # blip: 4 -> 2, not reset
    assert d.update(True, 0.25) is False  # 3/5
    assert d.update(True, 0.30) is False  # 4/5
    assert d.update(True, 0.35) is True  # 5/5 fires


def test_toggle_long_absence_decays_to_zero():
    d = DuoToggle(hold_frames=4, miss_decay=2, cooldown_s=0.0)
    assert d.update(True, 0.0) is False  # 1/4
    assert d.update(True, 0.1) is False  # 2/4
    assert d.update(False, 0.2) is False  # -> 0
    assert d.update(False, 0.3) is False  # stays 0
    assert d.progress == 0.0
    assert d.update(True, 0.4) is False  # must rebuild the full hold
    assert d.update(True, 0.5) is False
    assert d.update(True, 0.6) is False
    assert d.update(True, 0.7) is True


def test_toggle_progress_meter():
    d = DuoToggle(hold_frames=4, cooldown_s=0.0)
    assert d.progress == 0.0
    d.update(True, 0.0)
    d.update(True, 0.1)
    assert d.progress == 0.5
    d.reset()
    assert d.progress == 0.0
    assert GG.DUO_POSE == "index_only"
