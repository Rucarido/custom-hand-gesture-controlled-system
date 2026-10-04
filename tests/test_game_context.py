"""Battle-context tests — hardware-free (no cv2/mss/pyautogui).

game_context.py keeps all screen capture imports lazy inside
score_battle_screen(), so importing it here never touches hardware.
These tests cover only the pure core: pose->battle-key mapping,
the asymmetric context smoother, and the fire-once battle tap gate.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from game_context import (  # noqa: E402
    BATTLE,
    OVERWORLD,
    BattleTapGate,
    ContextSmoother,
    battle_key_for_pose,
)


def test_battle_key_for_pose_all_counts():
    assert battle_key_for_pose("index_only") == "1"  # 1 finger -> move 1
    assert battle_key_for_pose("peace") == "2"        # 2 -> move 2
    assert battle_key_for_pose("three") == "3"        # 3 -> move 3
    assert battle_key_for_pose("open_palm") == "4"    # 4 -> move 4
    assert battle_key_for_pose("fist") == "x"         # fist = cancel
    assert battle_key_for_pose("unknown") is None     # noise never fires
    assert battle_key_for_pose("no-hand") is None


def test_battle_key_for_pose_custom_keys():
    moves = ("a", "b", "c", "d")
    assert battle_key_for_pose("index_only", move_keys=moves) == "a"
    assert battle_key_for_pose("open_palm", move_keys=moves) == "d"
    assert battle_key_for_pose("fist", cancel_key="esc") == "esc"


def test_smoother_needs_streak_to_enter_battle():
    s = ContextSmoother(enter_frames=3, exit_frames=5)
    assert s.update(False) == OVERWORLD
    assert s.update(True) == OVERWORLD    # 1/3
    assert s.update(True) == OVERWORLD    # 2/3
    assert s.update(True) == BATTLE       # 3/3 commits
    # single overworld flicker mid-battle must NOT drop out
    assert s.update(False) == BATTLE
    assert s.update(False) == BATTLE


def test_smoother_needs_longer_streak_to_exit():
    s = ContextSmoother(enter_frames=2, exit_frames=5)
    s.update(True)
    assert s.update(True) == BATTLE
    for _ in range(4):
        assert s.update(False) == BATTLE  # 1..4/5 hold
    assert s.update(False) == OVERWORLD   # 5/5 exits


def test_smoother_force_and_reset():
    s = ContextSmoother()
    assert s.force(BATTLE) == BATTLE
    assert s.update(False) == BATTLE  # 1/5 exit streak, still battle
    s.reset()
    assert s.context == OVERWORLD


def test_battle_gate_fires_once_per_hold():
    g = BattleTapGate(stable_frames=3, cooldown_ms=10_000)
    assert g.update("1", 0.0) is None
    assert g.update("1", 0.1) is None
    assert g.update("1", 0.2) == "1"   # 3rd frame fires
    assert g.update("1", 0.3) is None  # same hold: must release first
    # global cooldown: a different key right after is also blocked —
    # morphing 1 -> fist passes through shapes that must not double-fire.
    assert g.update("x", 0.4) is None
    assert g.update("x", 0.5) is None
    assert g.update("x", 0.6) is None  # stable but cooldown active
    # ... once cooldown elapses the still-held pose fires (hold-to-repeat).
    assert g.update("x", 20.0) == "x"
    assert g.update("x", 20.1) is None  # post-fire: new streak starts


def test_battle_gate_unknown_blip_holds_streak():
    g = BattleTapGate(stable_frames=3, cooldown_ms=0)
    assert g.update("2", 0.0) is None
    assert g.update(None, 0.1) is None  # tracking blip: neither up nor reset
    assert g.update("2", 0.2) is None
    assert g.update("2", 0.3) == "2"


def test_battle_gate_cooldown_blocks_immediate_refire():
    g = BattleTapGate(stable_frames=2, cooldown_ms=5_000)
    assert g.update("3", 0.0) is None
    assert g.update("3", 0.1) == "3"
    # global cooldown: release onto another key quickly still blocked.
    assert g.update("x", 1.0) is None
    assert g.update("x", 1.1) is None  # stable but cooldown active
    assert g.update("3", 1.2) is None
    assert g.update("3", 1.3) is None  # stable but cooldown (from t=0.1) blocks
    assert g.update("3", 6.0) == "3"  # after 5 s: fires again
