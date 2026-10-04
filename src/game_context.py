"""Battle context — overworld vs battle detection + battle move mapping.

PBO spends most time in the OVERWORLD (finger counts = held walk keys)
but pops into a BATTLE screen where the same 1/2/3/4 fingers must mean
"pick move 1/2/3/4" (single taps, not held walking). This module makes
that switch automatic, with a manual override because auto-detection is
a heuristic and PBO's UI can change.

Hardware-free core (no cv2/mss/pyautogui at import): the live screen
scorer imports lazily inside score_battle_screen() so unit tests and
`pytest` never touch the screen. Tests inject fake scores/readings.

Contexts: OVERWORLD = "overworld", BATTLE = "battle".

Pipeline in game_mode (throttled to ~2 Hz to save CPU):
  screenshot -> score_battle_screen() 0..1 -> threshold -> bool reading
      -> ContextSmoother (N consecutive to switch) -> context
      -> manual force (off/auto/battle/overworld via 'b') wins if set
      -> on enter-battle: release walk keys; route poses to battle taps
"""

OVERWORLD = "overworld"
BATTLE = "battle"

# Pose -> battle action. Thumb ignored (same as movement). Fist in battle
# is cancel/back, NOT stop — there is nothing to walk into. Pinch stays
# confirm in both contexts (handled in game_mode, not here).
BATTLE_POSE_TO_MOVE_INDEX = {
    "index_only": 0,   # 1 finger -> move slot 1
    "peace": 1,        # 2 fingers -> move slot 2
    "three": 2,        # 3 fingers -> move slot 3
    "open_palm": 3,    # 4 fingers -> move slot 4
}
BATTLE_CANCEL_POSE = "fist"


def battle_key_for_pose(
    pose: str,
    move_keys: tuple | list = ("1", "2", "3", "4"),
    cancel_key: str = "x",
) -> str | None:
    """Map one stabilized pose to the battle key to tap.

    Returns None for UNKNOWN/unmapped (hold — never fire on noise).
    """
    if pose == BATTLE_CANCEL_POSE:
        return cancel_key
    idx = BATTLE_POSE_TO_MOVE_INDEX.get(pose)
    if idx is None:
        return None
    if 0 <= idx < len(move_keys):
        return move_keys[idx]
    return None


class ContextSmoother:
    """Debounced overworld<->battle switch with asymmetric timing.

    A single battle-score flicker (damage numbers, dialog box pop) must
    not yank the mapping mid-stride, and a mid-battle animation dip must
    not drop back to walking and march you into the wild. So entering
    battle needs `enter_frames` consecutive battle readings, leaving
    needs `exit_frames` (default slower) consecutive overworld readings.
    """

    def __init__(
        self,
        enter_frames: int = 3,
        exit_frames: int = 5,
        start: str = OVERWORLD,
    ) -> None:
        self.enter_frames = max(1, enter_frames)
        self.exit_frames = max(1, exit_frames)
        self.context = start
        self._battle_streak = 0
        self._over_streak = 0

    def update(self, battle_reading: bool) -> str:
        """Feed one thresholded reading; return the committed context."""
        if battle_reading:
            self._battle_streak += 1
            self._over_streak = 0
        else:
            self._over_streak += 1
            self._battle_streak = 0

        if self.context == OVERWORLD:
            if self._battle_streak >= self.enter_frames:
                self.context = BATTLE
                self._battle_streak = 0
                self._over_streak = 0
        else:  # BATTLE
            if self._over_streak >= self.exit_frames:
                self.context = OVERWORLD
                self._battle_streak = 0
                self._over_streak = 0
        return self.context

    def force(self, context: str) -> str:
        """Snap immediately (manual override / startup)."""
        self.context = context
        self._battle_streak = 0
        self._over_streak = 0
        return self.context

    def reset(self) -> None:
        self.context = OVERWORLD
        self._battle_streak = 0
        self._over_streak = 0


class BattleTapGate:
    """Fire-per-stable-hold battle taps with global cooldown.

    Same pose held -> taps once per `stable_frames`, then at most once
    per `cooldown_ms` while still held (hold-to-repeat, handy for picking
    the same move every turn without lowering the hand). Switching poses
    restarts the streak; a global cooldown after ANY tap blocks the next
    tap (even a different key), so morphing 1 -> 2 through a fist flash
    can't double-fire cancel mid-transition. UNKNOWN (None) readings
    neither advance nor break the streak, so a tracking blip mid-hold
    doesn't eat the move.
    """

    def __init__(
        self,
        stable_frames: int = 6,
        cooldown_ms: float = 1200.0,
    ) -> None:
        self.stable_frames = max(1, stable_frames)
        self.cooldown_s = max(0.0, cooldown_ms) / 1000.0
        self._streak_key: str | None = None
        self._streak = 0
        self._last_fire_t = -1e9

    def update(self, key: str | None, t: float) -> str | None:
        if key is None:
            return None  # tracking blip: hold streak, do nothing
        if key == self._streak_key:
            self._streak += 1
        else:
            self._streak_key = key
            self._streak = 1
        if self._streak >= self.stable_frames:
            if (t - self._last_fire_t) >= self.cooldown_s:
                self._last_fire_t = t
                fired = self._streak_key
                self._streak_key = None  # require release before refire
                self._streak = 0
                return fired
        return None

    def reset(self) -> None:
        self._streak_key = None
        self._streak = 0


# --- live screen scorer (lazy imports: never at module top) -------------------

_TEMPLATE_CACHE: list = []


def _load_templates(templates_dir: str = "assets/battle_templates"):
    """Load user-supplied battle UI templates once (best-effort)."""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE
    try:
        from pathlib import Path

        import cv2

        d = Path(templates_dir)
        if d.is_dir():
            for p in sorted(d.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    _TEMPLATE_CACHE.append(img)
    except Exception:
        pass
    return _TEMPLATE_CACHE


def score_battle_screen(
    roi_bottom: float = 0.55,
    templates_dir: str = "assets/battle_templates",
    debug: bool = False,
) -> float:
    """Return 0..1 battle-likelihood from a live screenshot.

    Heuristic (generic PBO-style bottom battle panel, no hard dependency
    on exact art):
      1. Grab the screen, crop the bottom ROI where battle menus live.
      2. Score A — panel structure: strong long horizontal + vertical
         edges (2x2 move grid separators + dialog box borders) via
         Canny + HoughLinesP line-length mass.
      3. Score B — template hit (optional): max normalized cross-
         correlation against user PNGs in assets/battle_templates/.
      Final = max(A, B). Returns 0.0 on any capture failure (fail to
    overworld — walking is the safe default, and the 'b' key overrides).
    """
    try:
        import numpy as np
        import pyautogui

        shot = pyautogui.screenshot()
        frame = np.asarray(shot)
    except Exception as e:
        if debug:
            print(f"[context] screenshot failed (non-fatal): {e}")
        return 0.0

    try:
        import cv2

        # pyautogui gives RGB; convert for edge work.
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        h = gray.shape[0]
        y0 = int(h * min(0.9, max(0.0, roi_bottom)))
        roi = gray[y0:h, :]
        if roi.size == 0:
            return 0.0

        small = cv2.resize(roi, (480, max(48, int(480 * roi.shape[0] / max(1, roi.shape[1])))))
        edges = cv2.Canny(small, 60, 150)
        lines = cv2.HoughLinesP(
            edges, 1, 3.14159 / 180, threshold=60,
            minLineLength=120, maxLineGap=12,
        )
        mass = 0.0
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                horizontal = abs(y2 - y1) < 0.25 * abs(x2 - x1) + 4
                vertical = abs(x2 - x1) < 0.25 * abs(y2 - y1) + 4
                if horizontal or vertical:
                    mass += length
        # ~2 long separators + box borders ≈ 1500+ px at 480 wide.
        score_a = min(1.0, mass / 1500.0)

        score_b = 0.0
        for tmpl in _load_templates(templates_dir):
            try:
                if tmpl.shape[0] > small.shape[0] or tmpl.shape[1] > small.shape[1]:
                    continue
                res = cv2.matchTemplate(small, tmpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, _ = cv2.minMaxLoc(res)
                score_b = max(score_b, float(mx))
            except Exception:
                continue

        score = max(score_a, score_b)
        if debug:
            print(f"[context] battle score={score:.2f} (edge={score_a:.2f} "
                  f"tmpl={score_b:.2f})")
        return float(score)
    except Exception as e:
        if debug:
            print(f"[context] scoring failed (non-fatal): {e}")
        return 0.0


def read_battle_bool(
    threshold: float = 0.5,
    roi_bottom: float = 0.55,
    templates_dir: str = "assets/battle_templates",
    debug: bool = False,
) -> bool:
    """Thresholded convenience wrapper for the live loop."""
    return score_battle_screen(
        roi_bottom=roi_bottom, templates_dir=templates_dir, debug=debug
    ) >= threshold
