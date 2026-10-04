"""Game gestures — finger-count driving + virtual joystick for tile games.

Designed for top-down browser RPGs like Pokemon Blaze Online
(https://play.pokemonblazeonline.com): arrow-key / WASD walking,
Space/Z/Enter to confirm, X/Esc to cancel.

Hardware-free by design (no cv2/mediapipe/pynput imports) so tests can
import this module directly.

GESTURE MAP (movement.mode = "fingers", the default):
  fist = 0 fingers clenched      -> STOP (release all direction keys)
  1 finger (index only)          -> walk FORWARD / UP
  2 fingers (index + middle)     -> walk BACKWARD / DOWN
  3 fingers (index+middle+ring)   -> walk LEFT
  4 fingers (all four, no thumb)  -> walk RIGHT
  pinch (thumb-tip to index-tip)  -> confirm / interact (Space, separate
                                     PinchClickDetector channel)

Legacy mode (movement.mode = "joystick") keeps the old scheme:
  palm position vs center deadzone -> walk; fist/palm taps -> X/Enter.

WHY THIS RECOGNITION IS MORE ROBUST than single-frame classify_static:
  1. Four cues per finger vote (PIP angle + DIP angle + radial order +
     segment straightness) instead of two — one bad cue (foreshortening,
     side view, jitter) no longer flips the finger.
  2. Per-finger hysteresis + temporal filter (FingerStateFilter): a finger
     must read the new state N frames in a row before the stable state
     flips, killing single-frame flicker at the open/curled boundary.
  3. Pose -> direction driver (FingerCountDriver) with asymmetric timing:
     STOP engages fast (safety), new directions need a longer hold, and
     UNKNOWN tracking blips hold the current direction instead of
     stutter-stepping the character.
"""

import landmarks as L


# --- finger state ----------------------------------------------------------

def is_thumb_extended(landmarks, ratio_thresh: float = 0.55) -> bool:
    """Heuristic thumb check: tip far from the index-MCP knuckle = open.

    Ratio = dist(thumb_tip, index_mcp) / hand_size. Open palm ~0.8-1.2,
    thumb-tucked fist ~0.2-0.4. Threshold 0.55 splits them with margin.
    Hardware-free and unit-testable. Thumb is IGNORED for finger-count
    movement (only index..pinky count) so thumb position never affects
    walking — it only matters for the legacy open_palm menu tap.
    """
    d = L.dist(landmarks[L.THUMB_TIP], landmarks[L.INDEX_MCP])
    return bool((d / L.hand_size(landmarks)) > ratio_thresh)


def finger_states(landmarks, thumb_thresh: float = 0.55) -> dict:
    """Single-frame extension flags (legacy API, kept for pointer gate).

    Uses landmarks.finger_extended (PIP angle + radial). Prefer
    robust_finger_states() + FingerStateFilter for game movement.
    """
    return {
        "thumb": is_thumb_extended(landmarks, thumb_thresh),
        "index": L.finger_extended(
            landmarks, L.INDEX_TIP, L.INDEX_DIP, L.INDEX_PIP, L.INDEX_MCP
        ),
        "middle": L.finger_extended(
            landmarks, L.MIDDLE_TIP, L.MIDDLE_DIP, L.MIDDLE_PIP, L.MIDDLE_MCP
        ),
        "ring": L.finger_extended(
            landmarks, L.RING_TIP, L.RING_DIP, L.RING_PIP, L.RING_MCP
        ),
        "pinky": L.finger_extended(
            landmarks, L.PINKY_TIP, L.PINKY_DIP, L.PINKY_PIP, L.PINKY_MCP
        ),
    }


def count_extended(states: dict, include_thumb: bool = False) -> int:
    """Number of extended fingers (optionally ignoring the thumb)."""
    keys = ("index", "middle", "ring", "pinky")
    if include_thumb:
        keys = ("thumb",) + keys
    return sum(1 for k in keys if states.get(k))


# --- robust per-finger cues (the "recognition better" layer) ----------------

FINGER_JOINTS = {
    "index": (L.INDEX_TIP, L.INDEX_DIP, L.INDEX_PIP, L.INDEX_MCP),
    "middle": (L.MIDDLE_TIP, L.MIDDLE_DIP, L.MIDDLE_PIP, L.MIDDLE_MCP),
    "ring": (L.RING_TIP, L.RING_DIP, L.RING_PIP, L.RING_MCP),
    "pinky": (L.PINKY_TIP, L.PINKY_DIP, L.PINKY_PIP, L.PINKY_MCP),
}
FINGER_ORDER = ("index", "middle", "ring", "pinky")


def finger_cues(landmarks, finger: str) -> dict:
    """Four scale-invariant straightness cues for one finger.

    Returns {pip_angle, dip_angle, radial_margin, straightness, votes}.
    All pure geometry on normalized landmarks:
      pip_angle  — interior angle at PIP (MCP-PIP-DIP), ~180 open/~90 shut
      dip_angle  — interior angle at DIP (PIP-DIP-TIP), ~180 open
      radial_margin — (tip_wrist_r - pip_wrist_r) / hand_size; >0 extended
      straightness — dist(MCP,TIP) / polyline(MCP..TIP); ~1.0 straight
    """
    tip_i, dip_i, pip_i, mcp_i = FINGER_JOINTS[finger]
    tip, dip, pip, mcp = (
        landmarks[tip_i], landmarks[dip_i], landmarks[pip_i], landmarks[mcp_i],
    )
    wrist = landmarks[L.WRIST]
    hs = L.hand_size(landmarks)
    pip_angle = L.joint_angle(mcp, pip, dip)
    dip_angle = L.joint_angle(pip, dip, tip)
    tip_r = L.dist(tip, wrist)
    pip_r = L.dist(pip, wrist)
    radial_margin = (tip_r - pip_r) / hs
    seg = L.dist(mcp, pip) + L.dist(pip, dip) + L.dist(dip, tip)
    straightness = (L.dist(mcp, tip) / seg) if seg > 1e-9 else 1.0
    return {
        "pip_angle": pip_angle,
        "dip_angle": dip_angle,
        "radial_margin": radial_margin,
        "straightness": straightness,
    }


def finger_extended_robust(
    landmarks,
    finger: str,
    pip_thresh: float = 140.0,
    dip_thresh: float = 130.0,
    straight_thresh: float = 0.80,
) -> bool:
    """Vote 4 cues, need >=3 to call extended (one bad cue tolerated).

    Per-finger loosening: ring/pinky foreshorten more often, so callers
    (robust_finger_states) pass them ~10 deg / 0.05 looser thresholds.
    """
    c = finger_cues(landmarks, finger)
    votes = sum(
        [
            c["pip_angle"] >= pip_thresh,
            c["dip_angle"] >= dip_thresh,
            c["radial_margin"] > 0.0,
            c["straightness"] >= straight_thresh,
        ]
    )
    return bool(votes >= 3)


def robust_finger_states(
    landmarks,
    pip_thresh: float = 140.0,
    dip_thresh: float = 130.0,
    straight_thresh: float = 0.80,
    thumb_thresh: float = 0.55,
) -> dict:
    """Single-frame extension flags with 4-cue voting per finger.

    Ring/pinky get looser thresholds (they curl tighter on camera and
    would otherwise flicker when the user holds 3/4 up).
    """
    out = {"thumb": is_thumb_extended(landmarks, thumb_thresh)}
    for f in FINGER_ORDER:
        loose = 10.0 if f in ("ring", "pinky") else 0.0
        out[f] = finger_extended_robust(
            landmarks,
            f,
            pip_thresh=pip_thresh - loose,
            dip_thresh=dip_thresh - loose,
            straight_thresh=straight_thresh - (0.05 if f == "pinky" else 0.0),
        )
    return out


# --- static pose classification --------------------------------------------

# Canonical names emitted by classify_static().
FIST = "fist"             # 0 fingers -> STOP
OPEN_PALM = "open_palm"   # 4 fingers (+thumb ignored) -> RIGHT (fingers mode)
PEACE = "peace"           # index+middle only (2) -> DOWN (fingers mode)
INDEX_ONLY = "index_only"  # 1 finger -> UP (fingers mode)
THREE = "three"           # index+middle+ring (3) -> LEFT (fingers mode)
UNKNOWN = "unknown"


def _classify_from_states(s: dict) -> str:
    idx, mid, ring, pinky = s["index"], s["middle"], s["ring"], s["pinky"]
    if not idx and not mid and not ring and not pinky:
        return FIST
    if idx and mid and ring and pinky:
        return OPEN_PALM
    if idx and mid and not ring and not pinky:
        return PEACE
    if idx and not mid and not ring and not pinky:
        return INDEX_ONLY
    if idx and mid and ring and not pinky:
        return THREE
    return UNKNOWN


def classify_static(landmarks, thumb_thresh: float = 0.55) -> str:
    """Legacy single-frame classifier (2-cue fingers, no filtering).

    Pure function of the 21 landmarks. Kept for the pointer gate and the
    joystick-mode HUD. Game movement should use robust_classify_static +
    FingerStateFilter + FingerCountDriver instead.
    """
    return _classify_from_states(finger_states(landmarks, thumb_thresh))


def robust_classify_static(
    landmarks,
    pip_thresh: float = 140.0,
    dip_thresh: float = 130.0,
    straight_thresh: float = 0.80,
    thumb_thresh: float = 0.55,
) -> str:
    """Single-frame classifier on the 4-cue robust finger states."""
    return _classify_from_states(
        robust_finger_states(
            landmarks,
            pip_thresh=pip_thresh,
            dip_thresh=dip_thresh,
            straight_thresh=straight_thresh,
            thumb_thresh=thumb_thresh,
        )
    )


# --- temporal filter: per-finger flip hysteresis ----------------------------

class FingerStateFilter:
    """Debounce each finger: flip stable state only after N repeat frames.

    Single-frame landmark jitter (a ring angle dipping 139.9 -> 140.1)
    used to strobe the count 2<->3<->2 and moonwalk the character. The
    filter holds the last stable 4-bit pattern until a new pattern
    repeats `flip_frames` in a row, then commits it at once so the count
    never passes through a phantom intermediate value one frame at a
    time.
    """

    def __init__(self, flip_frames: int = 3) -> None:
        self.flip_frames = max(1, flip_frames)
        self.stable: dict | None = None
        self._candidate: dict | None = None
        self._count = 0

    def update(self, raw: dict) -> dict:
        raw4 = {k: bool(raw.get(k, False)) for k in FINGER_ORDER}
        if self.stable is None:
            self.stable = dict(raw4)
            self._candidate = dict(raw4)
            self._count = 0
            return dict(self.stable)
        if raw4 == self.stable:
            self._candidate = dict(raw4)
            self._count = 0
            return dict(self.stable)
        if raw4 == self._candidate:
            self._count += 1
        else:
            self._candidate = dict(raw4)
            self._count = 1
        if self._count >= self.flip_frames:
            self.stable = dict(self._candidate)
            self._count = 0
        return dict(self.stable)

    def reset(self) -> None:
        self.stable = None
        self._candidate = None
        self._count = 0


# --- pose -> direction driver (finger-count walking) -------------------------

# Direction names. Cardinal-only: grid RPGs (PBO included) move on tiles.
UP, DOWN, LEFT, RIGHT, STOP = "up", "down", "left", "right", "stop"

# Default finger-count map requested for PBO:
#   fist(0)=stop, 1=forward/up, 2=back/down, 3=left, 4=right.
POSE_TO_DIRECTION = {
    FIST: STOP,
    INDEX_ONLY: UP,
    PEACE: DOWN,
    THREE: LEFT,
    OPEN_PALM: RIGHT,
}


def pose_to_direction(pose: str, mapping: dict | None = None) -> str | None:
    """Map a pose name to a direction; UNKNOWN -> None (hold current)."""
    table = mapping or POSE_TO_DIRECTION
    return table.get(pose)  # unknown poses -> None = hold, never stop-stutter


def finger_bar(states: dict) -> str:
    """Compact HUD readout, e.g. '12..' = index+middle up. Thumb excluded."""
    return "".join(
        {"index": "1", "middle": "2", "ring": "3", "pinky": "4"}[f]
        if states.get(f)
        else "."
        for f in FINGER_ORDER
    )


class FingerCountDriver:
    """Stabilized pose -> held direction with asymmetric timing.

    - A fresh non-stop direction must repeat `hold_frames` before the key
      presses (leans against accidental drive-bys while shaping fingers).
    - STOP (fist) engages after only `stop_frames` (default faster than
      hold): lifting into a fist must halt the character promptly so you
      don't walk into a trainer battle / off a ledge.
    - UNKNOWN (or unmapped) poses HOLD the current direction for up to
      `max_unknown_hold` frames — a tracking blip mid-stride doesn't
      release the key and stutter-step — then decay to STOP if the hand
      never resolves (hand actually left / turned away).
    """

    def __init__(
        self,
        mapping: dict | None = None,
        hold_frames: int = 4,
        stop_frames: int = 2,
        max_unknown_hold: int = 10,
    ) -> None:
        self.mapping = dict(mapping or POSE_TO_DIRECTION)
        self.hold_frames = max(1, hold_frames)
        self.stop_frames = max(1, stop_frames)
        self.max_unknown_hold = max(1, max_unknown_hold)
        self.active = STOP
        self.pose = FIST
        self._candidate: str | None = None
        self._count = 0
        self._unknown_streak = 0

    def update(self, pose: str) -> str:
        """Feed one (already finger-filtered) pose; return held direction."""
        self.pose = pose
        want = self.mapping.get(pose)  # None for UNKNOWN/unmapped
        if want is None:
            self._unknown_streak += 1
            if self._unknown_streak >= self.max_unknown_hold:
                self.active = STOP
                self._candidate = None
                self._count = 0
            return self.active
        self._unknown_streak = 0

        if want == self.active:
            self._candidate = None
            self._count = 0
            return self.active

        if want != self._candidate:
            self._candidate = want
            self._count = 1
        else:
            self._count += 1

        need = self.stop_frames if want == STOP else self.hold_frames
        if self._count >= need:
            self.active = want
            self._candidate = None
            self._count = 0
        return self.active

    def reset(self) -> None:
        self.active = STOP
        self.pose = FIST
        self._candidate = None
        self._count = 0
        self._unknown_streak = 0


# --- two-hand duo toggle (1+1) + battle cursor picking -------------------------
#
# Toggle gesture: LEFT hand shows "1" (index_only) AND RIGHT hand shows "1"
# at the same time, held briefly. game_mode flips forced OVERWORLD <->
# forced BATTLE on the toggle frame; battle-cursor mode then drives the OS
# cursor from the cursor hand's index tip, pinch clicks to select, and the
# same 1+1 hold flips back to movement mode.
#
# All helpers here are pure (no cv2/mediapipe/pynput) and unit-tested.
# Per-frame hand entries are plain dicts:
#   {"pose": str, "handedness": str, "index_extended": bool}

# The per-hand shape that forms the duo toggle.
DUO_POSE = INDEX_ONLY


def is_duo_one_one(hands_info, strict_handedness: bool = False) -> bool:
    """True when two distinct hands both show "1" (index_only).

    strict_handedness=True additionally requires one "Left" and one
    "Right" label. Default lenient (any two 1-hands) is more robust:
    MediaPipe handedness flips when hands cross or the camera mirrors,
    and a missed toggle is worse than a loose one here because the
    toggle also needs a sustained hold + latch-until-release + cooldown.
    """
    ones = [h for h in hands_info if h.get("pose") == DUO_POSE]
    if len(ones) < 2:
        return False
    if not strict_handedness:
        return True
    labels = {str(h.get("handedness", "Unknown")) for h in ones}
    return "Left" in labels and "Right" in labels


def pick_cursor_hand(hands_info, prefer: str = "Right") -> int:
    """Index into hands_info driving the battle cursor, or -1.

    Preference order: preferred-handed hand with index extended, then any
    hand with index extended. Hands making a fist (or lost tracking) never
    drive the cursor, so shaping taps elsewhere can't yank the pointer.
    """
    best = -1
    for i, h in enumerate(hands_info):
        if h.get("index_extended"):
            if str(h.get("handedness", "")) == prefer:
                return i
            if best == -1:
                best = i
    return best


class DuoToggle:
    """Hold-1+1-on-both-hands mode toggle with latch + cooldown.

    - `hold_frames`: duo must accumulate this many frames to fire. Misses
      only decay the count by `miss_decay` (gap tolerance) instead of
      resetting, so one or two tracking blips mid-hold don't eat the
      toggle — but waving 1+1 around all day never accumulates either.
    - Latch-until-release: after firing, further duo frames do nothing
      until duo reads False once (re-arm). The same hold that toggled
      can never toggle straight back.
    - `cooldown_s`: after firing, even a fresh re-armed hold is ignored
      until the cooldown elapses (stops double-toggle bounce).
    - update(duo: bool, t: float) -> True exactly on the toggle frame.
    """

    def __init__(
        self,
        hold_frames: int = 8,
        miss_decay: int = 2,
        cooldown_s: float = 1.5,
    ) -> None:
        self.hold_frames = max(1, hold_frames)
        self.miss_decay = max(1, miss_decay)
        self.cooldown_s = max(0.0, cooldown_s)
        self._count = 0
        self._latched = False
        self._last_fire_t = -1e9

    @property
    def progress(self) -> float:
        """0..1 hold progress for the overlay meter."""
        return min(1.0, self._count / self.hold_frames)

    @property
    def latched(self) -> bool:
        return self._latched

    def update(self, duo: bool, t: float) -> bool:
        if self._latched:
            if not duo:
                self._latched = False  # hand(s) lowered -> re-armed
                self._count = 0
            return False
        if (t - self._last_fire_t) < self.cooldown_s:
            if not duo:
                self._count = 0
            return False
        if duo:
            self._count += 1
            if self._count >= self.hold_frames:
                self._last_fire_t = t
                self._latched = True
                self._count = 0
                return True
            return False
        self._count = max(0, self._count - self.miss_decay)
        return False

    def reset(self) -> None:
        self._count = 0
        self._latched = False


# --- palm anchor (joystick input point, legacy mode) --------------------------

def palm_center(landmarks):
    """Stable palm anchor: mean of wrist + 4 non-thumb MCP joints.

    Fingertips jitter far more than knuckles; averaging the palm base
    gives a joystick input that does not wobble when pinching.
    Returns (x, y) in normalized 0-1 image coords.
    """
    pts = [
        landmarks[L.WRIST],
        landmarks[L.INDEX_MCP],
        landmarks[L.MIDDLE_MCP],
        landmarks[L.RING_MCP],
        landmarks[L.PINKY_MCP],
    ]
    n = len(pts)
    return (
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
    )


# --- virtual joystick (legacy movement.mode = "joystick") ---------------------

class VirtualJoystick:
    """Palm-position -> cardinal direction with deadzone + hysteresis.

    Input is the normalized palm_center (x, y). Output per update() is
    one of up/down/left/right/stop.

    - deadzone: radius around center inside which output is STOP.
    - hold_frames: a new non-stop direction must repeat N frames before
      it becomes active (kills flicker on the border).
    - release_frames: STOP must repeat N frames before motion releases
      (so a one-frame tracking wobble does not drop a held key and make
      the character stutter-step).
    - dominant axis wins (larger |dx| vs |dy|); diagonals collapse to the
      stronger axis, which is what tile movement wants.
    - mirror_x: when the camera preview is mirrored, flip x so moving
      the hand right on screen walks right in game.
    """

    def __init__(
        self,
        center_x: float = 0.5,
        center_y: float = 0.5,
        deadzone: float = 0.12,
        hold_frames: int = 2,
        release_frames: int = 3,
        mirror_x: bool = True,
    ) -> None:
        self.cx = center_x
        self.cy = center_y
        self.deadzone = max(0.01, deadzone)
        self.hold_frames = max(1, hold_frames)
        self.release_frames = max(1, release_frames)
        self.mirror_x = mirror_x
        self.active = STOP       # committed (key-holding) direction
        self._candidate = STOP   # raw direction currently counting
        self._count = 0

    def _raw(self, x: float, y: float) -> str:
        dx = x - self.cx
        if self.mirror_x:
            dx = -dx
        # NOTE: image y grows downward, game up is negative dy — but the
        # camera already shows the user mirrored intuitively (hand up on
        # screen = smaller y), so no extra flip: dy<0 means UP on screen.
        dy = y - self.cy
        if (dx * dx + dy * dy) ** 0.5 < self.deadzone:
            return STOP
        if abs(dx) > abs(dy):
            return RIGHT if dx > 0 else LEFT
        return DOWN if dy > 0 else UP

    def update(self, x: float, y: float) -> str:
        """Feed one palm position; return the committed direction."""
        raw = self._raw(x, y)
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1

        if self.active == STOP:
            # stopped -> need hold_frames to start moving
            if raw != STOP and self._count >= self.hold_frames:
                self.active = raw
        else:
            if raw == self.active:
                pass  # keep holding; candidate counter irrelevant
            elif raw == STOP:
                if self._count >= self.release_frames:
                    self.active = STOP
            else:
                # direction change (e.g. up -> left): switch fast but not
                # on a single noisy frame.
                if self._count >= self.hold_frames:
                    self.active = raw
        return self.active

    def reset(self) -> None:
        self.active = STOP
        self._candidate = STOP
        self._count = 0


# --- action debouncer (legacy joystick-mode taps) ------------------------------

class ActionDebouncer:
    """Require a pose to hold N frames, then fire once per cooldown.

    Used ONLY by legacy joystick mode (fist-tap = X, palm-tap = Enter).
    In fingers movement mode poses drive continuous movement instead, so
    no taps fire (a fist that both stops AND taps X would be infuriating).

    update(pose) -> pose name on the firing frame, else None.
    Any other pose (including UNKNOWN/STOP gaps) breaks the streak unless
    it is listed in `tolerant` (e.g. tolerate one UNKNOWN tracking blip).
    """

    def __init__(
        self,
        stable_frames: int = 12,
        cooldown_ms: float = 800.0,
        tolerant: tuple = ("unknown",),
    ) -> None:
        self.stable_frames = max(1, stable_frames)
        self.cooldown_s = max(0.0, cooldown_ms) / 1000.0
        self.tolerant = set(tolerant or ())
        self._streak_pose = None
        self._streak = 0
        self._last_fire_t = -1e9

    def update(self, pose: str | None, t: float) -> str | None:
        if pose is None:
            pose = UNKNOWN
        if pose in self.tolerant:
            # blip: hold the streak, don't advance or reset it
            return None
        if pose == self._streak_pose:
            self._streak += 1
        else:
            self._streak_pose = pose
            self._streak = 1
        if self._streak >= self.stable_frames:
            if (t - self._last_fire_t) >= self.cooldown_s:
                self._last_fire_t = t
                # require release before re-fire of the same hold:
                self._streak_pose = None
                self._streak = 0
                return pose
        return None

    def reset(self) -> None:
        self._streak_pose = None
        self._streak = 0
