"""MediaPipe hand landmark indices and finger connectivity.

21 landmarks per hand, normalized image coordinates [x, y, z] in [0, 1].
z is depth relative to the wrist, roughly on the same scale as x/y
(smaller z = closer to the camera for most poses).
"""

import math

# --- Landmark indices ---
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

NUM_LANDMARKS = 21

# --- Pure geometry helpers (hardware-free) ---------------------------------
# Scale-invariant metrics: dividing by hand size makes pinch thresholds hold
# whether the hand fills the frame or sits a metre back.


def _xy(landmarks, i):
    return landmarks[i][0], landmarks[i][1]


def dist(a, b) -> float:
    """2D Euclidean distance between two [x, y(, z)] points."""
    return math.dist(a[:2], b[:2])


def tip_distance(landmarks) -> float:
    """Raw thumb-tip (4) to index-tip (8) distance (normalized image units)."""
    return dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])


def hand_size(landmarks) -> float:
    """Hand scale: wrist (0) to middle-MCP (9) distance, floored > 0."""
    return max(dist(landmarks[WRIST], landmarks[MIDDLE_MCP]), 1e-6)


def pinch_ratio(landmarks) -> float:
    """Scale-invariant pinch metric: tip distance / hand size.

    Touching fingers ~0.05-0.15, clearly open ~0.5+. Thresholds in
    config/click (pinch_on 0.25 / pinch_off 0.40) are in these units.
    """
    return tip_distance(landmarks) / hand_size(landmarks)


def joint_angle(a, b, c) -> float:
    """Interior angle at point b (degrees) for segments b->a, b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def finger_extended(landmarks, tip, dip, pip, mcp, angle_thresh_deg: float = 140.0) -> bool:
    """Two-cue extension test: knuckle angle AND tip past pip radially.

    The old wrist-distance check (tip farther from wrist than pip) is fooled
    by curled-but-sideways fingers; requiring a straight PIP joint (~180 deg
    when open, ~90 deg when curled) kills those false positives.
    """
    wrist = landmarks[WRIST]
    tip_pt, dip_pt = landmarks[tip], landmarks[dip]
    pip_pt, mcp_pt = landmarks[pip], landmarks[mcp]
    angle_ok = joint_angle(mcp_pt, pip_pt, dip_pt) >= angle_thresh_deg
    tip_r2 = (tip_pt[0] - wrist[0]) ** 2 + (tip_pt[1] - wrist[1]) ** 2
    pip_r2 = (pip_pt[0] - wrist[0]) ** 2 + (pip_pt[1] - wrist[1]) ** 2
    radial_ok = tip_r2 > pip_r2
    return bool(angle_ok and radial_ok)


def is_index_extended(landmarks) -> bool:
    """Index finger extended (curl-robust)."""
    return finger_extended(landmarks, INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP)


def hand_area(landmarks) -> float:
    """Bounding-box area of the hand (normalized units^2) — proximity proxy."""
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-9)


def choose_hand_index(hands, names, prefer: str = "") -> int:
    """Pick which detected hand to track (pure — unit-testable).

    ``hands``: list of 21-landmark lists; ``names``: parallel handedness
    labels ("Left"/"Right"/"Unknown"). Preferred hand wins when present,
    else the largest (closest, most intentional) hand.
    """
    if not hands:
        return -1
    if prefer:
        for i, name in enumerate(names):
            if name == prefer:
                return i
    return max(range(len(hands)), key=lambda i: hand_area(hands[i]))

# (tip, dip, pip, mcp) per finger — used by curl detection
FINGERS = {
    "thumb": (THUMB_TIP, THUMB_IP, THUMB_MCP, THUMB_CMC),
    "index": (INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
    "ring": (RING_TIP, RING_DIP, RING_PIP, RING_MCP),
    "pinky": (PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
}

# Fingers whose curl state matters for common gestures (all but thumb use
# the same geometry; thumb is special-cased elsewhere).
NON_THUMB_FINGERS = ("index", "middle", "ring", "pinky")
