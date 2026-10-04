"""Game overlay HUD — virtual D-pad + gesture state on the camera frame.

The window is kept small and always-on-top (see game_mode.make_topmost)
so it floats over the browser game like a controller overlay while
pynput sends keys globally to whatever window has focus (the game).

Layout (bottom-right corner of the frame):
    [^]            active direction glows green, idle is grey
  [<][v][>]       center dot = STOP zone
Plus top-left HUD lines: FPS, pose, pinch ratio, held keys, toggles.
"""

import cv2

_ACTIVE = (0, 255, 0)
_IDLE = (90, 90, 90)
_TEXT = (0, 255, 0)
_WARN = (0, 200, 255)
_BATTLE = (0, 0, 255)
_BATTLE_DIM = (60, 60, 160)


def _btn(frame, cx, cy, s, label, active):
    color = _ACTIVE if active else _IDLE
    thick = -1 if active else 2
    x1, y1, x2, y2 = cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    if active:
        cv2.putText(frame, label, (cx - 8, cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, label, (cx - 8, cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_dpad(frame, direction: str, run: bool = False):
    """Draw a virtual D-pad; `direction` in up/down/left/right/stop."""
    h, w = frame.shape[:2]
    s = max(28, min(w, h) // 9)
    gap = s + 6
    cx, cy = w - 2 * gap - s, h - 2 * gap - s
    _btn(frame, cx, cy - gap, s, "^", direction == "up")
    _btn(frame, cx - gap, cy, s, "<", direction == "left")
    _btn(frame, cx, cy, s, "o" if direction == "stop" else "*", direction == "stop")
    _btn(frame, cx + gap, cy, s, ">", direction == "right")
    _btn(frame, cx, cy + gap, s, "v", direction == "down")
    if run:
        cv2.putText(frame, "RUN", (cx - 18, cy - gap - s // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ACTIVE, 2, cv2.LINE_AA)
    # joystick deadzone crosshair near top-right for aiming feedback
    return (cx, cy)


def draw_joystick_dot(frame, palm_xy, center=(0.5, 0.5), mirror_x=True):
    """Show where the palm sits vs the deadzone (mini radar, top-right)."""
    h, w = frame.shape[:2]
    rw, rh = 110, 110
    x0, y0 = w - rw - 10, 10
    cv2.rectangle(frame, (x0, y0), (x0 + rw, y0 + rh), (80, 80, 80), 1)
    ccx, ccy = x0 + rw // 2, y0 + rh // 2
    cv2.circle(frame, (ccx, ccy), 16, (80, 80, 80), 1)  # deadzone ring
    cv2.circle(frame, (ccx, ccy), 2, (80, 80, 80), -1)
    if palm_xy is not None:
        px, py = palm_xy
        if mirror_x:
            px = 1.0 - px
        dx = max(-0.5, min(0.5, px - center[0])) * 2.0
        dy = max(-0.5, min(0.5, py - center[1])) * 2.0
        dot = (int(ccx + dx * (rw // 2)), int(ccy + dy * (rh // 2)))
        cv2.circle(frame, dot, 6, _ACTIVE, -1)
        cv2.line(frame, (ccx, ccy), dot, _ACTIVE, 1)


def draw_context_badge(frame, context: str, forced: bool = False):
    """Top-center OVERWORLD/BATTLE badge; red border in battle."""
    h, w = frame.shape[:2]
    label = f"{context.upper()}{'' if not forced else ' (FORCED)'}"
    color = _BATTLE if context == "battle" else _ACTIVE
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = (w - tw) // 2
    cv2.rectangle(frame, (x - 10, 6), (x + tw + 10, 14 + th + 10), color, 2)
    cv2.putText(frame, label, (x, 14 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if context == "battle":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _BATTLE, 3)


def draw_battle_menu(frame, selected: int | None = None):
    """2x2 move grid bottom-right; selected in {0..3} lights red.

    selected = index of the last fired move (clears when fingers change).
    Labels 1-4 match the finger counts that fire them.
    """
    h, w = frame.shape[:2]
    s = max(30, min(w, h) // 8)
    gap = 6
    cx, cy = w - s - gap - 10, h - s - gap - 10
    boxes = [(cx - s - gap, cy - s - gap, "1"), (cx, cy - s - gap, "2"),
             (cx - s - gap, cy, "3"), (cx, cy, "4")]
    for i, (bx, by, label) in enumerate(boxes):
        active = (selected is not None and i == selected)
        color = _BATTLE if active else _BATTLE_DIM
        thick = -1 if active else 2
        cv2.rectangle(frame, (bx, by), (bx + s, by + s), color, thick)
        fg = (255, 255, 255) if active else color
        cv2.putText(frame, label, (bx + s // 2 - 7, by + s // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2 if active else 1,
                    cv2.LINE_AA)


def draw_duo_meter(frame, progress: float, latched: bool = False):
    """Two-hand 1+1 toggle hold meter (bottom-center).

    Fills as both-1 accumulates; "RELEASE" shows while latched (toggle
    fired — lower a hand to re-arm for the return trip).
    """
    h, w = frame.shape[:2]
    bw, bh = 180, 14
    x0, y0 = (w - bw) // 2, h - bh - 8
    label = "1+1 RELEASE" if latched else "1+1 BATTLE"
    cv2.putText(frame, label, (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1)
    fill = int(bw * min(1.0, max(0.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bh), _BATTLE, -1)


def draw_cursor_marker(frame, pos, pinching: bool = False,
                         locked: bool = False):
    """Crosshair ring at the battle-cursor index tip (frame pixels).

    Locked draws red with a LOCKED tag (cursor parked — pinch still
    clicks there); pinching draws white.
    """
    x, y = int(pos[0]), int(pos[1])
    if locked:
        color, tag = (0, 0, 255), "LOCKED"
    elif pinching:
        color, tag = (255, 255, 255), "CURSOR"
    else:
        color, tag = _ACTIVE, "CURSOR"
    cv2.circle(frame, (x, y), 14, color, 2)
    cv2.circle(frame, (x, y), 3, color, -1)
    cv2.line(frame, (x - 22, y), (x - 8, y), color, 2)
    cv2.line(frame, (x + 8, y), (x + 22, y), color, 2)
    cv2.line(frame, (x, y - 22), (x, y - 8), color, 2)
    cv2.line(frame, (x, y + 8), (x, y + 22), color, 2)
    cv2.putText(frame, tag, (x + 18, y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_game_hud(frame, lines: list[str]):
    y = 24
    for line in lines:
        color = _WARN if line.startswith(("!", "PAUSED", "KEYS OFF")) else _TEXT
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def draw_help(frame, mode: str = "fingers", context: str = "overworld",
              interact: str = "taps"):
    """Legend strip at the bottom (toggle with 'h')."""
    h, w = frame.shape[:2]
    if context == "battle" and interact == "cursor":
        help_lines = [
            "BATTLE-CURSOR: point index to aim | PINCH: click to select",
            "still 1s = auto-lock (pinch still clicks) | move 2s = free",
            "1+1 both hands: back to movement | v=duo-toggle q=quit",
        ]
    elif context == "battle":
        help_lines = [
            "BATTLE: 1/2/3/4 fingers = move 1/2/3/4 | FIST=cancel",
            "PINCH: confirm | hold shape, release to re-fire",
            "b=auto/force battle/force overworld q=quit h=help",
        ]
    elif mode == "fingers":
        help_lines = [
            "FIST=stop | 1=forward 2=back 3=left 4=right",
            "1+1 both hands: battle-cursor | PINCH: confirm (Space)",
            "b=battle ctx v=duo-toggle k=keys on/off q=quit h=help",
        ]
    else:
        help_lines = [
            "MOVE hand from center: walk | center = stop",
            "PINCH: confirm (Space) | FIST hold: cancel (X)",
            "OPEN PALM hold: menu (Enter) | PEACE: run (Shift)",
            "b=battle ctx k=keys on/off q=quit h=help c=recenter",
        ]
    y = h - len(help_lines) * 18 - 8
    for line in help_lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
