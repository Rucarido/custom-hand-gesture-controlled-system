"""Draw debug HUD on camera frame."""

import cv2

from tracker import HandTracker


def draw_hand(frame, hand_result, tracker_cfg: dict):
    if not hand_result:
        return

    h, w = frame.shape[:2]
    landmarks = hand_result["landmarks"]

    if tracker_cfg.get("show_landmarks", True):
        for i, (x, y, _z) in enumerate(landmarks):
            px, py = int(x * w), int(y * h)
            color = (0, 255, 0) if i in (4, 8) else (100, 100, 100)
            cv2.circle(frame, (px, py), 4 if i in (4, 8) else 2, color, -1)

        t = landmarks[HandTracker.THUMB_TIP]
        idx = landmarks[HandTracker.INDEX_TIP]
        cv2.line(
            frame,
            (int(t[0] * w), int(t[1] * h)),
            (int(idx[0] * w), int(idx[1] * h)),
            (255, 200, 0),
            2,
        )


def draw_motion_area(frame, motion_area: dict, mirror: bool):
    if not motion_area.get("enabled", True):
        return

    h, w = frame.shape[:2]
    x1 = motion_area["x_min"]
    x2 = motion_area["x_max"]
    y1 = motion_area["y_min"]
    y2 = motion_area["y_max"]

    if mirror:
        x1, x2 = 1.0 - x2, 1.0 - x1

    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))
    cv2.rectangle(frame, p1, p2, (0, 200, 100), 2)


def draw_lock_indicator(frame, locked: bool):
    if not locked:
        return

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)
    cv2.putText(
        frame, "LOCKED", (w // 2 - 60, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2, cv2.LINE_AA,
    )


def draw_hud(frame, lines: list[str]):
    y = 24
    for line in lines:
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += 22
