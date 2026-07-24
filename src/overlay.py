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

        # line between thumb and index
        t = landmarks[HandTracker.THUMB_TIP]
        idx = landmarks[HandTracker.INDEX_TIP]
        cv2.line(
            frame,
            (int(t[0] * w), int(t[1] * h)),
            (int(idx[0] * w), int(idx[1] * h)),
            (255, 200, 0),
            2,
        )


def draw_hud(frame, lines: list[str]):
    y = 24
    for line in lines:
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += 22
