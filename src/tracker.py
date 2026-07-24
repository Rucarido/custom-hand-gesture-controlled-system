"""MediaPipe Hand Landmarker (Tasks API, v0.10+) — 21 landmarks per hand."""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from model_utils import ensure_hand_model


class HandTracker:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_PIP = 6

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        model_path = str(ensure_hand_model())
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            #inside above line add delegate = BaseOptions.delegate.GPU as a parameter
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.perf_counter() * 1000)

    def process(self, frame_bgr, timestamp_s: float | None = None):
        """
        Returns dict with landmarks (21 x [x,y,z] normalized 0-1) or None.
        Uses VIDEO mode — timestamp must increase each call.
        """
        if timestamp_s is None:
            timestamp_ms = int(time.perf_counter() * 1000) - self._start_ms
        else:
            timestamp_ms = int(timestamp_s * 1000)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb.copy(),  # contiguous uint8 array
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand]

        handedness = "Unknown"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        return {
            "landmarks": landmarks,
            "handedness": handedness,
        }

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def is_index_extended(landmarks) -> bool:
        wrist = landmarks[HandTracker.WRIST]
        tip = landmarks[HandTracker.INDEX_TIP]
        pip = landmarks[HandTracker.INDEX_PIP]
        tip_dist = (tip[0] - wrist[0]) ** 2 + (tip[1] - wrist[1]) ** 2
        pip_dist = (pip[0] - wrist[0]) ** 2 + (pip[1] - wrist[1]) ** 2
        return tip_dist > pip_dist
