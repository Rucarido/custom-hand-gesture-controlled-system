"""MediaPipe Hand Landmarker (Tasks API, v0.10+) — 21 landmarks per hand."""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import landmarks as L
from model_utils import ensure_hand_model


def _to_ms(timestamp_s: float, start_ms: int) -> int:
    """VIDEO-mode stamp: wall-clock seconds -> ms since tracker start.

    Guarded: a bad stamp must never kill the loop — fall back to "now".
    """
    try:
        return int(float(timestamp_s) * 1000)
    except (TypeError, ValueError):
        return int(time.perf_counter() * 1000) - start_ms


class HandTracker:
    # Re-exported indices — single source of truth lives in landmarks.py.
    WRIST = L.WRIST
    THUMB_TIP = L.THUMB_TIP
    INDEX_TIP = L.INDEX_TIP
    INDEX_PIP = L.INDEX_PIP

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        use_gpu: bool = True,
        prefer_hand: str = "",
    ) -> None:
        # NOTE: the Python Tasks API exposes no GPU delegate — inference runs
        # on CPU. The flag is accepted (config-compatible) and reserved for a
        # future delegate; we log so the setting never silently misleads.
        if use_gpu:
            print("[tracker] GPU delegate unavailable in Python Tasks API — CPU")
        self.prefer_hand = prefer_hand or ""
        model_path = str(ensure_hand_model())
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_ms = _to_ms(time.perf_counter(), 0)

    def process_all(self, frame_bgr, timestamp_s: float | None = None):
        """
        Returns list of dicts [{landmarks, handedness}] (empty if no hands).
        Hardware-facing (MediaPipe); game_mode uses this for the two-hand
        1+1 toggle, main.py pointer mode keeps using process().
        """
        if timestamp_s is None:
            timestamp_ms = _to_ms(time.perf_counter(), self._start_ms)
        else:
            timestamp_ms = _to_ms(timestamp_s, self._start_ms)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb.copy(),  # contiguous uint8 array
        )
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        hands = [[[lm.x, lm.y, lm.z] for lm in h] for h in result.hand_landmarks]
        names = [
            cats[0].category_name if cats else "Unknown"
            for cats in (result.handedness or [])
        ]
        names += ["Unknown"] * (len(hands) - len(names))
        return [
            {"landmarks": h, "handedness": n}
            for h, n in zip(hands, names)
        ]

    def process(self, frame_bgr, timestamp_s: float | None = None):
        """
        Returns dict with landmarks (21 x [x,y,z] normalized 0-1) or None.
        Uses VIDEO mode — timestamp must increase each call.
        With several hands visible, the preferred (or largest) wins.
        """
        all_hands = self.process_all(frame_bgr, timestamp_s)
        if not all_hands:
            return None

        hands = [h["landmarks"] for h in all_hands]
        names = [h["handedness"] for h in all_hands]
        idx = L.choose_hand_index(hands, names, self.prefer_hand)

        return all_hands[idx]

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def is_index_extended(landmarks) -> bool:
        # Curl-robust check (knuckle angle + radial) — see landmarks.py.
        return L.is_index_extended(landmarks)
