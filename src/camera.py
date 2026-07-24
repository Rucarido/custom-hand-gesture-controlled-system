"""Webcam capture with low-latency settings."""

import time

import cv2


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        buffer_size: int = 1,
    ) -> None:
        # CAP_DSHOW = DirectShow backend — often faster on Windows
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        print(f"[camera] opened index={index} resolution={actual_w}x{actual_h}")

    def read(self):
        """Return (frame, timestamp) or (None, None) on failure."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return frame, time.perf_counter()

    def release(self) -> None:
        self.cap.release()
