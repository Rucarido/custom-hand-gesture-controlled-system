"""Webcam capture with low-latency, cross-platform settings."""

import time

import cv2

# Backend map: config `camera.backend` -> OpenCV flag. `auto` tries the OS
# native backend first (fastest buffer path), then falls back to ANY.
_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}


def resolve_backend(name: str) -> int:
    """Pure helper — maps a config backend name to an OpenCV flag."""
    name = (name or "auto").lower()
    if name == "auto":
        import sys

        if sys.platform.startswith("win"):
            return cv2.CAP_DSHOW
        if sys.platform == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_V4L2
    return _BACKENDS.get(name, cv2.CAP_ANY)


def _prop(cap, prop, default: float = 0.0) -> float:
    """Read a capture property defensively — some drivers return None."""
    try:
        value = cap.get(prop)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _prop_int(cap, prop, default: int = 0) -> int:
    """Integer capture property — conversion guarded inside."""
    try:
        value = cap.get(prop)
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        buffer_size: int = 1,
        fps: int = 30,
        backend: str = "auto",
        auto_focus: bool = False,
        auto_exposure: bool = False,
    ) -> None:
        flag = resolve_backend(backend)
        self.cap = cv2.VideoCapture(index, flag)
        if not self.cap.isOpened() and flag != cv2.CAP_ANY:
            # Native backend unavailable (e.g. V4L2 quirk) — fall back.
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        # MJPEG packs 720p within USB2 bandwidth — avoids forced low-fps YUYV.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        if not auto_focus:
            # Lock focus: AF hunting breathes the image and wobbles landmarks.
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not auto_exposure:
            # Manual exposure (0.25 = manual mode on most UVC drivers):
            # stable brightness = stable detection thresholds.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        actual_w = _prop_int(self.cap, cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = _prop_int(self.cap, cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = _prop(self.cap, cv2.CAP_PROP_FPS)

        # Warmup: discard first frames while AE/AWB settles.
        for _ in range(5):
            self.cap.read()

        print(
            f"[camera] opened index={index} backend={backend} "
            f"resolution={actual_w}x{actual_h} fps={actual_fps:.0f}"
        )

    def read(self):
        """Return (frame, timestamp) or (None, None) on failure."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return frame, time.perf_counter()

    def release(self) -> None:
        self.cap.release()
