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
        auto_focus: bool = True,
        auto_exposure: bool = True,
        exposure: float | None = None,
        gain: float | None = None,
        brightness: float | None = None,
        focus: float | None = None,
        auto_white_balance: bool | None = None,
        warmup_frames: int | None = None,
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

        is_dshow = flag == cv2.CAP_DSHOW

        # --- Exposure: default to AUTO (matches Camera app / Meet / Zoom).
        # Forcing manual exposure without an explicit EXPOSURE value leaves
        # most UVC drivers at minimum exposure -> near-black image
        # (measured mean pixel 117 -> 2.6 on this machine). So manual mode
        # is only engaged when the user also supplies `exposure` (and
        # optionally `gain`).
        if auto_exposure:
            # Re-enable auto: 0.75 = auto on DSHOW/UVC, 1.0 = auto elsewhere.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if is_dshow else 1.0)
            if exposure is not None:
                print("[camera] note: `exposure` ignored while auto_exposure=true")
            if gain is not None:
                self.cap.set(cv2.CAP_PROP_GAIN, gain)
        else:
            if exposure is None and gain is None:
                # Fail-safe: stay in auto instead of going black.
                print(
                    "[camera] WARNING: auto_exposure=false but no "
                    "`exposure`/`gain` given — staying in AUTO to avoid "
                    "a black image. Set camera.exposure (e.g. -5) to lock."
                )
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if is_dshow else 1.0)
            else:
                # Manual exposure (0.25 = manual on DSHOW/UVC, 0 = manual
                # on some V4L2 drivers) + explicit value = stable brightness.
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if is_dshow else 0.0)
                if exposure is not None:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                if gain is not None:
                    self.cap.set(cv2.CAP_PROP_GAIN, gain)

        # --- Focus: default to AUTO (matches other apps). Lock only on request.
        if auto_focus:
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass
            if focus is not None:
                print("[camera] note: `focus` ignored while auto_focus=true")
        else:
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
            if focus is not None:
                self.cap.set(cv2.CAP_PROP_FOCUS, focus)

        if brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        if auto_white_balance is not None:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
            except Exception:
                pass

        actual_w = _prop_int(self.cap, cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = _prop_int(self.cap, cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = _prop(self.cap, cv2.CAP_PROP_FPS)

        # Warmup: let AE/AWB settle when auto (matches Camera app behavior).
        # 5 frames is not enough for auto-exposure to converge from dark.
        if warmup_frames is None:
            warmup_frames = 30 if (auto_exposure or auto_focus) else 5
        for _ in range(max(1, warmup_frames)):
            self.cap.read()

        print(
            f"[camera] opened index={index} backend={backend} "
            f"resolution={actual_w}x{actual_h} fps={actual_fps:.0f} "
            f"auto_exposure={auto_exposure} auto_focus={auto_focus}"
        )

    def read(self):
        """Return (frame, timestamp) or (None, None) on failure."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return frame, time.perf_counter()

    def release(self) -> None:
        self.cap.release()
