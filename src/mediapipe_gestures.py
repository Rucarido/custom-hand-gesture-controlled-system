"""Landmark-based gesture recognition ported from hand-gesture-recognition-mediapipe.

Ports the vendored sample's ML stack (Apache-2.0, by Kazuhito00 / kinivi) onto
the Tasks-API HandLandmarker this project already uses:

  21 landmarks -> pre_process_landmark -> KeyPointClassifier (TFLite, [1,42])
                 -> hand sign (Open / Close / Pointer / OK)
  index-tip trail while "Pointer" -> pre_process_point_history
                 -> PointHistoryClassifier (TFLite, [1,32])
                 -> motion gesture (Stop / Clockwise / Counter Clockwise / Move)

Differences from the sample's app.py (kept deliberate):
  * Input is normalized float coordinates straight from the Tasks API
    (the sample feeds mirrored integer pixel coordinates - same math, since
    both pre-processing steps are translation+scale invariant).
  * TFLite runs through LiteRT (ai_edge_litert) when installed, falling back
    to tf.lite. No MediaPipe legacy-solutions code is used (gone in MP 1.x).
  * The motion-vote history only accumulates once the index-tip trail is full,
    so it never reports "Stop" while the history is still priming.
"""

from __future__ import annotations

import csv
import itertools
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "mediapipe_gestures"

POINT = 2  # "Pointer" hand sign id - drives the index-tip trail


def load_labels(csv_path: str | Path) -> list[str]:
    """Label CSV = one class name per row, first column."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return [row[0].strip() for row in csv.reader(f) if row]


def load_labels_with_fallback(csv_path: str | Path, fallback: list[str]) -> list[str]:
    try:
        return load_labels(csv_path)
    except OSError:
        return list(fallback)


def _create_interpreter(model_path: str | Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    else:
        interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    try:
        interpreter.set_num_threads(1)
    except AttributeError:
        pass
    return interpreter


class TFLiteClassifier:
    """[1, N] float32 in -> argmax class id out (Softmax MLP exported to TFLite).

    Mirrors model/keypoint_classifier.py in the vendored sample; `score_th`
    reproduces PointHistoryClassifier's confidence floor (low-confidence
    result -> `invalid_value` instead of the argmax).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int,
        score_th: float = 0.0,
        invalid_value: int = 0,
    ):
        import numpy as np

        self._np = np
        self.interpreter = _create_interpreter(model_path)
        self.input_size = input_size
        self.score_th = score_th
        self.invalid_value = invalid_value
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, features: Sequence[float]) -> int:
        batch = self._np.asarray(features, dtype=self._np.float32)
        batch = batch.reshape(1, self.input_size)
        self.interpreter.set_tensor(self.input_index, batch)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_index)[0]
        class_id = int(self._np.argmax(scores))
        if self.score_th > 0.0 and float(self._np.max(scores)) < self.score_th:
            return self.invalid_value
        return class_id


class KeyPointClassifier(TFLiteClassifier):
    def __init__(self, model_path: str | Path):
        super().__init__(model_path, input_size=42)


class PointHistoryClassifier(TFLiteClassifier):
    def __init__(self, model_path: str | Path, score_th: float = 0.5):
        super().__init__(model_path, input_size=32, score_th=score_th)


def pre_process_landmark(landmark_list: Sequence[Sequence[float]]) -> list[float]:
    """Wrist-relative + max-abs normalized, flattened to 42 floats.

    Identical math to the sample's pre_process_landmark(); translation and
    uniform scale invariant, so it works the same on pixel ints or the
    normalized floats from the Tasks API.
    """
    points = [[float(p[0]), float(p[1])] for p in landmark_list[:21]]
    base_x, base_y = points[0]
    relative: list[float] = []
    for x, y in points:
        relative.extend((x - base_x, y - base_y))
    max_value = max(abs(v) for v in relative)
    if max_value == 0.0:
        max_value = 1.0
    return [v / max_value for v in relative]


def pre_process_point_history(
    point_history: Sequence[Sequence[float]],
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> list[float]:
    """Trail points relativized to the oldest point, flattened to 32 floats.

    The sample divides x/y deltas by image width/height; with normalized
    floats that is 1.0, but the parameters stay so sample-trained data
    remains interchangeable.
    """
    points = [[float(p[0]), float(p[1])] for p in point_history]
    base_x, base_y = points[0]
    relative: list[float] = []
    for x, y in points:
        relative.extend(((x - base_x) / image_width, (y - base_y) / image_height))
    return relative


class MediaPipeGestureRecognizer:
    """Per-frame driver replicating app.py's recognition loop.

    Feed `update()` the tracker's 21 landmarks (normalized [x, y, z] or None
    when no hand is visible). Returns a result dict per frame once the trail
    history is primed, else None / a result without motion info.
    """

    HAND_SIGN_POINTER = POINT

    def __init__(
        self,
        keypoint_model: str | Path,
        point_history_model: str | Path,
        *,
        history_length: int = 16,
        mirror_x: bool = False,
    ):
        self.hand_sign_classifier = KeyPointClassifier(keypoint_model)
        self.hand_sign_labels = load_labels(
            Path(keypoint_model).with_name("keypoint_classifier_label.csv")
        )
        self.point_classifier = PointHistoryClassifier(point_history_model)
        self.point_labels = load_labels(
            Path(point_history_model).with_name("point_history_classifier_label.csv")
        )
        self.history_length = history_length
        self.mirror_x = mirror_x
        self.point_history: deque[list[float]] = deque(maxlen=history_length)
        self.motion_history: deque[int] = deque(maxlen=history_length)

    @classmethod
    def from_config(cls, cfg: dict) -> "MediaPipeGestureRecognizer":
        models_dir = Path(cfg.get("models_dir", DEFAULT_MODELS_DIR))
        return cls(
            keypoint_model=models_dir / "keypoint_classifier.tflite",
            point_history_model=models_dir / "point_history_classifier.tflite",
            history_length=int(cfg.get("history_length", 16)),
            mirror_x=bool(cfg.get("mirror_x", False)),
        )

    def reset(self) -> None:
        self.point_history.clear()
        self.motion_history.clear()

    def _mirror(self, point: Sequence[float]) -> list[float]:
        if self.mirror_x:
            return [1.0 - float(point[0]), float(point[1])]
        return [float(point[0]), float(point[1])]

    def update(self, landmarks: Sequence[Sequence[float]] | None) -> dict | None:
        if landmarks is None:
            # Sample behavior: keep the trail grounded while no hand is visible.
            self.point_history.append([0.0, 0.0])
            return None

        normalized = [self._mirror(p) for p in landmarks[:21]]

        landmark_features = pre_process_landmark(normalized)
        hand_sign_id = self.hand_sign_classifier(landmark_features)
        if hand_sign_id == self.HAND_SIGN_POINTER:
            self.point_history.append(normalized[8])
        else:
            self.point_history.append([0.0, 0.0])

        result = {
            "hand_sign_id": hand_sign_id,
            "hand_sign": self.hand_sign_labels[hand_sign_id],
            "gesture_id": None,
            "gesture": None,
            "landmark_features": landmark_features,
            "history_features": None,
        }

        if len(self.point_history) == self.history_length:
            features = pre_process_point_history(list(self.point_history))
            result["history_features"] = features
            gesture_id = self.point_classifier(features)
            self.motion_history.append(gesture_id)
            votes = Counter(self.motion_history).most_common()
            result["gesture_id"] = votes[0][0]
            result["gesture"] = self.point_labels[votes[0][0]]
            result["gesture_votes"] = votes
        return result


class SignAimGate:
    """Debounced hysteresis on "hand sign == Pointer".

    Needs `enter_frames` consecutive Pointer signs to claim aiming, then
    `exit_frames` consecutive non-Pointer signs to drop it. Consumers treat
    False as "no sign vouching" (the rule gate still decides) — only True
    overrides anything.
    """

    def __init__(self, enter_frames: int = 3, exit_frames: int = 4):
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self._on = 0
        self._off = 0
        self._aiming = False

    def update(self, is_pointer: bool) -> bool:
        if is_pointer:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0
        if not self._aiming and self._on >= self.enter_frames:
            self._aiming = True
        elif self._aiming and self._off >= self.exit_frames:
            self._aiming = False
        return self._aiming

    def reset(self) -> None:
        self._on = 0
        self._off = 0
        self._aiming = False


class GestureTapGate:
    """Fire once when a label holds for `stable_frames`; a label change (or
    reset) re-arms firing. A minimum `cooldown_ms` guards against flap-fire
    on unstable labels."""

    def __init__(self, stable_frames: int = 6, cooldown_ms: float = 600.0):
        self.stable_frames = max(1, int(stable_frames))
        self.cooldown_s = max(0.0, float(cooldown_ms)) / 1000.0
        self._label: str | None = None
        self._count = 0
        self._last_fire_t: float | None = None

    def update(self, label: str | None, now: float) -> str | None:
        if label is None or label != self._label:
            self._label = label
            self._count = 1 if label is not None else 0
            return None
        self._count += 1
        if self._count == self.stable_frames:
            if (
                self._last_fire_t is None
                or (now - self._last_fire_t) >= self.cooldown_s
            ):
                self._last_fire_t = now
                return label
        return None

    def reset(self) -> None:
        self._label = None
        self._count = 0
        self._last_fire_t = None


class FeatureLogger:
    """CSV logging identical to the vendored sample's logging_csv(): headerless
    rows of `class_id, feature0..N` for capture/training."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, number: int, features: Sequence[float]) -> None:
        path = self.directory / name
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([int(number), *features])


class GameGestures:
    """Per-handedness recognizers + game wiring for the ported recognizers.

    One MediaPipeGestureRecognizer per handedness label (each hand keeps its
    own trail/gesture history), so sign taps, motion taps, the debounced
    Pointer aim-gate and CSV capture (= the sample's k/h data logging) can
    all run from the game loop.
    """

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        game = cfg.get("game", {}) or {}
        aim_cfg = game.get("aim_gate", {}) or {}
        self.aim_enabled = bool(aim_cfg.get("enabled", True))
        sign_tap_cfg = game.get("sign_tap", {}) or {}
        motion_tap_cfg = game.get("motion_tap", {}) or {}
        self.sign_gate = GestureTapGate(
            stable_frames=sign_tap_cfg.get("stable_frames", 6),
            cooldown_ms=sign_tap_cfg.get("cooldown_ms", 600),
        )
        self.motion_gate = GestureTapGate(
            stable_frames=motion_tap_cfg.get("stable_frames", 2),
            cooldown_ms=motion_tap_cfg.get("cooldown_ms", 900),
        )
        self.sign_map: dict[str, str] = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in (game.get("sign_taps", {}) or {}).items()
            if str(v).strip()
        }
        self.motion_map: dict[str, str] = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in (game.get("motion_taps", {}) or {}).items()
            if str(v).strip()
        }
        collection_cfg = game.get("collection", {}) or {}
        self.logger = (
            FeatureLogger(collection_cfg["dir"])
            if collection_cfg.get("dir") else None
        )
        self.recognizers: dict[str, MediaPipeGestureRecognizer] = {}
        self.aim_gates: dict[str, SignAimGate] = {}
        self.models_ok = False
        self._warned = False

    def _recognizer(self, label: str) -> MediaPipeGestureRecognizer | None:
        rec = self.recognizers.get(label)
        if rec is not None:
            return rec
        try:
            rec = MediaPipeGestureRecognizer.from_config(self.cfg)
        except Exception as exc:
            if not self._warned:
                print(f"[game] mediapipe gestures unavailable: {exc}")
                self._warned = True
            return None
        self.recognizers[label] = rec
        self.models_ok = True
        return rec

    def aim_gate(self, label: str) -> SignAimGate:
        gate = self.aim_gates.get(label)
        if gate is None:
            aim_cfg = (self.cfg.get("game", {}) or {}).get("aim_gate", {}) or {}
            gate = SignAimGate(
                enter_frames=aim_cfg.get("enter_frames", 3),
                exit_frames=aim_cfg.get("exit_frames", 4),
            )
            self.aim_gates[label] = gate
        return gate

    def update(self, label: str, landmarks) -> dict | None:
        rec = self._recognizer(label)
        if rec is None:
            return None
        return rec.update(landmarks)

    def aim(self, label: str, res: dict | None) -> bool:
        """True only if this hand's Pointer sign is debounced in."""
        if not self.aim_enabled or res is None:
            return False
        return self.aim_gate(label).update(
            res["hand_sign_id"] == MediaPipeGestureRecognizer.HAND_SIGN_POINTER
        )

    def taps(self, res: dict | None, now: float) -> dict:
        out = {"sign": None, "motion": None}
        if res is None:
            self.sign_gate.update(None, now)
            self.motion_gate.update(None, now)
            return out
        sign = str(res.get("hand_sign", "")).lower()
        motion = res.get("gesture")
        fired_sign = self.sign_gate.update(sign, now)
        if fired_sign:
            out["sign"] = self.sign_map.get(fired_sign)
        fired_motion = self.motion_gate.update(motion, now)
        if fired_motion:
            out["motion"] = self.motion_map.get(fired_motion.lower())
        return out

    def collect(self, mode: int, number: int, res: dict | None) -> None:
        if self.logger is None or res is None:
            return
        if mode == 1 and res.get("landmark_features"):
            self.logger.log("keypoint.csv", number, res["landmark_features"])
        elif mode == 2 and res.get("history_features"):
            self.logger.log("point_history.csv", number, res["history_features"])

    def drop_missing(self, labels_seen: list[str]) -> None:
        for stale in [k for k in self.recognizers if k not in labels_seen]:
            del self.recognizers[stale]
        for stale in [k for k in self.aim_gates if k not in labels_seen]:
            del self.aim_gates[stale]

    def reset(self) -> None:
        self.recognizers.clear()
        self.aim_gates.clear()
        self.sign_gate.reset()
        self.motion_gate.reset()
