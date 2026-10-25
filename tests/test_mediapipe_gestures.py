"""Tests for the hand-gesture-recognition-mediapipe port (mediapipe_gestures).

Pure-logic tests never touch TF/LiteRT: stub classifiers replace the TFLite
wrappers. One optional test runs the real shipped .tflite models and skips
when no runtime is installed (the app itself degrades the same way).
"""

import csv
from pathlib import Path

import pytest

import mediapipe_gestures as mpg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "mediapipe_gestures"
VENDORED_KP_CSV = (
    PROJECT_ROOT
    / "hand-gesture-recognition-mediapipe"
    / "model"
    / "keypoint_classifier"
    / "keypoint.csv"
)


def make_hand():
    """21 normalized landmarks: wrist + spread joints (index tip at 8)."""
    wrist = (0.0, 0.0)
    spread = [
        (0.1, -0.1), (0.1, -0.2), (0.2, -0.05), (0.3, -0.25),
        (0.15, -0.3), (0.2, -0.4), (0.25, -0.2), (0.3, -0.25),
        (0.1, -0.2), (0.15, -0.3), (0.5, -0.3), (0.6, -0.4),
        (0.2, -0.4), (0.3, -0.5), (0.7, -0.2), (0.8, -0.3),
    ]
    pts = [wrist, (0.05, -0.05), *spread]
    while len(pts) < 21:
        pts.append((0.05, -0.05))
    return [[x, y, 0.0] for x, y in pts[:21]]


# ---------------------------------------------------------------- preprocessing


def test_pre_process_landmark_is_wrist_relative_and_max_normalized():
    feats = mpg.pre_process_landmark(make_hand())
    assert len(feats) == 42
    assert feats[0] == 0.0 and feats[1] == 0.0  # wrist-relative
    assert max(abs(v) for v in feats) == pytest.approx(1.0)  # max-abs norm


def test_pre_process_landmark_scale_and_translation_invariant():
    base = make_hand()
    scaled = [[x * 3.0 + 7.0, y * 3.0 - 2.0, 0.0] for x, y, _ in base]
    assert mpg.pre_process_landmark(scaled) == pytest.approx(
        mpg.pre_process_landmark(base), abs=1e-6
    )


def test_pre_process_landmark_matches_vendored_training_rows():
    rows = list(csv.reader(VENDORED_KP_CSV.open()))
    for row in rows[::500]:
        feats = [float(v) for v in row]
        points = [[feats[1 + 2 * i], feats[2 + 2 * i], 0.0] for i in range(21)]
        assert mpg.pre_process_landmark(points) == pytest.approx(feats[1:], abs=1e-6)


def test_pre_process_point_history_relativizes_to_oldest():
    history = [[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]]
    feats = mpg.pre_process_point_history(history)
    assert len(feats) == 6
    assert feats == pytest.approx([0.0, 0.0, 0.1, 0.2, 0.3, 0.4])


def test_pre_process_point_history_divides_by_image_size():
    history = [[0.0, 0.0], [0.5, 0.25]]
    feats = mpg.pre_process_point_history(history, image_width=2.0, image_height=4.0)
    assert feats == pytest.approx([0.0, 0.0, 0.25, 0.0625])


# ---------------------------------------------------------------- vote driver


class StubClassifier:
    """Replaces the KeyPoint/PointHistory TFLite wrappers; records inputs."""

    def __init__(self, results):
        self.results = results
        self.features_seen = []

    def __call__(self, features):
        self.features_seen.append(list(features))
        if callable(self.results):
            return self.results()
        return self.results


@pytest.fixture
def build_recognizer(monkeypatch, tmp_path):
    def build(kp_result, ph_result, **kwargs):
        kp = StubClassifier(kp_result)
        ph = StubClassifier(ph_result)
        monkeypatch.setattr(mpg, "KeyPointClassifier", lambda model: kp)
        monkeypatch.setattr(mpg, "PointHistoryClassifier", lambda model: ph)
        (tmp_path / "keypoint_classifier_label.csv").write_text("Open\nClose\nPointer\nOK\n")
        (tmp_path / "point_history_classifier_label.csv").write_text(
            "Stop\nClockwise\nCounter Clockwise\nMove\n"
        )
        rec = mpg.MediaPipeGestureRecognizer(
            tmp_path / "kp.tflite",
            tmp_path / "ph.tflite",
            **kwargs,
        )
        rec._stub_kp = kp  # noqa: SLF001
        rec._stub_ph = ph  # noqa: SLF001
        return rec

    return build


def test_labels_load_from_model_dir(tmp_path):
    (tmp_path / "label.csv").write_text("b,c\na\n")
    assert mpg.load_labels(tmp_path / "label.csv") == ["b", "a"]


def test_driver_returns_none_while_trail_priming(build_recognizer):
    rec = build_recognizer(kp_result=0, ph_result=1)
    hand = make_hand()
    for _ in range(15):
        res = rec.update(hand)
        assert res is not None
        assert res["hand_sign"] == "Open"
        assert res["gesture"] is None
        assert len(rec._stub_ph.features_seen) == 0
    last = rec.update(hand)
    assert last is not None
    assert last["gesture"] == "Clockwise"
    assert len(rec._stub_ph.features_seen) == 1


def test_driver_trails_index_tip_only_on_pointer(build_recognizer):
    rec = build_recognizer(kp_result=lambda: mpg.POINT, ph_result=1)
    hand = make_hand()
    rec.update(hand)
    assert rec.point_history[-1] == pytest.approx([hand[8][0], hand[8][1]])
    rec2 = build_recognizer(kp_result=lambda: 1, ph_result=1)
    rec2.update(hand)
    assert rec2.point_history[-1] == [0.0, 0.0]


def test_driver_majority_vote(build_recognizer):
    ids = iter([1] * 17)
    rec = build_recognizer(kp_result=lambda: mpg.POINT, ph_result=lambda: next(ids))
    hand = make_hand()
    for _ in range(16):
        rec.update(hand)
    res = rec.update(hand)
    assert res["gesture"] == "Clockwise"
    assert res["gesture_votes"][0] == (1, 2)


def test_driver_no_hand_keeps_trail_grounded(build_recognizer):
    rec = build_recognizer(kp_result=lambda: 0, ph_result=0)
    for _ in range(20):
        assert rec.update(None) is None
    assert len(rec.point_history) == 16
    assert all(p == [0.0, 0.0] for p in rec.point_history)
    assert len(rec.motion_history) == 0


def test_driver_mirror_x(build_recognizer):
    rec = build_recognizer(kp_result=lambda: mpg.POINT, ph_result=0, mirror_x=True)
    hand = make_hand()
    rec.update(hand)
    assert rec.point_history[-1] == pytest.approx([1.0 - hand[8][0], hand[8][1]])


def test_reset_clears_histories(build_recognizer):
    rec = build_recognizer(kp_result=lambda: mpg.POINT, ph_result=0)
    for _ in range(20):
        rec.update(make_hand())
    rec.reset()
    assert len(rec.point_history) == 0
    assert len(rec.motion_history) == 0


# ---------------------------------------------------------------- real models


def _tflite_runtime_available() -> bool:
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            return False
    return True


@pytest.mark.skipif(
    not _tflite_runtime_available(), reason="no LiteRT / TF runtime installed"
)
def test_real_models_load_and_classify():
    if not MODELS_DIR.exists():
        pytest.skip("models/mediapipe_gestures not present")
    rec = mpg.MediaPipeGestureRecognizer(
        MODELS_DIR / "keypoint_classifier.tflite",
        MODELS_DIR / "point_history_classifier.tflite",
        history_length=16,
    )
    rows = list(csv.reader(VENDORED_KP_CSV.open()))
    feats = [float(v) for v in rows[0]]
    points = [[feats[1 + 2 * i], feats[2 + 2 * i], 0.0] for i in range(21)]
    res = rec.update(points)
    assert res["hand_sign_id"] == int(feats[0])
    assert res["hand_sign"] in ("Open", "Close", "Pointer", "OK")

    trail = [[0.1, 0.0], [0.2, 0.1], [0.3, 0.2], [0.2, 0.3], [0.1, 0.2], [0.0, 0.1]]
    trail = trail * 2 + trail[:4]
    hand = make_hand()
    for _ in range(15):
        for p in trail:
            updated = [list(pt) for pt in hand]
            updated[8] = [p[0], p[1], 0.0]
            res2 = rec.update(updated)
    assert res2 is not None
    assert res2["gesture"] in ("Stop", "Clockwise", "Counter Clockwise", "Move")


# ------------------------------------------------- game-mode integration


def test_sign_aim_gate_hysteresis():
    gate = mpg.SignAimGate(enter_frames=3, exit_frames=4)
    assert gate.update(True) is False
    assert gate.update(True) is False
    assert gate.update(False) is False  # blip resets the enter streak
    assert gate.update(True) is False
    assert gate.update(True) is False
    assert gate.update(True) is True  # 3rd consecutive Pointer
    assert gate.update(False) is True
    assert gate.update(False) is True  # need exit_frames to drop
    assert gate.update(False) is True
    assert gate.update(False) is False  # 4th consecutive non-Pointer


def test_gesture_tap_gate_fires_once_per_hold():
    gate = mpg.GestureTapGate(stable_frames=3, cooldown_ms=0.0)
    t = 0
    assert gate.update("OK", t) is None
    assert gate.update("OK", t) is None
    assert gate.update("OK", t) == "OK"  # held exactly stable_frames
    assert gate.update("OK", t) is None  # no refire while held
    assert gate.update("Open", t) is None  # label change
    assert gate.update("OK", t) is None
    assert gate.update("OK", t) is None
    assert gate.update("OK", t) == "OK"  # re-armed


def test_feature_logger_writes_sample_format(tmp_path):
    logger = mpg.FeatureLogger(tmp_path / "cap")
    logger.log("kp.csv", 2, [0.0] * 42)
    logger.log("kp.csv", 3, [0.5] * 42)
    rows = [r for r in csv.reader((tmp_path / "cap" / "kp.csv").open())]
    assert len(rows) == 2
    assert len(rows[0]) == 43
    assert rows[0][0] == "2" and rows[1][0] == "3"


def _game_cfg(tmp_path, **game):
    return {
        "models_dir": str(tmp_path),
        "history_length": 2,
        "mirror_x": False,
        "game": game,
    }


def _seed_labels(tmp_path):
    (tmp_path / "keypoint_classifier_label.csv").write_text(
        "Open\nClose\nPointer\nOK\n"
    )
    (tmp_path / "point_history_classifier_label.csv").write_text(
        "Stop\nClockwise\nCounter Clockwise\nMove\n"
    )


def _gg(monkeypatch, tmp_path, kp_result, ph_result, **game):
    _seed_labels(tmp_path)
    kp = StubClassifier(kp_result)
    ph = StubClassifier(ph_result)
    monkeypatch.setattr(mpg, "KeyPointClassifier", lambda model: kp)
    monkeypatch.setattr(mpg, "PointHistoryClassifier", lambda model: ph)
    return mpg.GameGestures(_game_cfg(tmp_path, **game))


HAND = make_hand()


def test_game_gestures_recognizes_per_hand(monkeypatch, tmp_path):
    gg = _gg(monkeypatch, tmp_path, kp_result=0, ph_result=0)
    res_l = gg.update("Left", HAND)
    res_r = gg.update("Right", HAND)
    assert res_l["hand_sign"] == "Open"
    assert len(gg.recognizers) == 2
    gg.drop_missing(["Left"])
    assert list(gg.recognizers) == ["Left"]
    assert list(gg.aim_gates) == []  # aim gates build lazily on aim()
    gg.aim("Left", res_l)
    assert list(gg.aim_gates) == ["Left"]


def test_game_gestures_aim_gate_uses_pointer_sign(monkeypatch, tmp_path):
    gg = _gg(monkeypatch, tmp_path, kp_result=mpg.POINT, ph_result=0)
    res = gg.update("Right", HAND)
    assert gg.aim("Right", res) is False
    assert gg.aim("Right", res) is False
    assert gg.aim("Right", res) is True  # debounced in after 3 frames


def test_game_gestures_taps_sign_click(monkeypatch, tmp_path):
    gg = _gg(
        monkeypatch, tmp_path, kp_result=3, ph_result=0,
        sign_taps={"OK": "click"}, sign_tap={"stable_frames": 2, "cooldown_ms": 0},
    )
    res = gg.update("Right", HAND)
    assert gg.taps(res, 0.0)["sign"] is None
    fired = gg.taps(res, 1.0)
    assert fired["sign"] == "click"
    assert gg.taps(res, 2.0)["sign"] is None  # no double-fire while held
    # An unmapped sign produces no key.
    gg2 = _gg(monkeypatch, tmp_path, kp_result=0, ph_result=0)
    r2 = gg2.update("Right", HAND)
    assert gg2.taps(r2, 0.0)["sign"] is None


def test_game_gestures_motion_tap(monkeypatch, tmp_path):
    gg = _gg(
        monkeypatch, tmp_path, kp_result=mpg.POINT, ph_result=1,
        motion_taps={"Clockwise": "e"},
        motion_tap={"stable_frames": 2, "cooldown_ms": 0},
    )
    res = gg.update("Right", HAND)
    assert res["gesture"] is None  # history still priming (len 2 = full -> OK)
    fired = None
    for t in range(1, 5):
        res = gg.update("Right", HAND)
        fired = gg.taps(res, float(t))["motion"] or fired
    assert fired == "e"  # Clockwise vote held -> mapped key fired


def test_game_gestures_collect_writes_csvs(monkeypatch, tmp_path):
    kp_csv = tmp_path / "cap" / "keypoint.csv"
    hist_csv = tmp_path / "cap" / "point_history.csv"
    gg = _gg(
        monkeypatch, tmp_path, kp_result=2, ph_result=0,
        collection={"dir": str(tmp_path / "cap")},
    )
    res = gg.update("Right", HAND)
    # Trail history is at full length (2) now -> both channels available.
    gg.collect(0, 5, res)
    assert not kp_csv.exists()  # mode 0 = off


def test_game_gestures_collect_logs_both_modes(monkeypatch, tmp_path):
    kp_csv = tmp_path / "cap" / "keypoint.csv"
    hist_csv = tmp_path / "cap" / "point_history.csv"
    gg = _gg(
        monkeypatch, tmp_path, kp_result=mpg.POINT, ph_result=0,
        collection={"dir": str(tmp_path / "cap")},
    )
    res = gg.update("Right", HAND)
    gg.collect(1, 7, res)
    assert kp_csv.exists()
    assert len(list(csv.reader(kp_csv.open()))) == 1
    assert len(res["landmark_features"]) == 42
    res2 = gg.update("Right", HAND)
    assert res2["history_features"] is not None
    gg.collect(2, 7, res2)
    assert hist_csv.exists()
    rows = list(csv.reader(hist_csv.open()))
    assert rows[0][0] == "7"
    assert len(rows[0]) == 1 + len(res2["history_features"])


def test_game_gestures_unavailable_models_is_safe(monkeypatch, tmp_path):
    gg = _gg(monkeypatch, tmp_path, kp_result=0, ph_result=0)
    gg.cfg["models_dir"] = str(tmp_path / "missing-dir")
    assert gg.update("Left", HAND) is None
    assert gg.aim("Left", None) is False
    assert gg.taps(None, 0.0) == {"sign": None, "motion": None}
    assert gg.models_ok is False


def test_game_gestures_reset_clears_state(monkeypatch, tmp_path):
    gg = _gg(monkeypatch, tmp_path, kp_result=mpg.POINT, ph_result=0)
    res = gg.update("Right", HAND)
    gg.taps(res, 0.0)
    gg.reset()
    assert gg.recognizers == {} and gg.aim_gates == {}
    assert gg.sign_gate._count == 0 and gg.motion_gate._count == 0
