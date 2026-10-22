"""Trained gesture classifier — BlazeHand model wired into the game loop.

Wraps `models/blazehand-mini/best.pt` (MobileNetV3-Small fine-tuned on the
HaGRID subset, see training/train_gesture.py + README §8) so battle mode
can ask "is the hand pointing?" with learned appearance instead of
fragile geometric ratios.

Design notes:

* FULL-FFRAME input, not a hand crop: the model was trained on full
  HaGRID frames (person + background + gesturing hand), so feeding the
  whole webcam frame matches the training distribution. A tight hand
  crop would be out-of-distribution.
* Preprocessing mirrors the training val transform exactly: resize the
  shorter side to img_size*1.14, center-crop img_size, ImageNet normalize.
* Lazy torch import: if torch/torchvision or the checkpoint are missing
  (or CUDA is down), `available=False` and the caller falls back to the
  rule-based path — the game must never die over an optional model.
* Model head must stay in sync with training/train_gesture.py
  build_model(): Dropout + Linear on MobileNetV3-Small's classifier[-1].
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ImageNet stats used at train time (build_transforms in training script).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class GestureClassifier:
    """Full-frame gesture classifier backed by the trained best.pt.

    Usage:
        gc = GestureClassifier("models/blazehand-mini/best.pt")
        if gc.available:
            pose, conf, cls = gc.classify(frame_bgr)
            # pose is a BlazeHand pose name via labels.json blazehand_map
    """

    def __init__(self, model_path, conf_min: float = 0.5) -> None:
        self.available = False
        self.pose_by_class: dict[str, str] = {}
        self.classes: list[str] = []
        self.img_size = 224
        self.backbone = "mobilenet_v3_small"
        self.conf_min = conf_min
        self.last = (None, 0.0, None)  # (pose, conf, raw_class) cache

        model_path = Path(model_path)
        if not model_path.exists():
            print(f"[ai] model not found: {model_path} — staying rule-based")
            return
        try:
            import torch
            from torchvision import models as tv_models
            import torch.nn as nn
        except Exception as e:
            print(f"[ai] torch unavailable ({e}) — staying rule-based")
            return

        try:
            ckpt = torch.load(model_path, map_location="cpu",
                              weights_only=False)
            self.classes = list(ckpt["classes"])
            self.img_size = int(ckpt.get("img_size", 224))
            self.backbone = str(ckpt.get("backbone", "mobilenet_v3_small"))

            if self.backbone == "mobilenet_v3_small":
                net = tv_models.mobilenet_v3_small(
                    weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                )
                in_f = net.classifier[-1].in_features
                net.classifier[-1] = nn.Sequential(
                    nn.Dropout(p=0.2), nn.Linear(in_f, len(self.classes))
                )
            else:  # resnet18 (training --backbone resnet18)
                net = tv_models.resnet18(
                    weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
                )
                in_f = net.fc.in_features
                net.fc = nn.Sequential(
                    nn.Dropout(p=0.2), nn.Linear(in_f, len(self.classes))
                )
            net.load_state_dict(ckpt["model"])

            labels_path = model_path.parent / "labels.json"
            if labels_path.exists():
                meta = json.loads(labels_path.read_text(encoding="utf-8"))
                self.pose_by_class = {
                    c: str(p) for c, p in meta.get("blazehand_map", {}).items()
                }

            self._torch = torch
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            net.to(self.device).eval()
            self._net = net
            self.available = True
            print(f"[ai] loaded {model_path.name}: {len(self.classes)} "
                  f"classes, {self.backbone}, {self.img_size}px, "
                  f"device={self.device.type}")
        except Exception as e:
            print(f"[ai] failed to load {model_path} ({e}) — rule-based")
            self.available = False

    # -- preprocessing ----------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> normalized CHW float array (training val transform)."""
        size = self.img_size
        short = int(size * 1.14)
        h, w = frame_bgr.shape[:2]
        if w <= h:  # landscape: width is the short side
            nw, nh = short, max(1, round(h * short / w))
        else:
            nw, nh = max(1, round(w * short / h)), short
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        # center crop to size x size
        x0 = max(0, (nw - size) // 2)
        y0 = max(0, (nh - size) // 2)
        crop = resized[y0:y0 + size, x0:x0 + size]
        if crop.shape[0] < size or crop.shape[1] < size:  # tiny frame guard
            crop = cv2.resize(crop, (size, size))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[..., c] = (rgb[..., c] - _MEAN[c]) / _STD[c]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))  # CHW

    # -- inference ----------------------------------------------------------

    def classify(self, frame_bgr: np.ndarray):
        """Run the model on one full camera frame.

        Returns (blazehand_pose_or_None, confidence, raw_class). pose is
        None when the top class maps to "none"/unmapped or confidence is
        below conf_min — callers treat that as "not a game gesture".
        Caches the result in self.last for HUD display.
        """
        if not self.available:
            return None, 0.0, None
        try:
            torch = self._torch
            x = torch.from_numpy(self._preprocess(frame_bgr))
            x = x.unsqueeze(0).to(self.device, non_blocking=True)
            with torch.inference_mode():
                probs = torch.softmax(self._net(x)[0], dim=0)
            conf, idx = probs.max(dim=0)
            raw = self.classes[int(idx)]
            conf = float(conf)
            pose = self.pose_by_class.get(raw, "none")
            if pose == "none" or conf < self.conf_min:
                pose = None
            self.last = (pose, conf, raw)
            return pose, conf, raw
        except Exception as e:
            print(f"[ai] classify failed ({e}) — one frame skipped")
            return self.last if self.last[0] is not None else (None, 0.0, None)
