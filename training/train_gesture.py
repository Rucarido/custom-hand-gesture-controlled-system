"""BlazeHand gesture classifier training (HaGRID subset, RTX 4050 6GB).

What it trains
--------------
A crop-friendly image classifier that maps a hand photo -> one of the
HaGRID gesture labels. BlazeHand later feeds it MediaPipe hand crops
(crop-and-classify beats full-frame classification on cluttered
backgrounds) and maps labels -> game poses via BLAZEHAND_MAP below.

Dataset layout expected (--data):
    data/hagrid-150k/<class>/*.jpeg      (this repo: 19 classes, ~150k imgs)

Optimised for a 6 GB laptop GPU (RTX 4050, CUDA 12.6):
  * MobileNetV3-Small backbone (ImageNet pre-trained, ~2.5M params)
  * Mixed precision (autocast + GradScaler), channels-last memory format
  * cudnn.benchmark (fixed input size), pinned memory, multi-worker load
  * WeightedRandomSampler (no_gesture is ~4x the other classes)
  * AdamW + OneCycleLR + label smoothing; early stop on val macro-F1
  * Stratified 90/10 split, per-class P/R/F1 + confusion matrix reports
  * Best-checkpoint resume (run in chunks), ONNX export for inference

Usage (PowerShell, from repo root):
  # 1) 60-second smoke test (validates code + GPU path)
  python training/train_gesture.py --data data/hagrid-150k --smoke
  # 2) full run, e.g. 5 epochs at a time (Ctrl+C safe, --resume continues)
  python training/train_gesture.py --data data/hagrid-150k --epochs 5 --out-dir models/blazehand
  python training/train_gesture.py --data data/hagrid-150k --epochs 10 --resume models/blazehand/last.pt

Outputs in --out-dir: best.pt, last.pt, training_log.csv, metrics.txt,
confusion_matrix.npz, labels.json (+ .onnx export, see --export-onnx).

CC BY-SA 4.0: training data is HaGRID (Kapitanov et al., WACV 2024) via the
cj-mills 512p classification mirror. If you share the model, credit HaGRID.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2 as T

# --------------------------------------------------------------------------
# Label mapping: HaGRID class -> BlazeHand game pose.
# four/palm/stop all read as "4 extended non-thumb fingers" downstream;
# two_up joins peace as the 2-finger bucket. Everything else is kept as its
# own training class (richer features, future gestures) and maps to "none"
# (no game action) at inference time.
# --------------------------------------------------------------------------
BLAZEHAND_MAP = {
    "fist": "fist",
    "one": "index_only",
    "peace": "peace",
    "two_up": "peace",
    "three": "three",
    "three2": "three",
    "four": "four",
    "palm": "four",
    "stop": "four",
}

BACKBONES = ("mobilenet_v3_small", "resnet18")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class ImageFolderList(Dataset):
    """Minimal ImageFolder over an explicit (path, label_idx) list."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def scan_dataset(root: Path):
    """Return (classes, [(path, label_idx)]) sorted for stable indices."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    exts = (".jpg", ".jpeg", ".png", ".webp")
    samples, counts = [], Counter()
    for idx, cls in enumerate(classes):
        for p in sorted((root / cls).iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                samples.append((str(p), idx))
                counts[cls] += 1
    return classes, samples, counts


def stratified_split(samples, n_classes, val_frac=0.10, seed=7):
    """Deterministic per-class shuffle split -> (train, val) sample lists."""
    by_cls = [[] for _ in range(n_classes)]
    for s in samples:
        by_cls[s[1]].append(s)
    rng = random.Random(seed)
    train, val = [], []
    for items in by_cls:
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_transforms(img_size: int, aug_strength: str = "medium"):
    """Train: crop/flip/photometric jitter. Val: deterministic center crop."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if aug_strength == "strong":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.TrivialAugmentWide(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            *to_tensor,
            normalize,
        ])
    else:  # medium: safe for hands (no vertical flip / rotation that breaks L/R)
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.RandomErasing(p=0.15, scale=(0.02, 0.12)),
            *to_tensor,
            normalize,
        ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.14), antialias=True),
        T.CenterCrop(img_size),
        *to_tensor,
        normalize,
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(backbone: str, num_classes: int, dropout: float = 0.2):
    if backbone == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_f, num_classes)
        )
    elif backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"unknown backbone {backbone!r} (pick {BACKBONES})")
    return net.to(memory_format=torch.channels_last)


# --------------------------------------------------------------------------
# Metrics (pure torch/numpy, no sklearn needed)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, device, num_classes):
    net.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        pred = net(x).argmax(1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            conf[t, p] += 1
    conf_np = conf.numpy()
    tp = np.diag(conf_np).astype(float)
    pred_pos = conf_np.sum(axis=0).astype(float)
    actual = conf_np.sum(axis=1).astype(float)
    precision = np.zeros_like(tp)
    recall = np.zeros_like(tp)
    f1 = np.zeros_like(tp)
    np.divide(tp, pred_pos, out=precision, where=pred_pos > 0)
    np.divide(tp, actual, out=recall, where=actual > 0)
    denom = precision + recall
    np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)
    acc = tp.sum() / max(conf_np.sum(), 1)
    return acc, float(f1.mean()), precision, recall, f1, conf_np


def write_report(path, classes, acc, macro_f1, precision, recall, f1, counts):
    lines = [f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}", "",
             f"{'class':18s} {'n':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s}  blazehand"]
    for i, c in enumerate(classes):
        lines.append(f"{c:18s} {counts.get(c, 0):7d} {precision[i]:7.3f} "
                     f"{recall[i]:7.3f} {f1[i]:7.3f}  {BLAZEHAND_MAP.get(c, 'none')}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train BlazeHand gesture classifier")
    p.add_argument("--data", type=Path, default=None, help="ImageFolder root (required unless --export-only)")
    p.add_argument("--out-dir", type=Path, default=Path("models/blazehand"))
    p.add_argument("--backbone", default="mobilenet_v3_small", choices=BACKBONES)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="OneCycle max LR")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--aug", default="medium", choices=("medium", "strong"))
    p.add_argument("--resume", type=Path, default=None, help="last.pt to continue")
    p.add_argument("--export-onnx", action="store_true",
                   help="export best.pt -> blazehand_gesture.onnx at the end")
    p.add_argument("--export-only", action="store_true",
                   help="skip training; only export best.pt -> ONNX")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--smoke", action="store_true",
                   help="tiny 2-class CPU/GPU sanity run (ignores most flags)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> safe autotune
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[train] device={device} amp={use_amp} "
          f"torch={torch.__version__} backbone={args.backbone}")

    if args.export_only:
        best = args.resume if (args.resume and args.resume.exists()) \
            else out / "best.pt"
        probe = torch.load(best, map_location="cpu", weights_only=False)
        export_onnx(best, out / "blazehand_gesture.onnx",
                    device, len(probe["classes"]), args)
        return out

    assert args.data is not None, "--data is required for training"

    classes, all_samples, counts = scan_dataset(args.data)
    print(f"[train] {len(all_samples)} images, {len(classes)} classes:")
    for c in classes:
        print(f"          {c:18s} {counts[c]:6d} -> {BLAZEHAND_MAP.get(c, 'none')}")
    if args.smoke:
        keep = set(classes[:2])
        all_samples = [s for s in all_samples if classes[s[1]] in keep]
        classes = sorted(keep)
        remap = {c: i for i, c in enumerate(classes)}
        all_samples = [(p, remap[classes[old]]) for p, old in all_samples]
        args.epochs, args.batch_size, args.num_workers = 1, 32, 0
        print(f"[train] SMOKE: 2 classes, {len(all_samples)} images, 1 epoch")

    train_samples, val_samples = stratified_split(
        all_samples, len(classes), val_frac=args.val_frac, seed=args.seed)
    print(f"[train] split: {len(train_samples)} train / {len(val_samples)} val")

    train_tf, val_tf = build_transforms(args.img_size, args.aug)
    train_ds = ImageFolderList(train_samples, train_tf)
    val_ds = ImageFolderList(val_samples, val_tf)

    # Weighted sampler counters the ~4x no_gesture majority.
    train_labels = [lb for _, lb in train_samples]
    cls_counts = Counter(train_labels)
    w = [1.0 / cls_counts[lb] for lb in train_labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, **loader_kw)

    net = build_model(args.backbone, len(classes), args.dropout).to(device)
    tuned_params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tuned_params, lr=args.lr / 25,
                            weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.1,
        div_factor=25.0, final_div_factor=1e4)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_f1, global_step = 0, -1.0, 0
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        global_step = ckpt.get("global_step", 0)
        print(f"[train] resumed {args.resume} -> epoch {start_epoch} "
              f"(best_f1={best_f1:.4f})")

    log_path = out / "training_log.csv"
    log_exists = log_path.exists() and start_epoch > 0
    log_f = open(log_path, "a" if log_exists else "w", newline="")
    log_w = csv.writer(log_f)
    if not log_exists:
        log_w.writerow(["epoch", "train_loss", "val_acc", "val_macro_f1",
                        "lr", "secs"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        net.train()
        run_loss, run_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            run_loss += loss.item() * len(y)
            run_n += len(y)
        train_loss = run_loss / max(run_n, 1)

        acc, macro_f1, prec, rec, f1, conf = evaluate(
            net, val_loader, device, len(classes))
        secs = time.perf_counter() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"[train] epoch {epoch:3d}/{args.epochs} "
              f"loss={train_loss:.4f} acc={acc:.4f} macro_f1={macro_f1:.4f} "
              f"lr={lr_now:.2e} ({secs:.0f}s)")
        log_w.writerow([epoch, f"{train_loss:.4f}", f"{acc:.4f}",
                        f"{macro_f1:.4f}", f"{lr_now:.3e}", f"{secs:.0f}"])
        log_f.flush()

        torch.save({"epoch": epoch, "model": net.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_f1": best_f1,
                    "global_step": global_step, "classes": classes,
                    "args": vars(args)}, out / "last.pt")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"epoch": epoch, "model": net.state_dict(),
                        "classes": classes, "val_macro_f1": macro_f1,
                        "backbone": args.backbone, "img_size": args.img_size},
                       out / "best.pt")
            write_report(out / "metrics.txt", classes, acc, macro_f1,
                         prec, rec, f1, counts)
            np.savez(out / "confusion_matrix.npz", matrix=conf,
                     classes=np.array(classes))
            print(f"[train] *** new best macro_f1={macro_f1:.4f} -> best.pt")
    log_f.close()

    labels = {
        "classes": classes,
        "blazehand_map": {c: BLAZEHAND_MAP.get(c, "none") for c in classes},
        "backbone": args.backbone,
        "img_size": args.img_size,
        "license": "CC BY-SA 4.0 (HaGRID, Kapitanov et al., WACV 2024)",
    }
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"[train] done. best macro_f1={best_f1:.4f} -> {out / 'best.pt'}")

    if args.export_onnx:
        export_onnx(out / "best.pt", out / "blazehand_gesture.onnx",
                    device, len(classes), args)
    return out


def export_onnx(best_pt, onnx_path, device, num_classes, args):
    """Export best checkpoint to ONNX (dynamic batch) for game inference.

    Needs an unlocked onnx install; on locked-down Windows (App Control
    policy blocks onnx's native DLL) this raises ImportError — the plain
    PyTorch checkpoint (best.pt + labels.json) remains fully usable.
    """
    try:
        import onnx  # noqa: F401  (imported for its side effects / version check)
    except ImportError as e:
        raise SystemExit(
            "[train] ONNX export unavailable here: the `onnx` package "
            f"cannot load ({e}). best.pt + labels.json are complete and "
            "enough for inference with PyTorch; export the .onnx on an "
            "unrestricted machine if you need it.") from e
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    net = build_model(ckpt.get("backbone", args.backbone), len(classes))
    net.load_state_dict(ckpt["model"])
    net.eval()
    dummy = torch.randn(1, 3, ckpt.get("img_size", args.img_size),
                        ckpt.get("img_size", args.img_size))
    torch.onnx.export(
        net, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True)
    print(f"[train] ONNX exported -> {onnx_path}")


if __name__ == "__main__":
    main()
