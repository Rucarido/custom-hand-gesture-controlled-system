"""One-time shrink of the HaGRID subset to max-side 256px JPEGs.

Why: 512px JPEG decode dominates training time on CPU. At 256px the
RandomResizedCrop(224) training pipeline loses nothing measurable
(standard ImageNet-style scale), but decode gets ~4x faster and the
working set shrinks, so every epoch (and every future experiment) runs
much faster on the RTX 4050.

Input : data/hagrid-150k/<class>/*.jpeg   (~4 GB, 150k files)
Output: data/hagrid-150k-256/<class>/*.jpg (same layout, JPEG q92)
Skips files already present -> re-runnable / resumable.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SRC = Path("data/hagrid-150k")
DST = Path("data/hagrid-150k-256")
MAX_SIDE = 256
QUALITY = 92
WORKERS = max(2, (os.cpu_count() or 8) - 2)


def job(args):
    src, dst = args
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=QUALITY)
        return True
    except Exception as e:  # keep going; report at the end
        return (str(src), str(e))


def main():
    tasks, skipped = [], 0
    for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        for src in sorted(cls_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            dst = DST / cls_dir.name / (src.stem + ".jpg")
            if dst.exists():
                skipped += 1
            else:
                tasks.append((src, dst))
    print(f"to convert: {len(tasks)} (skipped present: {skipped}), "
          f"workers={WORKERS}")
    bad = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in tqdm(ex.map(job, tasks, chunksize=50), total=len(tasks),
                      unit="img"):
            if r is not True:
                bad.append(r)
    print(f"done. failures: {len(bad)}")
    for b in bad[:10]:
        print("  FAILED:", b)


if __name__ == "__main__":
    main()
