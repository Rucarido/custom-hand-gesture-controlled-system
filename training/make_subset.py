"""Build a tiny stratified course-project subset from the local HaGRID copy.

Samples N images per class (default 150) from data/hagrid-150k (512px
originals — at this scale decode cost is trivial and full resolution
keeps the most detail) into data/hagrid-mini/<class>/.

Deterministic (seeded) so the split is reproducible for a report.
"""

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/hagrid-150k"))
    p.add_argument("--dst", type=Path, default=Path("data/hagrid-mini"))
    p.add_argument("--per-class", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for cls_dir in sorted(x for x in args.src.iterdir() if x.is_dir()):
        files = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
        )
        n = min(args.per_class, len(files))
        picked = rng.sample(files, n)
        out_dir = args.dst / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(picked, desc=cls_dir.name, unit="img", leave=False):
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        total += n
        print(f"{cls_dir.name:18s} {n:5d}/{len(files)}")
    print(f"subset total: {total} images -> {args.dst}")


if __name__ == "__main__":
    main()
