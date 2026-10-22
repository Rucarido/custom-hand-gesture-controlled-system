"""Extract the HaGRID zip into data/hagrid-150k/<class>/*.jpeg (strip top dir)."""
import zipfile
from pathlib import Path

from tqdm import tqdm

SRC = Path("data/hagrid-150k.zip")
DST = Path("data/hagrid-150k")
DST.mkdir(parents=True, exist_ok=True)

z = zipfile.ZipFile(SRC)
members = [m for m in z.namelist() if not m.endswith("/")]
print(f"extracting {len(members)} files -> {DST}")
for m in tqdm(members, unit="img"):
    parts = m.split("/", 1)
    rel = parts[1] if len(parts) == 2 else parts[0]
    out = DST / rel
    if out.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    with z.open(m) as src, open(out, "wb") as fh:
        fh.write(src.read())

# summary
for d in sorted(p for p in DST.iterdir() if p.is_dir()):
    n = sum(1 for _ in d.iterdir())
    print(f"{d.name:18s} {n}")
