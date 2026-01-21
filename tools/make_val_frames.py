# tools/make_val_frames.py
import os, shutil, random
from pathlib import Path

# Source folders (adjust if your paths differ)
REAL_SRC = Path("data/real_frames")
FAKE_SRC = Path("data/fake_frames")

OUT = Path("data/val_frames")
OUT.mkdir(parents=True, exist_ok=True)

# number to sample per class (adjust; 50-200 recommended)
N_PER_CLASS = 100

def sample_and_copy(src, out, n):
    if not src.exists():
        print("Source does not exist:", src)
        return 0
    files = sorted([p for p in src.glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
    if not files:
        print("No files in", src)
        return 0
    if len(files) <= n:
        chosen = files
    else:
        chosen = random.sample(files, n)
    for p in chosen:
        dest = out / p.name
        shutil.copy2(str(p), str(dest))
    return len(chosen)

random.seed(42)
r = sample_and_copy(REAL_SRC, OUT, N_PER_CLASS)
f = sample_and_copy(FAKE_SRC, OUT, N_PER_CLASS)
print(f"Copied {r} real and {f} fake frames to {OUT} (total {r+f})")
