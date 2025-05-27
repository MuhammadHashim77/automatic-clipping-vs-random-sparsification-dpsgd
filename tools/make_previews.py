from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import dcm_to_uint8, rle_decode, save_preview   # ← helpers

DATA_DIR    = Path("data")
IMG_DIR     = DATA_DIR / "images"          # flat folder
PREVIEW_DIR = DATA_DIR / "masks-preview"
PREVIEW_DIR.mkdir(exist_ok=True)

# df = pd.read_csv(DATA_DIR / "train-rle.csv")
df = pd.read_csv(DATA_DIR / "train-rle.csv", skipinitialspace=True)  # skip leading spaces in column names

def main(n=20, seed=0):
    rows = df[df["EncodedPixels"] != "-1"].sample(n, random_state=seed)
    for _, row in rows.iterrows():
        uid, rle = row["ImageId"], row["EncodedPixels"]

        img  = dcm_to_uint8(IMG_DIR / f"{uid}.dcm")
        mask = rle_decode(rle, shape=img.shape)

        save_preview(img, mask, PREVIEW_DIR / f"{uid}.png")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.n, args.seed)
