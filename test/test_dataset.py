import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader
from src.dataset import PneumoDataset

ds = PneumoDataset("data/train-rle.csv", "data/images", split="train")
print(len(ds), "samples")
x, y = ds[0]
print("Image tensor", x.shape, x.min(), x.max(), "| mask", y.shape, y.sum())
