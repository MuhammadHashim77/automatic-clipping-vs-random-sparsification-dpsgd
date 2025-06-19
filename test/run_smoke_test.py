import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.train import run_baseline
from src.dataset import PneumoDataset
import torch
from torch.utils.data import Subset

def run_fast_smoke_test():
    # Create a small subset of the dataset for smoke testing
    full_train_ds = PneumoDataset("data/train-rle.csv", "data/images", "train")
    full_val_ds = PneumoDataset("data/train-rle.csv", "data/images", "val")
    
    # Take only 50 samples for training and 10 for validation
    train_subset = Subset(full_train_ds, range(50))
    val_subset = Subset(full_val_ds, range(10))
    
    # Run with minimal parameters for smoke testing
    run_baseline(
        batch_size=8,
        epochs=2,
        train_dataset=train_subset,
        val_dataset=val_subset,
        num_workers=2  # Use fewer workers for smoke testing
    )

if __name__ == "__main__":
    run_fast_smoke_test()
