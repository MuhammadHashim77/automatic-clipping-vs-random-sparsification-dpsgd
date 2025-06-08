import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dp_train import run_comparison_experiment
from src.dataset import PneumoDataset
from torch.utils.data import Subset

def run_dp_smoke_test():
    # Create a small subset of the dataset for smoke testing
    full_train_ds = PneumoDataset("data/train-rle.csv", "data/images", "train")
    full_val_ds = PneumoDataset("data/train-rle.csv", "data/images", "val")
    
    # Take only 100 samples for training and 20 for validation
    train_subset = Subset(full_train_ds, range(100))
    val_subset = Subset(full_val_ds, range(20))
    
    # Run comparison with minimal parameters for smoke testing
    run_comparison_experiment(
        batch_size=8,
        epochs=2,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0,
        target_delta=1e-5,
        train_dataset=train_subset,
        val_dataset=val_subset,
        num_workers=2
    )

if __name__ == "__main__":
    run_dp_smoke_test() 