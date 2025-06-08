from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .utils import rle_decode
from .preprocessing import MedicalImagePreprocessor


class PneumoDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        split: str = "train",            # "train" | "val" | "test"
        fold_seed: int = 42,
        val_pct: float = 0.15,
        target_size: tuple = (256, 256),
        normalize: bool = True,
        augment: bool = True
    ):
        self.df = pd.read_csv(csv_path, skipinitialspace=True)
        self.df.columns = [c.strip() for c in self.df.columns]   # trim headers
        id_col, rle_col = "ImageId", self.df.columns[-1]

        # Split at patient level to prevent data leakage
        rng = np.random.default_rng(fold_seed)
        uids = self.df[id_col].unique()
        rng.shuffle(uids)
        n = len(uids)
        train_cut = int((1 - 2 * val_pct) * n)
        val_cut   = int((1 - val_pct) * n)
        uid_splits = {
            "train": uids[:train_cut],
            "val":   uids[train_cut:val_cut],
            "test":  uids[val_cut:],
        }[split]

        self.records = (
            self.df[self.df[id_col].isin(uid_splits)]
            .reset_index(drop=True)
            .rename(columns={id_col: "uid", rle_col: "rle"})
        )
        self.img_dir = Path(img_dir)
        
        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor(
            target_size=target_size,
            normalize=normalize,
            augment=augment and split == "train"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        uid, rle = self.records.loc[idx, ["uid", "rle"]]
        
        # Load and preprocess image
        img_path = self.img_dir / f"{uid}.dcm"
        mask = rle_decode(rle, (1024, 1024))  # Original size
        
        # Apply preprocessing pipeline
        img_tensor, mask_tensor = self.preprocessor(
            img_path,
            mask,
            is_training=self.preprocessor.augment
        )
        
        return img_tensor, mask_tensor
