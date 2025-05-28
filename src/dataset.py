from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import pandas as pd
from .utils import dcm_to_uint8, rle_decode


class PneumoDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        split: str = "train",            # "train" | "val" | "test"
        transform: A.Compose | None = None,
        fold_seed: int = 42,
        val_pct: float = 0.15,
    ):
        self.df = pd.read_csv(csv_path, skipinitialspace=True)
        self.df.columns = [c.strip() for c in self.df.columns]   # trim headers
        id_col, rle_col = "ImageId", self.df.columns[-1]

        # ── split at *patient* level so CT neighbours don't leak ────────────
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
        self.transform = transform or self._default_augs(split)

    # ------------------------------------------------------------------ #
    def __len__(self):               # batches know how many samples
        return len(self.records)

    def __getitem__(self, idx):      # ← core PyTorch contract
        uid, rle = self.records.loc[idx, ["uid", "rle"]]
        img = dcm_to_uint8(self.img_dir / f"{uid}.dcm")          # (H,W)
        mask = rle_decode(rle, img.shape)                        # (H,W)

        # Albumentations expects (H,W,C); X-ray is single-channel
        img = np.expand_dims(img, axis=-1)
        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented["image"], augmented["mask"]

        # to CHW torch tensors
        img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask

    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_augs(split: str) -> A.Compose:
        if split == "train":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05,
                                       scale_limit=0.1,
                                       rotate_limit=10, p=0.5),
                    A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
                ]
            )
        else:
            return A.Compose([], p=1.0)
