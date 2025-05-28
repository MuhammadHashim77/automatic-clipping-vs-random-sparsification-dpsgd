# Lightning module + trainer for Pneumonia segmentation
import torch, lightning as L
from torch.optim import Adam
from torch.utils.data import DataLoader
from .dataset import PneumoDataset
from .models import build_unet
from .metrics import dice

class LitSeg(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_unet()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage: str):
        img, mask = batch
        logits = self(img)
        loss = self.loss_fn(logits, mask)
        d = dice(logits, mask)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_dice", d,   prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def run_baseline(batch_size=8, epochs=30):
    train_ds = PneumoDataset("data/train-rle.csv", "data/images", "train")
    val_ds   = PneumoDataset("data/train-rle.csv", "data/images", "val")

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir="outputs/baseline",
        log_every_n_steps=10,
    )
    model = LitSeg()
    trainer.fit(
        model,
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4),
        DataLoader(val_ds,   batch_size, shuffle=False, num_workers=4),
    )
