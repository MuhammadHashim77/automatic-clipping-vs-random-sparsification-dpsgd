import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

from .models import build_unet
from .dataset import PneumoDataset
from .metrics import dice

class DPLitSeg(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        use_automatic_clipping: bool = True,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        batch_size: int = 8,
        epochs: int = 30,
        dataset_size: int = 1000,
        num_groups: int = 32
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model with GroupNorm
        self.net = build_unet(num_groups=num_groups)
        
        # Validate model for DP
        self.net = ModuleValidator.fix(self.net)
        errors = ModuleValidator.validate(self.net, strict=True)
        if len(errors) > 0:
            raise ValueError(f"Model validation failed: {errors}")
            
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Privacy parameters
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.use_automatic_clipping = use_automatic_clipping
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        # For privacy accounting
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_size = dataset_size
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.privacy_metrics = []

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage: str):
        img, mask = batch
        logits = self(img)
        loss = self.loss_fn(logits, mask)
        d = dice(logits, mask)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_dice", d, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def on_train_epoch_end(self):
        # Record metrics at the end of each epoch
        metrics = {
            'train_loss': self.trainer.callback_metrics['train_loss'].item(),
            'train_dice': self.trainer.callback_metrics['train_dice'].item(),
            'val_loss': self.trainer.callback_metrics['val_loss'].item(),
            'val_dice': self.trainer.callback_metrics['val_dice'].item(),
            'epoch': self.current_epoch
        }
        self.train_metrics.append(metrics)

def run_dp_training(
    batch_size: int = 8,
    epochs: int = 30,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    use_automatic_clipping: bool = True,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    train_dataset=None,
    val_dataset=None,
    num_workers: int = 4,
    output_dir: str = "outputs/dp_training",
    num_groups: int = 32
):
    if train_dataset is None:
        train_dataset = PneumoDataset("data/train-rle.csv", "data/images", "train")
    if val_dataset is None:
        val_dataset = PneumoDataset("data/train-rle.csv", "data/images", "val")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = DPLitSeg(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        use_automatic_clipping=use_automatic_clipping,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        batch_size=batch_size,
        epochs=epochs,
        dataset_size=len(train_dataset),
        num_groups=num_groups
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Initialize privacy engine
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=len(train_dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        automatic_clipping=use_automatic_clipping
    )
    privacy_engine.attach(model)

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_path),
        log_every_n_steps=10,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save metrics
    metrics = {
        'train_metrics': model.train_metrics,
        'val_metrics': model.val_metrics,
        'privacy_metrics': model.privacy_metrics
    }
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    return model, metrics

def plot_metrics(metrics: Dict, output_dir: str):
    """Plot training metrics and privacy loss."""
    output_path = Path(output_dir)
    
    # Plot accuracy metrics
    plt.figure(figsize=(10, 6))
    epochs = [m['epoch'] for m in metrics['train_metrics']]
    train_dice = [m['train_dice'] for m in metrics['train_metrics']]
    val_dice = [m['val_dice'] for m in metrics['train_metrics']]
    
    plt.plot(epochs, train_dice, label='Train Dice')
    plt.plot(epochs, val_dice, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / 'accuracy_metrics.png')
    plt.close()

    # Plot privacy loss
    if metrics['privacy_metrics']:
        plt.figure(figsize=(10, 6))
        epsilons = [m['epsilon'] for m in metrics['privacy_metrics']]
        plt.plot(epochs, epsilons, label='Privacy Loss (ε)')
        plt.xlabel('Epoch')
        plt.ylabel('Privacy Loss (ε)')
        plt.title('Privacy Loss Over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'privacy_loss.png')
        plt.close()

def run_comparison_experiment(
    batch_size: int = 8,
    epochs: int = 30,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    train_dataset=None,
    val_dataset=None,
    num_workers: int = 4,
    num_groups: int = 32
):
    """Run comparison between RS and Automatic Clipping."""
    # Run with Random Sparsification
    rs_model, rs_metrics = run_dp_training(
        batch_size=batch_size,
        epochs=epochs,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        use_automatic_clipping=False,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=num_workers,
        output_dir="outputs/rs_training",
        num_groups=num_groups
    )

    # Run with Automatic Clipping
    ac_model, ac_metrics = run_dp_training(
        batch_size=batch_size,
        epochs=epochs,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        use_automatic_clipping=True,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=num_workers,
        output_dir="outputs/automatic_clipping",
        num_groups=num_groups
    )

    # Plot comparison
    plot_comparison(rs_metrics, ac_metrics, "outputs/comparison")

def plot_comparison(rs_metrics: Dict, ac_metrics: Dict, output_dir: str):
    """Plot comparison between RS and Automatic Clipping."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    epochs = [m['epoch'] for m in rs_metrics['train_metrics']]
    
    plt.plot(epochs, [m['val_dice'] for m in rs_metrics['train_metrics']], 
             label='RS Validation Dice', linestyle='--')
    plt.plot(epochs, [m['val_dice'] for m in ac_metrics['train_metrics']], 
             label='AC Validation Dice', linestyle='-')
    
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice Coefficient: RS vs Automatic Clipping')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / 'accuracy_comparison.png')
    plt.close()

    # Plot privacy loss comparison
    if rs_metrics['privacy_metrics'] and ac_metrics['privacy_metrics']:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, [m['epsilon'] for m in rs_metrics['privacy_metrics']], 
                 label='RS Privacy Loss', linestyle='--')
        plt.plot(epochs, [m['epsilon'] for m in ac_metrics['privacy_metrics']], 
                 label='AC Privacy Loss', linestyle='-')
        
        plt.xlabel('Epoch')
        plt.ylabel('Privacy Loss (ε)')
        plt.title('Privacy Loss: RS vs Automatic Clipping')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / 'privacy_comparison.png')
        plt.close() 