import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

from .models import build_unet
from .dataset import PneumoDataset
from .metrics import dice

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    total_loss = 0
    total_dice = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        img, mask = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice += dice(logits, mask).item()
    
    return {
        'loss': total_loss / len(train_loader),
        'dice': total_dice / len(train_loader)
    }

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            img, mask = [b.to(device) for b in batch]
            logits = model(img)
            loss = loss_fn(logits, mask)
            
            total_loss += loss.item()
            total_dice += dice(logits, mask).item()
    
    return {
        'loss': total_loss / len(val_loader),
        'dice': total_dice / len(val_loader)
    }

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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = build_unet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()

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

    # Initialize optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )

    # Initialize privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        clipping="per_layer"
    )

    # Training loop
    metrics = {
        'train_metrics': [],
        'val_metrics': [],
        'privacy_metrics': []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Record metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_dice': train_metrics['dice'],
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice']
        }
        metrics['train_metrics'].append(epoch_metrics)
        
        # Print progress
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")

    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    return model, metrics

def plot_metrics(metrics: Dict, output_dir: str):
    """Plot training metrics."""
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