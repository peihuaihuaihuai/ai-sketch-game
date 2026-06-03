"""
Training script for the QuickDraw sketch classifier (Phase 2).

Trains a ResNet-based model on the official Google QuickDraw dataset.
Supports automatic data downloading, GPU training with mixed precision,
data augmentation, early stopping, and saves the best model checkpoint.

Usage:
    D:/python3.10/python.exe model/train.py

Output:
    - model/quickdraw_cnn.pth  (best model weights)
    - model/training_log.txt   (per-epoch training log)
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from model.model import create_model, get_model_summary, QuickDrawResNet
from model.dataset import (
    QuickDrawDataset,
    load_quickdraw_data,
    generate_synthetic_data,
    DEFAULT_CLASS_NAMES,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASS_NAMES = DEFAULT_CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)

HYPERPARAMS = {
    "model": "resnet",               # 'resnet' or 'cnn'
    "epochs": 25,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "train_split": 0.80,             # 80% train, 20% validation
    "max_samples_per_class": 20000,  # Cap per class for speed
    "early_stop_patience": 5,        # Stop if val acc doesn't improve
    "num_workers": 0,                # DataLoader workers (0 for Windows safety)
    "mixed_precision": True,         # Use torch.cuda.amp if available
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Mixed precision scaler (only used if CUDA available)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
if scaler:
    print("Mixed precision training enabled (AMP)")


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Validate for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Simple training logger that writes to file and prints to console."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lines = []

    def log(self, message: str) -> None:
        print(message)
        self.lines.append(message)

    def save(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    """Main training entry point."""
    logger = TrainingLogger(MODEL_DIR / "training_log.txt")

    logger.log("=" * 60)
    logger.log("QuickDraw AI Training - Phase 2")
    logger.log("=" * 60)
    logger.log(f"Model: {HYPERPARAMS['model']}")
    logger.log(f"Classes: {', '.join(CLASS_NAMES)}")
    logger.log(f"Device: {device}")
    logger.log("")
    data_source = "Unknown"

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    images, labels = load_quickdraw_data(
        CLASS_NAMES,
        DATA_DIR,
        max_samples_per_class=HYPERPARAMS["max_samples_per_class"],
        download=True,
    )

    logger.log("Data source: QuickDraw (real)")
    logger.log(f"Total samples: {len(images):,}")

    # -----------------------------------------------------------------------
    # Create datasets and dataloaders
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Train/validation split at numpy level (prevents data leakage)
    # -----------------------------------------------------------------------

    total_samples = len(images)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(HYPERPARAMS["train_split"] * total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Verify zero overlap between train and validation sets
    overlap = set(train_indices.tolist()) & set(val_indices.tolist())
    assert len(overlap) == 0, f"CRITICAL: train/val overlap detected: {len(overlap)} samples"

    # Split images and labels into disjoint sets
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]

    # Create separate datasets from disjoint data
    train_dataset = QuickDrawDataset(train_images, train_labels, augment=True)
    val_dataset = QuickDrawDataset(val_images, val_labels, augment=False)

    # Optimized DataLoader settings
    pin_memory = device.type == "cuda"
    persistent_workers = HYPERPARAMS["num_workers"] > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        num_workers=HYPERPARAMS["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
        num_workers=HYPERPARAMS["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    logger.log(f"Train samples: {len(train_dataset):,}")
    logger.log(f"Val samples:   {len(val_dataset):,}")
    logger.log(f"Batch size:    {HYPERPARAMS['batch_size']}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Initialize model
    # -----------------------------------------------------------------------

    model = create_model(
        HYPERPARAMS["model"],
        num_classes=NUM_CLASSES,
    ).to(device)

    stats = get_model_summary(model)
    logger.log(f"Model: {HYPERPARAMS['model']}")
    logger.log(f"Parameters: {stats['total_params']:,}")
    logger.log(f"Size: ~{stats['size_mb']} MB")
    logger.log("")

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"],
    )

    # Cosine annealing with warm restarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,          # First restart at epoch 10
        T_mult=2,        # Double the restart interval each time
        eta_min=1e-6,    # Minimum learning rate
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    logger.log("=" * 70)
    logger.log(
        f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} "
        f"{'Val Loss':<12} {'Val Acc':<12} {'LR':<12} {'Time':<8}"
    )
    logger.log("=" * 70)

    start_time_total = time.time()

    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        # Update scheduler
        scheduler.step(epoch)

        epoch_time = time.time() - epoch_start

        log_line = (
            f"{epoch:<6} "
            f"{train_loss:<12.4f} "
            f"{train_acc:<12.4f} "
            f"{val_loss:<12.4f} "
            f"{val_acc:<12.4f} "
            f"{current_lr:<12.6f} "
            f"{epoch_time:<8.1f}s"
        )
        logger.log(log_line)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time": epoch_time,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_path = MODEL_DIR / "quickdraw_cnn.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
                "model_name": HYPERPARAMS["model"],
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
            }, str(save_path))
            logger.log(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= HYPERPARAMS["early_stop_patience"]:
            logger.log(
                f"\nEarly stopping triggered "
                f"(no improvement for {HYPERPARAMS['early_stop_patience']} epochs)"
            )
            break

    total_time = time.time() - start_time_total

    logger.log("=" * 70)
    logger.log("")
    logger.log(f"Training complete in {total_time:.1f}s!")
    logger.log(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.log(f"Model saved to: {MODEL_DIR / 'quickdraw_cnn.pth'}")
    logger.log("")

    # -----------------------------------------------------------------------
    # Save detailed training log
    # -----------------------------------------------------------------------

    logger.log("=" * 70)
    logger.log("Training Summary")
    logger.log("=" * 70)
    logger.log(f"Data source:        {data_source}")
    logger.log(f"Classes:            {', '.join(CLASS_NAMES)}")
    logger.log(f"Total samples:      {len(images):,}")
    logger.log(f"Train/Val split:    {HYPERPARAMS['train_split']*100:.0f}/{100 - HYPERPARAMS['train_split']*100:.0f}")
    logger.log(f"Model architecture: {HYPERPARAMS['model']}")
    logger.log(f"Parameters:         {stats['total_params']:,}")
    logger.log(f"Epochs trained:     {epoch}")
    logger.log(f"Batch size:         {HYPERPARAMS['batch_size']}")
    logger.log(f"Initial LR:         {HYPERPARAMS['learning_rate']}")
    logger.log(f"Weight decay:       {HYPERPARAMS['weight_decay']}")
    logger.log(f"Device:             {device}")
    logger.log(f"Mixed precision:    {scaler is not None}")
    logger.log("")
    logger.log(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.log(f"Final model path: {MODEL_DIR / 'quickdraw_cnn.pth'}")
    logger.log("")
    logger.log("Per-epoch history:")
    logger.log("-" * 70)
    logger.log(
        f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} "
        f"{'Val Loss':<12} {'Val Acc':<12} {'LR':<12}"
    )
    for h in history:
        logger.log(
            f"{h['epoch']:<6} "
            f"{h['train_loss']:<12.4f} "
            f"{h['train_acc']:<12.4f} "
            f"{h['val_loss']:<12.4f} "
            f"{h['val_acc']:<12.4f} "
            f"{h['lr']:<12.6f}"
        )

    logger.save()
    print(f"\nTraining log saved to: {MODEL_DIR / 'training_log.txt'}")


if __name__ == "__main__":
    main()
