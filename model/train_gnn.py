"""
Training script for the GNN-only sketch classifier (Step 3).

Trains a SketchGNNClassifier on synthetic stroke data.
When real QuickDraw stroke (.ndjson) data is available, it will be used instead.

Usage:
    D:/python3.10/python.exe model/train_gnn.py

Output:
    - model/quickdraw_gnn.pth  (best model weights)
    - model/training_log_gnn.txt
"""

import os
import sys
import time
import math
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from model.stroke_graph import SketchGNNClassifier, build_graph_from_strokes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASS_NAMES = ["airplane", "car", "cat", "dog", "house", "tree"]
NUM_CLASSES = len(CLASS_NAMES)

HYPERPARAMS = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "train_split": 0.80,
    "max_samples_per_class": 5000,
    "early_stop_patience": 7,
    "num_workers": 0,
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.2,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Synthetic Stroke Generation
# ---------------------------------------------------------------------------

def _interpolate_line(x1: float, y1: float, x2: float, y2: float, n_points: int = 10) -> List[Dict[str, float]]:
    """Generate points along a line."""
    return [{"x": x1 + (x2 - x1) * t / max(n_points - 1, 1),
             "y": y1 + (y2 - y1) * t / max(n_points - 1, 1)}
            for t in range(n_points)]


def _generate_circle_points(cx: float, cy: float, r: float, n_points: int = 12) -> List[Dict[str, float]]:
    """Generate points along a circle."""
    return [{"x": cx + r * math.cos(2 * math.pi * t / n_points),
             "y": cy + r * math.sin(2 * math.pi * t / n_points)}
            for t in range(n_points)]


def generate_synthetic_strokes(name: str, rng: random.Random, n_points_per_stroke: int = 10) -> List[List[Dict[str, float]]]:
    """Generate synthetic stroke sequences for a given class."""
    strokes = []
    cx, cy = 0.5, 0.5
    scale = rng.uniform(0.7, 1.0)
    jitter = lambda: rng.uniform(-0.03, 0.03)

    if name == "airplane":
        # Fuselage
        strokes.append(_interpolate_line(cx - 0.3*scale, cy, cx + 0.3*scale, cy, n_points_per_stroke))
        # Wings
        strokes.append(_interpolate_line(cx - 0.1*scale, cy - 0.15*scale, cx + 0.1*scale, cy - 0.15*scale, n_points_per_stroke))
        # Tail
        strokes.append(_interpolate_line(cx + 0.2*scale, cy - 0.05*scale, cx + 0.3*scale, cy - 0.12*scale, n_points_per_stroke))
        strokes.append(_interpolate_line(cx + 0.2*scale, cy + 0.05*scale, cx + 0.3*scale, cy + 0.08*scale, n_points_per_stroke))

    elif name == "car":
        # Body
        body = [
            {"x": cx - 0.25*scale, "y": cy + 0.1*scale},
            {"x": cx + 0.25*scale, "y": cy + 0.1*scale},
            {"x": cx + 0.25*scale, "y": cy - 0.1*scale},
            {"x": cx - 0.25*scale, "y": cy - 0.1*scale},
            {"x": cx - 0.25*scale, "y": cy + 0.1*scale},
        ]
        strokes.append(body)
        # Roof
        strokes.append(_interpolate_line(cx - 0.15*scale, cy - 0.1*scale, cx, cy - 0.22*scale, n_points_per_stroke))
        strokes.append(_interpolate_line(cx, cy - 0.22*scale, cx + 0.18*scale, cy - 0.1*scale, n_points_per_stroke))
        # Wheels
        strokes.append(_generate_circle_points(cx - 0.15*scale, cy + 0.1*scale, 0.06*scale, 8))
        strokes.append(_generate_circle_points(cx + 0.15*scale, cy + 0.1*scale, 0.06*scale, 8))

    elif name == "cat":
        # Head
        strokes.append(_generate_circle_points(cx, cy - 0.1*scale, 0.12*scale, 14))
        # Ears
        strokes.append(_interpolate_line(cx - 0.08*scale, cy - 0.18*scale, cx - 0.03*scale, cy - 0.28*scale, 6))
        strokes.append(_interpolate_line(cx - 0.03*scale, cy - 0.28*scale, cx + 0.02*scale, cy - 0.18*scale, 6))
        strokes.append(_interpolate_line(cx + 0.03*scale, cy - 0.18*scale, cx + 0.08*scale, cy - 0.28*scale, 6))
        strokes.append(_interpolate_line(cx + 0.08*scale, cy - 0.28*scale, cx + 0.13*scale, cy - 0.18*scale, 6))
        # Body
        strokes.append(_interpolate_line(cx, cy - 0.02*scale, cx, cy + 0.18*scale, 8))
        strokes.append(_interpolate_line(cx, cy + 0.12*scale, cx + 0.15*scale, cy + 0.08*scale, 6))
        # Whiskers
        strokes.append(_interpolate_line(cx - 0.1*scale, cy - 0.08*scale, cx - 0.2*scale, cy - 0.1*scale, 4))
        strokes.append(_interpolate_line(cx + 0.1*scale, cy - 0.08*scale, cx + 0.2*scale, cy - 0.1*scale, 4))

    elif name == "dog":
        # Head
        strokes.append(_generate_circle_points(cx, cy - 0.1*scale, 0.12*scale, 14))
        # Ears (floppy)
        strokes.append(_interpolate_line(cx - 0.1*scale, cy - 0.15*scale, cx - 0.18*scale, cy - 0.05*scale, 6))
        strokes.append(_interpolate_line(cx + 0.1*scale, cy - 0.15*scale, cx + 0.18*scale, cy - 0.05*scale, 6))
        # Body
        strokes.append(_interpolate_line(cx, cy - 0.02*scale, cx, cy + 0.18*scale, 8))
        strokes.append(_interpolate_line(cx, cy + 0.12*scale, cx + 0.18*scale, cy + 0.05*scale, 6))
        # Tail
        strokes.append(_interpolate_line(cx + 0.1*scale, cy + 0.15*scale, cx + 0.22*scale, cy + 0.05*scale, 6))
        # Snout
        strokes.append(_interpolate_line(cx - 0.03*scale, cy - 0.02*scale, cx + 0.03*scale, cy - 0.02*scale, 4))
        strokes.append(_interpolate_line(cx, cy - 0.02*scale, cx, cy + 0.03*scale, 4))

    elif name == "house":
        # Walls
        wall = [
            {"x": cx - 0.2*scale, "y": cy + 0.2*scale},
            {"x": cx + 0.2*scale, "y": cy + 0.2*scale},
            {"x": cx + 0.2*scale, "y": cy - 0.1*scale},
            {"x": cx - 0.2*scale, "y": cy - 0.1*scale},
            {"x": cx - 0.2*scale, "y": cy + 0.2*scale},
        ]
        strokes.append(wall)
        # Roof
        strokes.append(_interpolate_line(cx - 0.25*scale, cy - 0.1*scale, cx, cy - 0.3*scale, 8))
        strokes.append(_interpolate_line(cx, cy - 0.3*scale, cx + 0.25*scale, cy - 0.1*scale, 8))
        strokes.append(_interpolate_line(cx - 0.25*scale, cy - 0.1*scale, cx + 0.25*scale, cy - 0.1*scale, 4))
        # Door
        strokes.append(_interpolate_line(cx - 0.05*scale, cy + 0.2*scale, cx - 0.05*scale, cy + 0.05*scale, 5))
        strokes.append(_interpolate_line(cx + 0.05*scale, cy + 0.2*scale, cx + 0.05*scale, cy + 0.05*scale, 5))
        strokes.append(_interpolate_line(cx - 0.05*scale, cy + 0.05*scale, cx + 0.05*scale, cy + 0.05*scale, 4))

    elif name == "tree":
        # Trunk
        strokes.append(_interpolate_line(cx, cy + 0.1*scale, cx, cy + 0.25*scale, 6))
        # Foliage (3 overlapping circles)
        strokes.append(_generate_circle_points(cx, cy - 0.05*scale, 0.1*scale, 10))
        strokes.append(_generate_circle_points(cx - 0.08*scale, cy + 0.02*scale, 0.08*scale, 8))
        strokes.append(_generate_circle_points(cx + 0.08*scale, cy + 0.02*scale, 0.08*scale, 8))

    # Add jitter to all points
    jittered = []
    for stroke in strokes:
        jstroke = []
        for pt in stroke:
            jstroke.append({
                "x": np.clip(pt["x"] + jitter(), 0.0, 1.0),
                "y": np.clip(pt["y"] + jitter(), 0.0, 1.0),
            })
        jittered.append(jstroke)

    return jittered


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StrokeDataset(Dataset):
    """PyTorch Dataset for stroke sequences."""

    def __init__(
        self,
        stroke_samples: List[Tuple[List[List[Dict]], int]],
    ):
        """
        Args:
            stroke_samples: List of (strokes, label) tuples.
        """
        self.samples = stroke_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        strokes, label = self.samples[idx]
        node_feats, edge_index, edge_feats, _ = build_graph_from_strokes(strokes)
        if node_feats is None:
            # Fallback empty graph
            node_feats = torch.zeros(1, 5, dtype=torch.float32)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_feats = torch.zeros(0, 2, dtype=torch.float32)
        return node_feats, edge_index, edge_feats, torch.tensor(label, dtype=torch.long)


def collate_stroke_batch(batch):
    """Collate function for batched graph training."""
    node_feats_list = []
    edge_index_list = []
    edge_feats_list = []
    labels = []
    node_offset = 0

    for node_feats, edge_index, edge_feats, label in batch:
        node_feats_list.append(node_feats)
        labels.append(label)

        if edge_index.size(1) > 0:
            edge_index_list.append(edge_index + node_offset)
            edge_feats_list.append(edge_feats)

        node_offset += node_feats.size(0)

    batch_node_feats = torch.cat(node_feats_list, dim=0)
    batch_labels = torch.stack(labels)

    if edge_index_list:
        batch_edge_index = torch.cat([ei.unsqueeze(0) for ei in edge_index_list], dim=1)
        # Need to stack as (2, total_edges)
        batch_edge_index = torch.cat(edge_index_list, dim=1)
        batch_edge_feats = torch.cat(edge_feats_list, dim=0)
    else:
        batch_edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch_edge_feats = torch.zeros(0, 2, dtype=torch.float32)

    return batch_node_feats, batch_edge_index, batch_edge_feats, batch_labels


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for node_feats, edge_index, edge_feats, labels in loader:
        node_feats = node_feats.to(device)
        edge_index = edge_index.to(device)
        edge_feats = edge_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(node_feats, edge_index, edge_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for node_feats, edge_index, edge_feats, labels in loader:
            node_feats = node_feats.to(device)
            edge_index = edge_index.to(device)
            edge_feats = edge_feats.to(device)
            labels = labels.to(device)

            logits = model(node_feats, edge_index, edge_feats)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GNN Sketch Classifier Training - Step 3")
    print("=" * 60)
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Device: {device}")
    print("")

    # Generate synthetic stroke data
    rng = random.Random(42)
    all_samples = []
    for class_idx, name in enumerate(CLASS_NAMES):
        n_samples = HYPERPARAMS["max_samples_per_class"]
        for _ in range(n_samples):
            strokes = generate_synthetic_strokes(name, rng)
            all_samples.append((strokes, class_idx))

    print(f"Total synthetic stroke samples: {len(all_samples):,}")

    # Train/val split
    indices = list(range(len(all_samples)))
    random.Random(42).shuffle(indices)
    train_size = int(HYPERPARAMS["train_split"] * len(all_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    train_dataset = StrokeDataset(train_samples)
    val_dataset = StrokeDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_stroke_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_stroke_batch,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    print("")

    # Model
    model = SketchGNNClassifier(
        num_classes=NUM_CLASSES,
        hidden_dim=HYPERPARAMS["hidden_dim"],
        num_layers=HYPERPARAMS["num_layers"],
        dropout=HYPERPARAMS["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SketchGNNClassifier")
    print(f"Parameters: {total_params:,}")
    print("")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("=" * 60)

    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, optimizer)
        scheduler.step(epoch)

        print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "hyperparams": HYPERPARAMS,
            }, str(MODEL_DIR / "quickdraw_gnn.pth"))
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= HYPERPARAMS["early_stop_patience"]:
            print(f"\nEarly stopping triggered")
            break

    print("=" * 60)
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'quickdraw_gnn.pth'}")


if __name__ == "__main__":
    main()
