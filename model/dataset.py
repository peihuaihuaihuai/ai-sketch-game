"""
QuickDraw dataset loader and data acquisition utilities.

Supports:
  - Downloading official Google QuickDraw .npy files
  - Loading and normalizing bitmap data
  - Synthetic fallback when downloads fail
  - PyTorch Dataset with optional augmentation
"""

import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Resolve project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocessing import normalize_image, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUICKDRAW_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy"
)

DEFAULT_CLASS_NAMES = ["airplane", "car", "cat", "dog", "house", "tree"]


# ---------------------------------------------------------------------------
# Data Acquisition
# ---------------------------------------------------------------------------

def _validate_npy_file(filepath: Path, name: str) -> None:
    """
    Validate that a .npy file has the expected QuickDraw format (N, 784) uint8.

    Args:
        filepath: Path to the .npy file
        name: Category name for error messages

    Raises:
        ValueError: If the file is corrupted or has wrong shape.
    """
    try:
        raw = np.load(filepath)
    except Exception as e:
        raise ValueError(f"[{name}] Cannot load .npy file: {e}")

    if raw.ndim != 2:
        raise ValueError(
            f"[{name}] Expected 2D array, got shape {raw.shape}"
        )
    if raw.shape[1] != 784:
        raise ValueError(
            f"[{name}] Expected 784 columns, got {raw.shape[1]}"
        )
    if raw.dtype != np.uint8:
        raise ValueError(
            f"[{name}] Expected dtype uint8, got {raw.dtype}"
        )

    # Also verify total byte count consistency
    expected_bytes = raw.shape[0] * raw.shape[1]
    actual_bytes = raw.nbytes
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"[{name}] Size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
        )


def download_quickdraw_class(name: str, data_dir: Path, timeout: int = 120) -> Path:
    """
    Download a single QuickDraw class .npy file from Google Cloud Storage.
    Validates the downloaded file before returning.

    Args:
        name: Category name (e.g., "cat")
        data_dir: Directory to save downloaded files
        timeout: Download timeout in seconds

    Returns:
        Path to the downloaded .npy file

    Raises:
        RuntimeError: If download fails or file is corrupted.
    """
    filepath = data_dir / f"{name}.npy"

    # If file exists but is corrupted, delete it first
    if filepath.exists():
        try:
            _validate_npy_file(filepath, name)
            return filepath
        except ValueError as e:
            print(f"  [{name}] existing file invalid: {e}")
            print(f"  [{name}] deleting corrupted file...")
            filepath.unlink()

    url = QUICKDRAW_URL.format(name=name)
    print(f"  [{name}] downloading from {url} ...")

    try:
        urllib.request.urlretrieve(url, str(filepath))
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  [{name}] saved ({file_size_mb:.1f} MB)")

        # Validate downloaded file
        _validate_npy_file(filepath, name)
        print(f"  [{name}] validation passed")
        return filepath
    except Exception as e:
        # Clean up partial or corrupted file
        if filepath.exists():
            filepath.unlink()
        raise RuntimeError(f"Failed to download {name}: {e}")


def load_quickdraw_class(
    name: str,
    data_dir: Path,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Load a single QuickDraw class .npy file.

    Args:
        name: Category name
        data_dir: Directory containing .npy files
        max_samples: Maximum samples to load (None = all)

    Returns:
        Tuple of (images, n_use) where images is (N, 28, 28) uint8.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is corrupted.
    """
    filepath = data_dir / f"{name}.npy"
    if not filepath.exists():
        raise FileNotFoundError(f"QuickDraw data not found: {filepath}")

    # Load directly (mmap can exhaust handles on Windows with many files)
    raw = np.load(filepath)
    n_available = raw.shape[0]
    n_use = min(n_available, max_samples) if max_samples else n_available

    # Extract subset and reshape from (N, 784) to (N, 28, 28)
    images = raw[:n_use].reshape(-1, IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
    del raw  # free memory immediately

    return images, n_use


def load_quickdraw_data(
    class_names: list[str],
    data_dir: Path,
    max_samples_per_class: int | None = None,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load QuickDraw .npy files for all classes.
    Auto-re-downloads corrupted files.

    Args:
        class_names: List of category names
        data_dir: Directory containing .npy files
        max_samples_per_class: Maximum samples per class
        download: If True, attempt to download missing or corrupted files

    Returns:
        Tuple of (images, labels) where:
            images: shape (N, 28, 28) uint8
            labels: shape (N,) int64 class indices

    Raises:
        RuntimeError: If any file cannot be loaded or downloaded.
    """
    all_images = []
    all_labels = []

    print("Loading QuickDraw data...")
    data_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, name in enumerate(class_names):
        filepath = data_dir / f"{name}.npy"

        # Check if file exists and is valid; if not, re-download
        file_valid = False
        if filepath.exists():
            try:
                _validate_npy_file(filepath, name)
                file_valid = True
            except ValueError as e:
                print(f"  [{name}] file corrupted: {e}")
                if download:
                    print(f"  [{name}] will re-download...")
                    filepath.unlink()
                else:
                    raise RuntimeError(f"[{name}] corrupted file and download disabled: {e}")

        if not file_valid and download:
            try:
                download_quickdraw_class(name, data_dir)
            except Exception as e:
                raise RuntimeError(f"[{name}] download failed: {e}")
        elif not file_valid:
            raise FileNotFoundError(f"QuickDraw data not found: {filepath}")

        images, n_use = load_quickdraw_class(name, data_dir, max_samples_per_class)
        labels = np.full(n_use, class_idx, dtype=np.int64)

        all_images.append(images)
        all_labels.append(labels)

        print(f"  [{name}] loaded {n_use:,} samples")

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Total dataset: {len(images):,} samples, {len(class_names)} classes")
    print("Using real QuickDraw dataset")
    return images, labels


# ---------------------------------------------------------------------------
# Synthetic Fallback
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    class_names: list[str],
    samples_per_class: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sketch-like data when real QuickDraw data is unavailable.

    Creates simple geometric patterns to simulate different categories.

    Args:
        class_names: List of category names
        samples_per_class: Number of synthetic samples per class
        seed: Random seed for reproducibility

    Returns:
        Tuple of (images, labels) arrays
    """
    rng = np.random.RandomState(seed)
    all_images = []
    all_labels = []

    print("WARNING: Using synthetic data (QuickDraw download unavailable)")
    print("For real accuracy, ensure internet connection for QuickDraw download.")

    def _draw_line(img, x1, y1, x2, y2, thickness=1):
        """Bresenham-like line drawing on 28x28 canvas."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            for tx in range(-thickness // 2, thickness // 2 + 1):
                for ty in range(-thickness // 2, thickness // 2 + 1):
                    px, py = x1 + tx, y1 + ty
                    if 0 <= px < 28 and 0 <= py < 28:
                        img[py, px] = 255.0
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _draw_circle(img, cx, cy, r, thickness=1):
        """Draw a circle on 28x28 canvas."""
        for angle in np.linspace(0, 2 * math.pi, int(2 * math.pi * r * 2)):
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            for tx in range(-thickness // 2, thickness // 2 + 1):
                for ty in range(-thickness // 2, thickness // 2 + 1):
                    px, py = x + tx, y + ty
                    if 0 <= px < 28 and 0 <= py < 28:
                        img[py, px] = 255.0

    def _draw_rect(img, x1, y1, x2, y2, thickness=1):
        """Draw a rectangle on 28x28 canvas."""
        _draw_line(img, x1, y1, x2, y1, thickness)
        _draw_line(img, x2, y1, x2, y2, thickness)
        _draw_line(img, x2, y2, x1, y2, thickness)
        _draw_line(img, x1, y2, x1, y1, thickness)

    for class_idx, name in enumerate(class_names):
        for _ in range(samples_per_class):
            img = np.zeros((28, 28), dtype=np.float32)

            # Add random noise background
            img += rng.randint(0, 20, size=(28, 28)).astype(np.float32)

            # Center offset with some randomness
            cx, cy = 14 + rng.randint(-3, 4), 14 + rng.randint(-3, 4)
            scale = rng.uniform(0.7, 1.1)

            if name == "airplane":
                _draw_line(img, cx - 8, cy, cx + 8, cy, 2)
                _draw_line(img, cx - 3, cy - 6, cx + 3, cy - 6, 2)
                _draw_line(img, cx - 2, cy + 5, cx + 2, cy + 5, 2)
                _draw_line(img, cx + 6, cy - 1, cx + 10, cy - 4, 2)
                _draw_line(img, cx + 6, cy - 1, cx + 10, cy + 2, 2)

            elif name == "car":
                _draw_rect(img, cx - 7, cy - 3, cx + 7, cy + 3, 2)
                _draw_line(img, cx - 4, cy - 3, cx - 2, cy - 7, 2)
                _draw_line(img, cx + 2, cy - 3, cx + 4, cy - 7, 2)
                _draw_line(img, cx - 2, cy - 7, cx + 4, cy - 7, 2)
                _draw_circle(img, cx - 5, cy + 3, 2, 2)
                _draw_circle(img, cx + 5, cy + 3, 2, 2)

            elif name == "cat":
                _draw_circle(img, cx, cy - 2, int(5 * scale), 2)
                _draw_line(img, cx - 4, cy - 5, cx - 2, cy - 10, 2)
                _draw_line(img, cx - 2, cy - 10, cx, cy - 5, 2)
                _draw_line(img, cx, cy - 5, cx + 2, cy - 10, 2)
                _draw_line(img, cx + 2, cy - 10, cx + 4, cy - 5, 2)
                _draw_line(img, cx, cy + 2, cx, cy + 8, 2)
                _draw_line(img, cx, cy + 6, cx + 6, cy + 4, 2)
                _draw_line(img, cx - 5, cy - 2, cx - 9, cy - 3, 1)
                _draw_line(img, cx + 5, cy - 2, cx + 9, cy - 3, 1)

            elif name == "dog":
                _draw_circle(img, cx, cy - 2, int(5 * scale), 2)
                _draw_line(img, cx - 5, cy - 3, cx - 7, cy + 2, 2)
                _draw_line(img, cx + 5, cy - 3, cx + 7, cy + 2, 2)
                _draw_line(img, cx, cy + 2, cx, cy + 8, 2)
                _draw_line(img, cx, cy + 6, cx + 7, cy + 2, 2)
                _draw_line(img, cx - 1, cy + 1, cx + 1, cy + 1, 2)
                _draw_line(img, cx, cy + 1, cx, cy + 3, 2)

            elif name == "house":
                _draw_rect(img, cx - 6, cy, cx + 6, cy + 8, 2)
                _draw_line(img, cx - 7, cy, cx, cy - 8, 2)
                _draw_line(img, cx, cy - 8, cx + 7, cy, 2)
                _draw_line(img, cx - 7, cy, cx + 7, cy, 2)
                _draw_line(img, cx - 2, cy + 8, cx - 2, cy + 3, 2)
                _draw_line(img, cx + 2, cy + 8, cx + 2, cy + 3, 2)
                _draw_line(img, cx - 2, cy + 3, cx + 2, cy + 3, 2)
                _draw_rect(img, cx - 5, cy + 1, cx - 3, cy + 3, 1)
                _draw_rect(img, cx + 3, cy + 1, cx + 5, cy + 3, 1)

            elif name == "tree":
                _draw_line(img, cx, cy + 3, cx, cy + 10, 3)
                _draw_circle(img, cx, cy - 3, int(5 * scale), 2)
                _draw_circle(img, cx - 4, cy + 1, int(4 * scale), 2)
                _draw_circle(img, cx + 4, cy + 1, int(4 * scale), 2)

            all_images.append(img)
            all_labels.append(class_idx)

    images = np.stack(all_images, axis=0).astype(np.uint8)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"Synthetic dataset: {len(images):,} samples, {len(class_names)} classes")
    return images, labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class QuickDrawDataset(Dataset):
    """
    PyTorch Dataset for QuickDraw sketch images.

    Each sample is a 28x28 grayscale image normalized to [0, 1].
    Data augmentation is applied during training via torchvision transforms.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        """
        Args:
            images: Array of shape (N, 28, 28) with values in [0, 255] uint8
            labels: Array of shape (N,) with class indices
            augment: If True, apply random augmentation transforms
        """
        if len(images) != len(labels):
            raise ValueError(f"Images and labels must have same length: {len(images)} vs {len(labels)}")

        self.images = images
        self.labels = labels
        self.augment = augment

        # Pre-normalize all images to [0, 1] for faster access
        # uint8 [0, 255] -> float32 [0, 1]
        self.normalized_images = images.astype(np.float32) / 255.0

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(
                    size=IMAGE_SIZE,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, label) where image_tensor is (1, 28, 28)
        """
        image = self.normalized_images[idx]  # (28, 28), values in [0, 1]
        label = self.labels[idx]

        if self.augment:
            # Convert to uint8 for torchvision transforms
            image_uint8 = (image * 255).astype(np.uint8)
            image_tensor = self.transform(image_uint8)  # (1, 28, 28)
        else:
            # Direct tensor conversion, add channel dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, 28, 28)

        return image_tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.

    Args:
        labels: Array of class indices.

    Returns:
        Tensor of shape (num_classes,) with class weights.
    """
    num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    # Inverse frequency normalization
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()


if __name__ == "__main__":
    # Quick test
    print("Testing synthetic data generation...")
    images, labels = generate_synthetic_data(DEFAULT_CLASS_NAMES, samples_per_class=10)
    print(f"Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    print("\nTesting dataset...")
    dataset = QuickDrawDataset(images, labels, augment=True)
    img, lbl = dataset[0]
    print(f"Dataset output shape: {img.shape}, label: {lbl}")
