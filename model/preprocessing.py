"""
Shared preprocessing utilities for QuickDraw sketch classification.

This module ensures consistent preprocessing between training and inference:
  - Normalization: uint8 [0, 255] -> float32 [0, 1]
  - Centering and scaling (handled differently in train vs inference)
  - Tensor conversion with correct shapes

The preprocessing contract:
  Input:  28x28 grayscale image, white strokes on black background
  Output: tensor of shape (1, 1, 28, 28) with values in [0, 1]
          where stroke pixels are ~1.0 and background is ~0.0
"""

from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

# QuickDraw .npy files store images as uint8 [0, 255]
# We normalize to float32 [0, 1] to match the frontend pixel format
MAX_PIXEL_VALUE = 255.0

# Expected input dimensions
IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 784


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pixel_list(pixels: Sequence[float]) -> None:
    """
    Validate a raw pixel list from the frontend.

    Args:
        pixels: Sequence of 784 float values representing a 28x28 grayscale image.

    Raises:
        ValueError: If the input fails any validation check.
        TypeError: If the input type is wrong.
    """
    if pixels is None:
        raise ValueError("pixels cannot be None")

    if not hasattr(pixels, '__len__') or not hasattr(pixels, '__getitem__'):
        raise TypeError(f"Expected sequence, got {type(pixels).__name__}")

    if len(pixels) != NUM_PIXELS:
        raise ValueError(f"Expected {NUM_PIXELS} pixels, got {len(pixels)}")

    for i, value in enumerate(pixels):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Pixel at index {i} is not a number: {type(value).__name__}")
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Pixel at index {i} is NaN or Inf")
        if not (-0.01 <= value <= 1.01):  # tiny tolerance for floating point
            raise ValueError(f"Pixel at index {i} is out of range [0, 1]: {value}")


def validate_numpy_image(image: np.ndarray) -> None:
    """
    Validate a numpy image array for training/inference.

    Args:
        image: Numpy array of shape (28, 28) or (784,) with values in [0, 255] or [0, 1].

    Raises:
        ValueError: If the input fails validation.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image).__name__}")

    if image.dtype not in (np.uint8, np.float32, np.float64):
        raise ValueError(f"Expected dtype uint8 or float, got {image.dtype}")

    flat_size = np.prod(image.shape)
    if flat_size != NUM_PIXELS:
        raise ValueError(f"Expected {NUM_PIXELS} pixels, got shape {image.shape} ({flat_size} elements)")


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize a raw QuickDraw image to [0, 1] float32.

    QuickDraw .npy files store images as uint8 [0, 255] with white strokes
    on black background. The frontend sends pixels in [0, 1] with the same
    semantic meaning (high values = stroke, low values = background).

    Args:
        image: Numpy array of shape (28, 28) or (784,) with dtype uint8 or float.
               Values should be in [0, 255] for uint8 or [0, 1] for float.

    Returns:
        Normalized float32 array of shape (28, 28) with values in [0, 1].
    """
    validate_numpy_image(image)

    # Reshape to 2D if needed
    if image.ndim == 1:
        image = image.reshape(IMAGE_SIZE, IMAGE_SIZE)

    # Ensure 2D
    if image.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {image.shape}")

    if image.dtype == np.uint8:
        # uint8 [0, 255] -> float32 [0, 1]
        return image.astype(np.float32) / MAX_PIXEL_VALUE
    else:
        # Already float, just ensure correct dtype
        return image.astype(np.float32)


def preprocess_for_training(
    image: np.ndarray,
    augment: bool = False,
) -> torch.Tensor:
    """
    Preprocess a single image for training/validation.

    Args:
        image: Numpy array of shape (28, 28) uint8 or (784,) uint8.
        augment: If True, random augmentation should be applied by the caller
                 (this function only does normalization).

    Returns:
        Tensor of shape (1, 28, 28) float32 in [0, 1].
    """
    normalized = normalize_image(image)  # (28, 28) float32
    tensor = torch.from_numpy(normalized).unsqueeze(0)  # (1, 28, 28)
    return tensor


def preprocess_for_inference(pixels: Sequence[float]) -> torch.Tensor:
    """
    Preprocess a pixel list from the frontend for model inference.

    Args:
        pixels: Sequence of 784 float values in [0, 1].

    Returns:
        Tensor of shape (1, 1, 28, 28) float32.

    Raises:
        ValueError: If validation fails.
    """
    validate_pixel_list(pixels)

    # Convert to numpy then tensor for efficiency
    arr = np.array(pixels, dtype=np.float32).reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    return torch.from_numpy(arr)


def batch_preprocess_for_inference(
    pixel_lists: Sequence[Sequence[float]],
) -> torch.Tensor:
    """
    Preprocess multiple pixel lists for batched inference.

    Args:
        pixel_lists: List of pixel sequences, each of length 784.

    Returns:
        Tensor of shape (batch_size, 1, 28, 28) float32.
    """
    if not pixel_lists:
        raise ValueError("pixel_lists cannot be empty")

    batch = []
    for pixels in pixel_lists:
        validate_pixel_list(pixels)
        arr = np.array(pixels, dtype=np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE)
        batch.append(arr)

    stacked = np.stack(batch, axis=0)  # (B, 1, 28, 28)
    return torch.from_numpy(stacked)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with synthetic data
    dummy_uint8 = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    normalized = normalize_image(dummy_uint8)
    print(f"Input dtype: {dummy_uint8.dtype}, range: [{dummy_uint8.min()}, {dummy_uint8.max()}]")
    print(f"Output dtype: {normalized.dtype}, range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test pixel list validation
    valid_pixels = [0.5] * 784
    tensor = preprocess_for_inference(valid_pixels)
    print(f"Inference tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

    # Test invalid inputs
    try:
        validate_pixel_list([0.5] * 783)
    except ValueError as e:
        print(f"Validation correctly caught: {e}")
