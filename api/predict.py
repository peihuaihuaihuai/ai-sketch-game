"""
Prediction endpoint logic for the Flask backend.

This module handles model loading, image preprocessing, and inference
for the QuickDraw sketch classifier. Optimized for sub-100ms latency.

Supports both CNN-only and CNN+GNN hybrid inference when stroke data
is provided.
"""

import os
import math
import time
import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

from model.model import create_model, MODEL_REGISTRY
from model.preprocessing import preprocess_for_inference, validate_pixel_list

try:
    from model.stroke_graph import (
        build_graph_from_strokes,
        HybridQuickDrawModel,
    )
    _HYBRID_AVAILABLE = True
except Exception:
    _HYBRID_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Ordered class labels (must match training label order)
CLASS_LABELS = ['airplane', 'car', 'cat', 'dog', 'house', 'tree']

# Number of top predictions to return
TOP_K = 5

# Path to saved model weights
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'quickdraw_cnn.pth'
GNN_MODEL_PATH = Path(__file__).parent.parent / 'model' / 'quickdraw_gnn.pth'

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_device = torch.device('cpu')
_model_name = 'resnet'
_model_classes = len(CLASS_LABELS)


def _load_model():
    """
    Load the trained model from disk.

    Supports both old format (raw state_dict) and new format (checkpoint dict
    with metadata).

    Returns:
        The loaded model in evaluation mode.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
        RuntimeError: If model state dict loading fails.
    """
    global _model_name, _model_classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please train the model first by running: python model/train.py"
        )

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    # Detect format: new checkpoint dict or old raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        _model_name = checkpoint.get('model_name', 'resnet')
        _model_classes = checkpoint.get('num_classes', len(CLASS_LABELS))
        loaded_classes = checkpoint.get('class_names', CLASS_LABELS)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
              f"val_acc={checkpoint.get('val_acc', 0):.4f}")
    else:
        # Old format: raw state_dict
        state_dict = checkpoint
        _model_name = 'cnn'
        _model_classes = len(CLASS_LABELS)
        loaded_classes = CLASS_LABELS

    # Warn if class mismatch
    if loaded_classes != CLASS_LABELS:
        print(f"WARNING: Model trained with classes {loaded_classes}, "
              f"but API expects {CLASS_LABELS}")

    model = create_model(_model_name, num_classes=_model_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(_device)
    model.eval()

    # Compile model for faster inference if available (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile') and _device.type == 'cuda':
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for optimized inference")
    except Exception:
        pass  # torch.compile may not be available

    return model


def get_model():
    """
    Get the singleton model instance, initializing it on first call.

    Returns:
        The loaded model.
    """
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ---------------------------------------------------------------------------
# Hybrid model loading (optional)
# ---------------------------------------------------------------------------

_hybrid_model = None


def _load_hybrid_model():
    """
    Load a hybrid CNN+GNN model.

    Workflow:
      1. Create a fresh CNN backbone and load the trained checkpoint into it.
      2. Pass the pre-loaded backbone to HybridQuickDrawModel, which replaces
         the final classifier with a 128D embedding projection (preserving all
         trained conv-layer weights).
      3. The GNN branch and joint classifier start randomly initialized —
         the CNN branch still contributes meaningful features immediately.

    Returns the hybrid model if available, otherwise None.
    """
    if not _HYBRID_AVAILABLE:
        return None

    if not MODEL_PATH.exists():
        return None

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            base_name = checkpoint.get('model_name', 'resnet')
            num_classes = checkpoint.get('num_classes', len(CLASS_LABELS))
        else:
            state_dict = checkpoint
            base_name = 'cnn'
            num_classes = len(CLASS_LABELS)

        # Step 1: Create CNN backbone and load trained weights FIRST
        cnn_backbone = create_model(base_name, num_classes=num_classes)
        cnn_backbone.load_state_dict(state_dict, strict=True)

        # Step 2: Build hybrid — _replace_cnn_classifier swaps the final
        # Linear(num_classes) for Linear(128) embedding. Conv weights survive.
        hybrid = HybridQuickDrawModel(
            num_classes=num_classes,
            cnn_backbone=cnn_backbone,
        )
        hybrid.to(_device)
        hybrid.eval()
        print(f"Hybrid model loaded (CNN backbone: {base_name}, "
              f"CNN conv weights preserved, GNN + joint head random-init)")
        return hybrid
    except Exception as e:
        print(f"Hybrid model initialization skipped: {e}")
        return None


def get_hybrid_model():
    """Get the singleton hybrid model instance."""
    global _hybrid_model
    if _hybrid_model is None:
        _hybrid_model = _load_hybrid_model()
    return _hybrid_model


# ---------------------------------------------------------------------------
# GNN-only model loading (baseline)
# ---------------------------------------------------------------------------

_gnn_model = None


def _load_gnn_model():
    """
    Load a standalone GNN classifier from disk.

    Returns the GNN model if a checkpoint exists, otherwise None.
    """
    if not _HYBRID_AVAILABLE:
        return None

    if not GNN_MODEL_PATH.exists():
        return None

    try:
        from model.stroke_graph import SketchGNNClassifier
        checkpoint = torch.load(GNN_MODEL_PATH, map_location=_device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_classes = checkpoint.get('num_classes', len(CLASS_LABELS))
        else:
            state_dict = checkpoint
            num_classes = len(CLASS_LABELS)

        gnn = SketchGNNClassifier(num_classes=num_classes)
        gnn.load_state_dict(state_dict, strict=True)
        gnn.to(_device)
        gnn.eval()
        print(f"GNN model loaded from {GNN_MODEL_PATH}")
        return gnn
    except Exception as e:
        print(f"GNN model loading skipped: {e}")
        return None


def get_gnn_model():
    """Get the singleton GNN model instance."""
    global _gnn_model
    if _gnn_model is None:
        _gnn_model = _load_gnn_model()
    return _gnn_model


def warm_up_model() -> float:
    """
    Run dummy inferences to warm up PyTorch and measure latency.

    Returns:
        Duration of the warm-up inference in milliseconds.
    """
    model = get_model()
    dummy_input = torch.zeros(1, 1, 28, 28, device=_device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000 / 100
    return duration_ms


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_pixels(pixels: list) -> None:
    """
    Validate the input pixel array from the client.

    Args:
        pixels: List of 784 float values representing a 28x28 grayscale image.

    Raises:
        ValueError: If the input fails any validation check.
    """
    validate_pixel_list(pixels)


def validate_strokes(strokes: list) -> None:
    """
    Validate stroke sequence data from the client.

    Args:
        strokes: List of strokes, each stroke is a list of {x, y} dicts.

    Raises:
        ValueError: If the input fails validation.
    """
    if not isinstance(strokes, list):
        raise ValueError("strokes must be a list")
    if len(strokes) > 100:
        raise ValueError("Too many strokes (max 100)")
    for i, stroke in enumerate(strokes):
        if not isinstance(stroke, list):
            raise ValueError(f"Stroke {i} must be a list of points")
        if len(stroke) > 5000:
            raise ValueError(f"Stroke {i} has too many points (max 5000)")
        for j, pt in enumerate(stroke):
            if not isinstance(pt, dict):
                raise ValueError(f"Stroke {i} point {j} must be an object")
            if 'x' not in pt or 'y' not in pt:
                raise ValueError(f"Stroke {i} point {j} missing x or y")
            if not isinstance(pt['x'], (int, float)) or not isinstance(pt['y'], (int, float)):
                raise ValueError(f"Stroke {i} point {j} coordinates must be numbers")
            if not (0.0 <= pt['x'] <= 1.0) or not (0.0 <= pt['y'] <= 1.0):
                raise ValueError(f"Stroke {i} point {j} coordinates out of range [0, 1]")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(pixels: list, strokes: list = None) -> dict:
    """
    Run CNN inference on a preprocessed sketch and return Top-K predictions.

    Args:
        pixels: List of 784 float values in [0, 1] representing a 28x28
                grayscale image (white strokes on black background).
        strokes: Optional list of stroke sequences (reserved for future
                 hybrid CNN+GNN inference; currently unused).

    Returns:
        Dictionary with 'top5' (list of {label, probability}), 'latency_ms',
        and 'model' keys.
    """
    # Validate input
    validate_pixels(pixels)
    if strokes is not None:
        validate_strokes(strokes)

    # Preprocess: list -> tensor (1, 1, 28, 28)
    tensor = preprocess_for_inference(pixels)
    tensor = tensor.to(_device)

    # CNN inference
    model = get_model()
    infer_start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    infer_end = time.perf_counter()
    latency_ms = (infer_end - infer_start) * 1000

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # Get Top-K predictions sorted by probability descending
    top_indices = np.argsort(probs)[-TOP_K:][::-1]

    top5 = [
        {
            'label': CLASS_LABELS[idx],
            'probability': round(float(probs[idx]), 4),
        }
        for idx in top_indices
    ]

    return {
        'top5': top5,
        'latency_ms': round(latency_ms, 3),
        'model': _model_name,
    }


def predict_batch(pixel_batches: list, stroke_batches: list = None) -> list:
    """
    Run batched inference on multiple sketches.

    Args:
        pixel_batches: List of pixel lists, each of length 784.
        stroke_batches: Optional list of stroke sequences per sketch.

    Returns:
        List of prediction result dictionaries.
    """
    from model.preprocessing import batch_preprocess_for_inference

    if not pixel_batches:
        return []

    # Validate all inputs
    for pixels in pixel_batches:
        validate_pixels(pixels)

    # Batch preprocess
    tensor = batch_preprocess_for_inference(pixel_batches)
    tensor = tensor.to(_device)

    model = get_model()

    with torch.no_grad():
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for prob in probs:
        top_indices = np.argsort(prob)[-TOP_K:][::-1]
        top5 = [
            {'label': CLASS_LABELS[idx], 'probability': round(float(prob[idx]), 4)}
            for idx in top_indices
        ]
        results.append({'top5': top5, 'model': _model_name})

    return results


# ---------------------------------------------------------------------------
# Reset / state management
# ---------------------------------------------------------------------------

# Simple in-memory session tracking (resets on server restart)
_prediction_count = 0
_last_prediction_time = None


def reset_state() -> dict:
    """
    Reset server-side prediction tracking state.

    Returns:
        Dictionary with reset confirmation.
    """
    global _prediction_count, _last_prediction_time
    _prediction_count = 0
    _last_prediction_time = None
    return {
        'status': 'reset',
        'message': 'Prediction state cleared successfully',
    }


def get_stats() -> dict:
    """
    Get server-side prediction statistics.

    Returns:
        Dictionary with prediction count and timing info.
    """
    return {
        'predictions_served': _prediction_count,
        'last_prediction': _last_prediction_time,
        'model_name': _model_name,
        'model_classes': _model_classes,
    }
