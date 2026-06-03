"""
Prediction endpoint logic for the Flask backend.

Handles model loading, image preprocessing, and CNN inference
for the QuickDraw sketch classifier.
"""

import time
import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

from model.model import create_model
from model.preprocessing import preprocess_for_inference, validate_pixel_list


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

CLASS_LABELS = ['airplane', 'car', 'cat', 'dog', 'house', 'tree']
TOP_K = 5
MODEL_PATH = Path(__file__).parent.parent / 'model' / 'quickdraw_cnn.pth'

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_device = torch.device('cpu')
_model_name = 'resnet'
_model_classes = len(CLASS_LABELS)


def _load_model():
    """Load the trained QuickDrawResNet model from disk."""
    global _model_name, _model_classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please train the model first by running: python model/train.py"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        _model_name = checkpoint.get('model_name', 'resnet')
        _model_classes = checkpoint.get('num_classes', len(CLASS_LABELS))
        loaded_classes = checkpoint.get('class_names', CLASS_LABELS)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
              f"val_acc={checkpoint.get('val_acc', 0):.4f}")
    else:
        state_dict = checkpoint
        _model_name = 'cnn'
        _model_classes = len(CLASS_LABELS)
        loaded_classes = CLASS_LABELS

    if loaded_classes != CLASS_LABELS:
        print(f"WARNING: Model trained with classes {loaded_classes}, "
              f"but API expects {CLASS_LABELS}")

    model = create_model(_model_name, num_classes=_model_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(_device)
    model.eval()

    try:
        if hasattr(torch, 'compile') and _device.type == 'cuda':
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for optimized inference")
    except Exception:
        pass

    return model


def get_model():
    """Get the singleton model instance, initializing it on first call."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def warm_up_model() -> float:
    """Run dummy inferences to warm up PyTorch and measure latency."""
    model = get_model()
    dummy_input = torch.zeros(1, 1, 28, 28, device=_device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.perf_counter()

    return (end - start) * 1000 / 100


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_pixels(pixels: list) -> None:
    """Validate the input pixel array from the client."""
    validate_pixel_list(pixels)


def validate_strokes(strokes: list) -> None:
    """Validate stroke sequence data from the client (reserved for future use)."""
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
    """Run CNN inference on a preprocessed sketch and return Top-K predictions.

    Args:
        pixels: List of 784 float values in [0, 1] representing a 28x28
                grayscale image (white strokes on black background).
        strokes: Optional list of stroke sequences (reserved for future use).

    Returns:
        Dictionary with 'top5', 'latency_ms', and 'model' keys.
    """
    validate_pixels(pixels)
    if strokes is not None:
        validate_strokes(strokes)

    tensor = preprocess_for_inference(pixels)
    tensor = tensor.to(_device)

    model = get_model()
    infer_start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    infer_end = time.perf_counter()
    latency_ms = (infer_end - infer_start) * 1000

    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    top_indices = np.argsort(probs)[-TOP_K:][::-1]

    top5 = [
        {'label': CLASS_LABELS[idx], 'probability': round(float(probs[idx]), 4)}
        for idx in top_indices
    ]

    return {
        'top5': top5,
        'latency_ms': round(latency_ms, 3),
        'model': _model_name,
    }


def predict_batch(pixel_batches: list, stroke_batches: list = None) -> list:
    """Run batched inference on multiple sketches."""
    from model.preprocessing import batch_preprocess_for_inference

    if not pixel_batches:
        return []

    for pixels in pixel_batches:
        validate_pixels(pixels)

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

_prediction_count = 0
_last_prediction_time = None


def reset_state() -> dict:
    """Reset server-side prediction tracking state."""
    global _prediction_count, _last_prediction_time
    _prediction_count = 0
    _last_prediction_time = None
    return {'status': 'reset', 'message': 'Prediction state cleared successfully'}


def get_stats() -> dict:
    """Get server-side prediction statistics."""
    return {
        'predictions_served': _prediction_count,
        'last_prediction': _last_prediction_time,
        'model_name': _model_name,
        'model_classes': _model_classes,
    }
