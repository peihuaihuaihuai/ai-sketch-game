"""
Model package for the QuickDraw AI sketch recognition web app.

This package contains the PyTorch model architectures, data loading utilities,
preprocessing pipeline, and training scripts for the QuickDraw classifier.
"""

from .model import QuickDrawCNN, QuickDrawResNet, create_model, get_model_summary
from .preprocessing import (
    normalize_image,
    preprocess_for_training,
    preprocess_for_inference,
    validate_pixel_list,
)
from .dataset import QuickDrawDataset, load_quickdraw_data, generate_synthetic_data

try:
    from .stroke_graph import (
        build_graph_from_strokes,
        SketchGNNClassifier,
        HybridQuickDrawModel,
        StrokeGNN,
    )
    _STROKE_GRAPH_AVAILABLE = True
except Exception:
    _STROKE_GRAPH_AVAILABLE = False

__all__ = [
    'QuickDrawCNN',
    'QuickDrawResNet',
    'create_model',
    'get_model_summary',
    'normalize_image',
    'preprocess_for_training',
    'preprocess_for_inference',
    'validate_pixel_list',
    'QuickDrawDataset',
    'load_quickdraw_data',
    'generate_synthetic_data',
]

if _STROKE_GRAPH_AVAILABLE:
    __all__.extend([
        'build_graph_from_strokes',
        'SketchGNNClassifier',
        'HybridQuickDrawModel',
        'StrokeGNN',
    ])
