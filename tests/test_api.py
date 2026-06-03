"""
Unit tests for the Flask prediction API and preprocessing pipeline.

Verifies request handling, response schema, input validation,
and preprocessing consistency.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.predict import validate_pixels, validate_strokes, CLASS_LABELS
from model.preprocessing import (
    validate_pixel_list,
    normalize_image,
    preprocess_for_inference,
    preprocess_for_training,
)


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for the input validation function."""

    def test_valid_input(self):
        """Valid 784-float array should pass validation."""
        pixels = [0.5] * 784
        validate_pixels(pixels)

    def test_wrong_length(self):
        """Array with wrong length should raise ValueError."""
        pixels = [0.5] * 780
        with pytest.raises(ValueError, match="Expected 784 pixels"):
            validate_pixels(pixels)

    def test_non_sequence_input(self):
        """Non-sequence input should raise TypeError/ValueError."""
        with pytest.raises((TypeError, ValueError)):
            validate_pixels("not a list")

    def test_none_input(self):
        """None input should raise ValueError."""
        with pytest.raises(ValueError):
            validate_pixels(None)

    def test_nan_values(self):
        """NaN values should raise ValueError."""
        pixels = [0.5] * 784
        pixels[100] = float('nan')
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_pixels(pixels)

    def test_inf_values(self):
        """Inf values should raise ValueError."""
        pixels = [0.5] * 784
        pixels[100] = float('inf')
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_pixels(pixels)

    def test_out_of_range_high(self):
        """Values > 1.0 should raise ValueError."""
        pixels = [0.5] * 784
        pixels[100] = 1.5
        with pytest.raises(ValueError, match="out of range"):
            validate_pixels(pixels)

    def test_out_of_range_low(self):
        """Values < 0.0 should raise ValueError."""
        pixels = [0.5] * 784
        pixels[100] = -0.1
        with pytest.raises(ValueError, match="out of range"):
            validate_pixels(pixels)

    def test_boundary_values(self):
        """Values exactly at 0.0 and 1.0 should pass."""
        pixels = [0.0] * 392 + [1.0] * 392
        validate_pixels(pixels)

    def test_tuple_input(self):
        """Tuple of 784 values should pass."""
        pixels = tuple([0.5] * 784)
        validate_pixels(pixels)


class TestStrokeValidation:
    """Tests for stroke sequence validation."""

    def test_valid_strokes(self):
        strokes = [
            [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}],
            [{"x": 0.5, "y": 0.6}],
        ]
        validate_strokes(strokes)

    def test_empty_strokes(self):
        validate_strokes([])

    def test_invalid_not_list(self):
        with pytest.raises(ValueError, match="strokes must be a list"):
            validate_strokes("not a list")

    def test_invalid_point_missing_key(self):
        strokes = [[{"x": 0.1}]]
        with pytest.raises(ValueError, match="missing x or y"):
            validate_strokes(strokes)

    def test_invalid_coordinate_out_of_range(self):
        strokes = [[{"x": 1.5, "y": 0.5}]]
        with pytest.raises(ValueError, match="out of range"):
            validate_strokes(strokes)

    def test_invalid_coordinate_type(self):
        strokes = [[{"x": "a", "y": 0.5}]]
        with pytest.raises(ValueError, match="must be numbers"):
            validate_strokes(strokes)


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    def test_normalize_uint8(self):
        """Test normalization of uint8 image to [0, 1]."""
        img = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
        normalized = normalize_image(img)
        assert normalized.dtype == np.float32
        assert normalized.shape == (28, 28)
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0

    def test_normalize_float(self):
        """Test normalization of float image."""
        img = np.random.rand(28, 28).astype(np.float32)
        normalized = normalize_image(img)
        assert normalized.dtype == np.float32
        assert normalized.shape == (28, 28)

    def test_normalize_flattened(self):
        """Test normalization of flattened 784 array."""
        img = np.random.randint(0, 256, size=(784,), dtype=np.uint8)
        normalized = normalize_image(img)
        assert normalized.shape == (28, 28)

    def test_preprocess_for_inference_shape(self):
        """Test inference preprocessing returns correct shape."""
        pixels = [0.5] * 784
        tensor = preprocess_for_inference(pixels)
        assert tensor.shape == (1, 1, 28, 28)
        assert tensor.dtype == torch.float32

    def test_preprocess_for_training_shape(self):
        """Test training preprocessing returns correct shape."""
        img = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
        tensor = preprocess_for_training(img)
        assert tensor.shape == (1, 28, 28)
        assert tensor.dtype == torch.float32

    def test_preprocess_value_range(self):
        """Test that preprocessed values are in [0, 1]."""
        pixels = [0.0] * 784
        tensor = preprocess_for_inference(pixels)
        assert tensor.min() == 0.0
        assert tensor.max() == 0.0

        pixels = [1.0] * 784
        tensor = preprocess_for_inference(pixels)
        assert tensor.min() == 1.0
        assert tensor.max() == 1.0


# ---------------------------------------------------------------------------
# Class Labels Tests
# ---------------------------------------------------------------------------

class TestClassLabels:
    """Tests for class label configuration."""

    def test_label_count(self):
        """Should have exactly 6 class labels."""
        assert len(CLASS_LABELS) == 6

    def test_label_contents(self):
        """Should contain the expected categories."""
        expected = ['airplane', 'car', 'cat', 'dog', 'house', 'tree']
        assert sorted(CLASS_LABELS) == sorted(expected)
