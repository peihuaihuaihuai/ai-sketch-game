"""
Unit tests for the QuickDraw model architectures.

Verifies model architecture, output shapes, and inference behavior
for both CNN (baseline) and ResNet (Phase 2).
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import QuickDrawCNN, QuickDrawResNet, create_model, get_model_summary


# ---------------------------------------------------------------------------
# QuickDrawCNN Tests (Baseline)
# ---------------------------------------------------------------------------

class TestQuickDrawCNN:
    """Tests for the baseline CNN model."""

    def test_model_instantiation(self):
        model = QuickDrawCNN(num_classes=6)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_output_shape(self):
        model = QuickDrawCNN(num_classes=6)
        batch_size = 4
        dummy_input = torch.zeros(batch_size, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (batch_size, 6)

    def test_model_single_inference(self):
        model = QuickDrawCNN(num_classes=6)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 28, 28))
        assert output.shape == (1, 6)
        assert torch.isfinite(output).all()

    def test_model_softmax_sum(self):
        model = QuickDrawCNN(num_classes=6)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, 28, 28))
            probs = torch.softmax(logits, dim=1)
        assert probs.shape == (1, 6)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_parameter_count(self):
        model = QuickDrawCNN(num_classes=6)
        stats = get_model_summary(model)
        assert stats['total_params'] < 1_000_000
        assert stats['size_mb'] < 5.0

    def test_model_cpu_inference(self):
        model = QuickDrawCNN(num_classes=6)
        model.eval()
        input_tensor = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            output = model(input_tensor)
        assert output.shape == (2, 6)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# QuickDrawResNet Tests (Phase 2)
# ---------------------------------------------------------------------------

class TestQuickDrawResNet:
    """Tests for the lightweight ResNet model."""

    def test_model_instantiation(self):
        model = QuickDrawResNet(num_classes=6)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_output_shape(self):
        model = QuickDrawResNet(num_classes=6)
        batch_size = 4
        dummy_input = torch.zeros(batch_size, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (batch_size, 6)

    def test_model_single_inference(self):
        model = QuickDrawResNet(num_classes=6)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 28, 28))
        assert output.shape == (1, 6)
        assert torch.isfinite(output).all()

    def test_model_softmax_sum(self):
        model = QuickDrawResNet(num_classes=6)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, 28, 28))
            probs = torch.softmax(logits, dim=1)
        assert probs.shape == (1, 6)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_parameter_count(self):
        model = QuickDrawResNet(num_classes=6)
        stats = get_model_summary(model)
        # ResNet should have fewer parameters than CNN
        assert stats['total_params'] < 500_000
        assert stats['size_mb'] < 2.0

    def test_model_cpu_inference(self):
        model = QuickDrawResNet(num_classes=6)
        model.eval()
        input_tensor = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            output = model(input_tensor)
        assert output.shape == (2, 6)
        assert torch.isfinite(output).all()

    def test_residual_block_preserves_shape(self):
        """Test that a ResBlock with stride=1 preserves spatial dimensions."""
        from model.model import ResidualBlock
        block = ResidualBlock(32, 32, stride=1)
        x = torch.randn(1, 32, 28, 28)
        out = block(x)
        assert out.shape == (1, 32, 28, 28)

    def test_residual_block_downsamples(self):
        """Test that a ResBlock with stride=2 halves spatial dimensions."""
        from model.model import ResidualBlock
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(1, 32, 28, 28)
        out = block(x)
        assert out.shape == (1, 64, 14, 14)


# ---------------------------------------------------------------------------
# Model Factory Tests
# ---------------------------------------------------------------------------

class TestModelFactory:
    """Tests for the model factory function."""

    def test_create_cnn(self):
        model = create_model('cnn', num_classes=6)
        assert isinstance(model, QuickDrawCNN)

    def test_create_resnet(self):
        model = create_model('resnet', num_classes=6)
        assert isinstance(model, QuickDrawResNet)

    def test_create_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model('transformer', num_classes=6)

    def test_create_case_insensitive(self):
        model = create_model('ReSnEt', num_classes=6)
        assert isinstance(model, QuickDrawResNet)


# ---------------------------------------------------------------------------
# Multi-class Tests
# ---------------------------------------------------------------------------

class TestVariableClasses:
    """Test models with different numbers of output classes."""

    @pytest.mark.parametrize("num_classes", [2, 6, 10, 100])
    def test_cnn_variable_classes(self, num_classes):
        model = QuickDrawCNN(num_classes=num_classes)
        output = model(torch.zeros(1, 1, 28, 28))
        assert output.shape == (1, num_classes)

    @pytest.mark.parametrize("num_classes", [2, 6, 10, 100])
    def test_resnet_variable_classes(self, num_classes):
        model = QuickDrawResNet(num_classes=num_classes)
        output = model(torch.zeros(1, 1, 28, 28))
        assert output.shape == (1, num_classes)
