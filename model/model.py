"""
PyTorch model architectures for QuickDraw sketch classification.

This module provides two model options:
  1. QuickDrawCNN  - Baseline CNN (legacy, Phase 1)
  2. QuickDrawResNet - Lightweight ResNet with residual blocks (Phase 2, recommended)

Both models accept (batch, 1, 28, 28) grayscale images and output
(batch, num_classes) logits.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline CNN (Phase 1)
# ---------------------------------------------------------------------------

class QuickDrawCNN(nn.Module):
    """
    Convolutional Neural Network for QuickDraw sketch classification.

    Architecture:
        - 2 Convolutional blocks (Conv2d + ReLU + MaxPool2d)
        - Flatten layer
        - 2 Fully connected layers (with Dropout regularization)

    Input:  (batch_size, 1, 28, 28) grayscale images normalized to [0, 1]
    Output: (batch_size, num_classes) logits for each class

    Expected model size: ~1.5MB, ~400K parameters
    """

    def __init__(self, num_classes: int = 6) -> None:
        super(QuickDrawCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Lightweight residual block with optional downsampling.

    Architecture:
        Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> (+ shortcut) -> ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (1 = same size, 2 = downsample)
        dropout: Dropout probability after the block (0 = no dropout)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)

        return out


# ---------------------------------------------------------------------------
# Lightweight ResNet (Phase 2)
# ---------------------------------------------------------------------------

class QuickDrawResNet(nn.Module):
    """
    Lightweight ResNet for QuickDraw sketch classification.

    Uses residual blocks with BatchNorm for better gradient flow and
    faster convergence than the baseline CNN. Designed for 28x28 grayscale
    images with ~150K parameters for fast inference.

    Architecture:
        Conv1 (1->32) -> ResBlock(32,32) -> ResBlock(32,64,stride=2)
        -> ResBlock(64,128,stride=2) -> GlobalAvgPool -> Dropout -> FC

    Input:  (batch_size, 1, 28, 28) grayscale images normalized to [0, 1]
    Output: (batch_size, num_classes) logits for each class

    Expected model size: ~600KB, ~150K parameters
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super(QuickDrawResNet, self).__init__()

        # Initial convolution: 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        # 28x28 -> 28x28
        self.layer1 = ResidualBlock(32, 32, stride=1, dropout=0.0)
        # 28x28 -> 14x14
        self.layer2 = ResidualBlock(32, 64, stride=2, dropout=dropout)
        # 14x14 -> 7x7
        self.layer3 = ResidualBlock(64, 128, stride=2, dropout=dropout)

        # Global average pooling: (B, 128, 7, 7) -> (B, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """He initialization for conv layers, zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'cnn': QuickDrawCNN,
    'resnet': QuickDrawResNet,
    'hybrid': None,  # Imported lazily from stroke_graph to avoid circular deps
}


def create_model(
    model_name: str = 'resnet',
    num_classes: int = 6,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: 'cnn' or 'resnet'
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_model_summary(model: nn.Module) -> dict:
    """
    Compute model statistics (parameter count and size estimate).

    Args:
        model: The neural network model

    Returns:
        Dictionary with 'total_params', 'trainable_params', and 'size_mb' keys
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': round(size_mb, 2),
    }


def count_flops(model: nn.Module, input_shape: tuple = (1, 1, 28, 28)) -> int:
    """
    Estimate FLOPs for a single forward pass.
    Simple estimation: 2 * params for conv/linear layers.

    Args:
        model: The neural network model
        input_shape: Input tensor shape

    Returns:
        Estimated FLOPs count
    """
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * K^2 * Hout * Wout
            # Simplified: 2 * output_channels * kernel_size^2 * input_channels * output_spatial
            flops += 2 * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
    # Rough spatial scaling factor for 28x28
    return flops * 28 * 28 // 4  # approximate average spatial size


if __name__ == '__main__':
    # Compare both models
    print("=" * 50)
    print("Model Comparison")
    print("=" * 50)

    for name, cls in MODEL_REGISTRY.items():
        model = cls(num_classes=6)
        stats = get_model_summary(model)
        print(f"\n{name.upper()}:")
        print(f"  Parameters: {stats['total_params']:,} ({stats['trainable_params']:,} trainable)")
        print(f"  Size: {stats['size_mb']} MB")

        # Test forward pass
        dummy_input = torch.zeros(2, 1, 28, 28)
        output = model(dummy_input)
        print(f"  Input:  {dummy_input.shape}")
        print(f"  Output: {output.shape}")

        # Test inference latency
        import time
        model.eval()
        with torch.no_grad():
            # Warm up
            for _ in range(10):
                _ = model(dummy_input[:1])
            # Measure
            start = time.perf_counter()
            for _ in range(100):
                _ = model(dummy_input[:1])
            elapsed_ms = (time.perf_counter() - start) * 1000 / 100
        print(f"  Avg inference latency: {elapsed_ms:.3f} ms")
