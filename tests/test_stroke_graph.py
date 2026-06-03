"""
Unit tests for stroke graph construction and GNN models.

Verifies:
  - Graph construction from stroke sequences
  - GNN-only classifier (SketchGNNClassifier)
  - CNN-GNN hybrid model (HybridQuickDrawModel)
  - Edge cases (empty strokes, single stroke, etc.)
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.stroke_graph import (
    build_graph_from_strokes,
    GraphConvLayer,
    StrokeGNN,
    SketchGNNClassifier,
    HybridQuickDrawModel,
    collate_graph_batch,
)


# ---------------------------------------------------------------------------
# Graph Construction Tests
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    """Tests for build_graph_from_strokes."""

    def test_basic_strokes(self):
        strokes = [
            [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
            [{"x": 0.3, "y": 0.3}, {"x": 0.4, "y": 0.4}],
        ]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        assert nf is not None
        assert nf.shape[1] == 5  # 5 node features
        assert ei.shape[0] == 2  # (2, E)
        assert ef.shape[1] == 2  # 2 edge features
        assert ef.shape[0] == ei.shape[1]

    def test_empty_strokes(self):
        nf, ei, ef, _ = build_graph_from_strokes([])
        assert nf is None

    def test_single_stroke(self):
        strokes = [[{"x": 0.5, "y": 0.5}]]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        assert nf is not None
        assert nf.shape[0] == 1

    def test_node_features_range(self):
        strokes = [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]]
        nf, _, _, _ = build_graph_from_strokes(strokes)
        # x, y should be in [0, 1]
        assert (nf[:, 0] >= 0.0).all() and (nf[:, 0] <= 1.0).all()
        assert (nf[:, 1] >= 0.0).all() and (nf[:, 1] <= 1.0).all()
        # is_start, is_end should be 0 or 1
        assert (nf[:, 2] >= 0.0).all() and (nf[:, 2] <= 1.0).all()
        assert (nf[:, 3] >= 0.0).all() and (nf[:, 3] <= 1.0).all()

    def test_downsampling(self):
        """Test that max_nodes limit is respected."""
        strokes = [[{"x": i / 200, "y": i / 200}] for i in range(200)]
        nf, _, _, _ = build_graph_from_strokes(strokes, max_nodes=50)
        assert nf.shape[0] <= 50

    def test_spatial_edges(self):
        """Spatial edges should connect nearby points from different strokes."""
        strokes = [
            [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
            [{"x": 0.15, "y": 0.15}, {"x": 0.25, "y": 0.25}],
        ]
        _, ei, ef, _ = build_graph_from_strokes(strokes, spatial_threshold=0.1)
        # Should have some spatial edges (edge feature [0, 1])
        spatial_mask = (ef[:, 1] > 0.5)
        assert spatial_mask.sum().item() > 0


# ---------------------------------------------------------------------------
# Graph Convolution Layer Tests
# ---------------------------------------------------------------------------

class TestGraphConvLayer:
    """Tests for GraphConvLayer."""

    def test_forward_shape(self):
        layer = GraphConvLayer(in_dim=8, out_dim=16, edge_dim=2)
        node_feats = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_feats = torch.randn(4, 2)
        out = layer(node_feats, edge_index, edge_feats)
        assert out.shape == (5, 16)

    def test_forward_no_edge_feats(self):
        layer = GraphConvLayer(in_dim=8, out_dim=16)
        node_feats = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(node_feats, edge_index)
        assert out.shape == (5, 16)

    def test_residual_compatibility(self):
        """Output dimension must match input for residual connections."""
        layer = GraphConvLayer(in_dim=16, out_dim=16)
        node_feats = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = layer(node_feats, edge_index)
        assert out.shape == node_feats.shape


# ---------------------------------------------------------------------------
# StrokeGNN Tests
# ---------------------------------------------------------------------------

class TestStrokeGNN:
    """Tests for the StrokeGNN embedding model."""

    def test_embedding_shape(self):
        gnn = StrokeGNN(output_dim=64)
        strokes = [
            [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
            [{"x": 0.3, "y": 0.3}, {"x": 0.4, "y": 0.4}],
        ]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        embed = gnn(nf, ei, ef)
        assert embed.shape == (64,)

    def test_embedding_finite(self):
        gnn = StrokeGNN()
        strokes = [[{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}]]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        embed = gnn(nf, ei, ef)
        assert torch.isfinite(embed).all()


# ---------------------------------------------------------------------------
# SketchGNNClassifier Tests (Baseline)
# ---------------------------------------------------------------------------

class TestSketchGNNClassifier:
    """Tests for the GNN-only baseline classifier."""

    def test_output_shape(self):
        model = SketchGNNClassifier(num_classes=6)
        strokes = [
            [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],
            [{"x": 0.3, "y": 0.3}, {"x": 0.4, "y": 0.4}],
        ]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        logits = model(nf, ei, ef)
        assert logits.shape == (6,)

    def test_softmax_sum(self):
        model = SketchGNNClassifier(num_classes=6)
        model.eval()
        strokes = [[{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}]]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        with torch.no_grad():
            logits = model(nf, ei, ef)
            probs = torch.softmax(logits, dim=0)
        assert probs.shape == (6,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_variable_classes(self):
        for num_classes in [2, 6, 10]:
            model = SketchGNNClassifier(num_classes=num_classes)
            strokes = [[{"x": 0.5, "y": 0.5}]]
            nf, ei, ef, _ = build_graph_from_strokes(strokes)
            logits = model(nf, ei, ef)
            assert logits.shape == (num_classes,)

    def test_parameter_count(self):
        model = SketchGNNClassifier(num_classes=6)
        total = sum(p.numel() for p in model.parameters())
        assert total < 200_000  # Should be lightweight


# ---------------------------------------------------------------------------
# Hybrid Model Tests
# ---------------------------------------------------------------------------

class TestHybridQuickDrawModel:
    """Tests for the CNN+GNN hybrid model."""

    def test_hybrid_output_shape(self):
        from model.model import QuickDrawResNet
        cnn = QuickDrawResNet(num_classes=6)
        hybrid = HybridQuickDrawModel(num_classes=6, cnn_backbone=cnn)
        image = torch.zeros(1, 1, 28, 28)
        strokes = [[{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}]]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        logits = hybrid(image, nf, ei, ef)
        assert logits.shape == (1, 6)

    def test_cnn_only_fallback(self):
        from model.model import QuickDrawResNet
        cnn = QuickDrawResNet(num_classes=6)
        hybrid = HybridQuickDrawModel(num_classes=6, cnn_backbone=cnn)
        image = torch.zeros(2, 1, 28, 28)
        logits = hybrid.forward_cnn_only(image)
        assert logits.shape == (2, 6)

    def test_hybrid_finite(self):
        from model.model import QuickDrawResNet
        cnn = QuickDrawResNet(num_classes=6)
        hybrid = HybridQuickDrawModel(num_classes=6, cnn_backbone=cnn)
        hybrid.eval()
        image = torch.randn(1, 1, 28, 28)
        strokes = [[{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}]]
        nf, ei, ef, _ = build_graph_from_strokes(strokes)
        with torch.no_grad():
            logits = hybrid(image, nf, ei, ef)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Batch Collate Tests
# ---------------------------------------------------------------------------

class TestCollateGraphBatch:
    """Tests for collate_graph_batch."""

    def test_basic_batch(self):
        batch_strokes = [
            [[{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}]],
            [[{"x": 0.3, "y": 0.3}, {"x": 0.4, "y": 0.4}]],
        ]
        nf, ei, ef, bv = collate_graph_batch(batch_strokes)
        assert nf is not None
        assert nf.shape[0] > 0
        assert ei.shape[0] == 2
        assert bv.shape[0] == nf.shape[0]
        assert bv.max().item() == 1  # 2 items in batch

    def test_empty_batch(self):
        nf, ei, ef, bv = collate_graph_batch([])
        assert nf is None
