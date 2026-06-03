"""
Stroke graph construction and Graph Neural Network for sketch classification.

Provides:
  - build_graph_from_strokes: converts frontend stroke sequences to a graph
  - StrokeGNN: lightweight GNN in pure PyTorch (no PyTorch Geometric required)
  - HybridQuickDrawModel: CNN + GNN hybrid that fuses bitmap and stroke features
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph_from_strokes(
    strokes: List[List[dict]],
    spatial_threshold: float = 0.15,
    max_nodes: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Build a graph from normalized stroke sequences.

    Args:
        strokes: List of strokes, each stroke is a list of {x, y} dicts with
                 values in [0, 1].
        spatial_threshold: Distance threshold for adding spatial edges between
                           points from different strokes (in normalized space).
        max_nodes: Maximum number of nodes to keep per graph. If exceeded,
                   strokes are downsampled uniformly.

    Returns:
        Tuple of (node_features, edge_index, edge_features, batch_index):
          - node_features: (N, 5) tensor [x, y, is_start, is_end, stroke_idx_norm]
          - edge_index: (2, E) tensor of edge connections
          - edge_features: (E, 2) tensor [is_sequential, is_spatial]
          - batch_index: None for single graph (or (N,) for batched)

    Returns (None, None, None, None) if strokes is empty.
    """
    if not strokes:
        return None, None, None, None

    # Flatten strokes into nodes with metadata
    nodes = []  # list of (x, y, stroke_idx, is_start, is_end)
    for stroke_idx, stroke in enumerate(strokes):
        if not stroke:
            continue
        n_points = len(stroke)
        for i, pt in enumerate(stroke):
            is_start = 1.0 if i == 0 else 0.0
            is_end = 1.0 if i == n_points - 1 else 0.0
            nodes.append((float(pt["x"]), float(pt["y"]), stroke_idx, is_start, is_end))

    if not nodes:
        return None, None, None, None

    # Downsample if too many nodes
    if len(nodes) > max_nodes:
        indices = np.linspace(0, len(nodes) - 1, max_nodes, dtype=int)
        nodes = [nodes[i] for i in indices]

    num_strokes = max(s_idx for (_, _, s_idx, _, _) in nodes) + 1
    n_nodes = len(nodes)

    # Build node features: (N, 5)
    node_feats = torch.zeros(n_nodes, 5, dtype=torch.float32)
    for i, (x, y, s_idx, is_start, is_end) in enumerate(nodes):
        node_feats[i, 0] = x
        node_feats[i, 1] = y
        node_feats[i, 2] = is_start
        node_feats[i, 3] = is_end
        node_feats[i, 4] = s_idx / max(num_strokes, 1)

    # Build edges
    edge_src = []
    edge_dst = []
    edge_feats = []

    # Sequential edges: connect consecutive points within each stroke
    point_idx = 0
    for stroke in strokes:
        if not stroke:
            continue
        n_points = len(stroke)
        # Account for downsampling - find which original points survived
        # For simplicity, we rebuild mapping
        actual_len = min(n_points, max_nodes // max(num_strokes, 1) + 1)
        # Just connect consecutive nodes that belong to the same stroke
        # We need to know which nodes belong to which stroke
        # Since we may have downsampled, this gets tricky.
        # Simpler approach: rebuild stroke membership from nodes list
        point_idx += n_points

    # Rebuild: group nodes by stroke
    stroke_nodes = {}
    for i, (_, _, s_idx, _, _) in enumerate(nodes):
        stroke_nodes.setdefault(s_idx, []).append(i)

    for s_idx, idxs in stroke_nodes.items():
        for j in range(len(idxs) - 1):
            u = idxs[j]
            v = idxs[j + 1]
            edge_src.append(u)
            edge_dst.append(v)
            edge_feats.append([1.0, 0.0])  # sequential
            # Undirected
            edge_src.append(v)
            edge_dst.append(u)
            edge_feats.append([1.0, 0.0])

    # Spatial edges: connect nearby points from different strokes
    coords = node_feats[:, :2].numpy()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Only between different strokes
            si = nodes[i][2]
            sj = nodes[j][2]
            if si == sj:
                continue
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < spatial_threshold:
                edge_src.append(i)
                edge_dst.append(j)
                edge_feats.append([0.0, 1.0])  # spatial
                edge_src.append(j)
                edge_dst.append(i)
                edge_feats.append([0.0, 1.0])

    if not edge_src:
        # Fallback: connect all nodes to prevent empty graph
        if n_nodes == 1:
            # Self-loop for single node
            edge_src.append(0)
            edge_dst.append(0)
            edge_feats.append([0.0, 0.0])
        else:
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        edge_feats.append([0.0, 0.0])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_features = torch.tensor(edge_feats, dtype=torch.float32)

    return node_feats, edge_index, edge_features, None


# ---------------------------------------------------------------------------
# Graph Neural Network (pure PyTorch)
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """
    Simple graph convolution layer: aggregate neighbor features and combine with self.

    message = MLP([h_u || h_v || e_uv])
    h_u' = MLP([h_u || aggregate(messages)])
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (N, in_dim)
            edge_index: (2, E)
            edge_feats: (E, edge_dim) or None

        Returns:
            Updated node features: (N, out_dim)
        """
        src, dst = edge_index
        h_src = node_feats[src]  # (E, in_dim)
        h_dst = node_feats[dst]  # (E, in_dim)

        if edge_feats is not None:
            msg_input = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        else:
            # Pad with zeros to match expected edge_dim
            E = src.size(0)
            zero_pad = torch.zeros(E, self.message_mlp[0].in_features - h_src.size(1) - h_dst.size(1),
                                    device=node_feats.device, dtype=node_feats.dtype)
            msg_input = torch.cat([h_src, h_dst, zero_pad], dim=-1)

        messages = self.message_mlp(msg_input)  # (E, out_dim)

        # Aggregate messages per destination node (mean aggregation)
        N = node_feats.size(0)
        agg = torch.zeros(N, messages.size(1), device=node_feats.device, dtype=node_feats.dtype)
        counts = torch.zeros(N, 1, device=node_feats.device, dtype=node_feats.dtype)

        agg.index_add_(0, dst, messages)
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32).unsqueeze(1))
        agg = agg / (counts + 1e-8)

        update_input = torch.cat([node_feats, agg], dim=-1)
        updated = self.update_mlp(update_input)
        return self.norm(updated)


class StrokeGNN(nn.Module):
    """
    Lightweight Graph Neural Network for stroke graphs.

    Processes node features through 3 graph convolution layers,
    then produces a graph-level embedding via global mean + max pooling.
    """

    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_dim, hidden_dim, edge_dim, dropout)
            )

        # Graph-level pooling: concatenate mean and max
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (N, node_dim)
            edge_index: (2, E)
            edge_feats: (E, edge_dim) or None

        Returns:
            Graph embedding: (output_dim,)
        """
        h = self.input_proj(node_feats)

        for conv in self.convs:
            h_new = conv(h, edge_index, edge_feats)
            h = h + h_new  # residual connection

        # Global pooling
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0)[0]
        h_pool = torch.cat([h_mean, h_max], dim=-1)

        return self.output_proj(h_pool)


# ---------------------------------------------------------------------------
# GNN-Only Classifier (Baseline)
# ---------------------------------------------------------------------------

class SketchGNNClassifier(nn.Module):
    """
    Standalone GNN classifier for stroke graphs (no CNN).

    This is the baseline model that processes only stroke topology
    and geometry to classify sketches.

    Architecture:
        Input projection -> 3 GraphConv layers (residual) -> Global pool -> MLP classifier
    """

    def __init__(
        self,
        num_classes: int = 6,
        node_dim: int = 5,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_dim, hidden_dim, edge_dim, dropout)
            )

        # Classifier head after global pooling
        pool_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (N, node_dim)
            edge_index: (2, E)
            edge_feats: (E, edge_dim) or None

        Returns:
            Logits: (num_classes,)
        """
        h = self.input_proj(node_feats)

        for conv in self.convs:
            h_new = conv(h, edge_index, edge_feats)
            h = h + h_new  # residual

        # Global pooling
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0)[0]
        h_pool = torch.cat([h_mean, h_max], dim=-1)

        return self.classifier(h_pool)


# ---------------------------------------------------------------------------
# CNN / GNN Hybrid Model
# ---------------------------------------------------------------------------

class HybridQuickDrawModel(nn.Module):
    """
    CNN + GNN hybrid model for sketch classification.

    The CNN branch processes the 28x28 bitmap (same as QuickDrawResNet).
    The GNN branch processes the stroke graph.
    Embeddings are concatenated and fed into a joint classifier.

    When strokes are not available, the GNN branch outputs zeros and the
    model falls back to CNN-only inference.
    """

    def __init__(
        self,
        num_classes: int = 6,
        cnn_backbone: Optional[nn.Module] = None,
        gnn_output_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # CNN branch: either provided or built inline
        if cnn_backbone is not None:
            self.cnn = cnn_backbone
        else:
            from model.model import QuickDrawResNet
            self.cnn = QuickDrawResNet(num_classes=num_classes, dropout=dropout)

        # Extract CNN embedding dimension by inspecting the classifier
        # QuickDrawResNet: classifier is Sequential(Flatten, Dropout, Linear(128, num_classes))
        cnn_embed_dim = 128

        # Replace CNN classifier with an embedding head
        self._replace_cnn_classifier(cnn_embed_dim, dropout)

        # GNN branch
        self.gnn = StrokeGNN(output_dim=gnn_output_dim, dropout=dropout)

        # Joint classifier
        self.joint_classifier = nn.Sequential(
            nn.Linear(cnn_embed_dim + gnn_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _replace_cnn_classifier(self, embed_dim: int, dropout: float) -> None:
        """Replace the CNN's final classifier with an embedding projection."""
        # Find the flatten + classifier layers in the CNN
        # We assume the CNN has a .global_pool and .classifier
        if hasattr(self.cnn, 'classifier'):
            self.cnn.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError("CNN backbone must have a .classifier attribute")

    def forward(
        self,
        image: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image: (B, 1, 28, 28) bitmap tensor
            node_feats: (N, 5) or None
            edge_index: (2, E) or None
            edge_feats: (E, 2) or None

        Returns:
            Logits: (B, num_classes)
        """
        batch_size = image.size(0)

        # CNN embedding
        cnn_embed = self._cnn_forward(image)  # (B, embed_dim)

        # GNN embedding (one per batch item; for now assume batch_size == 1)
        if node_feats is not None and edge_index is not None:
            gnn_embed = self.gnn(node_feats, edge_index, edge_feats)  # (gnn_output_dim,)
            gnn_embed = gnn_embed.unsqueeze(0).expand(batch_size, -1)  # (B, gnn_output_dim)
        else:
            gnn_embed = torch.zeros(batch_size, self.gnn.output_proj[0].out_features,
                                     device=image.device, dtype=image.dtype)

        # Concatenate and classify
        joint = torch.cat([cnn_embed, gnn_embed], dim=-1)
        logits = self.joint_classifier(joint)
        return logits

    def _cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through CNN backbone up to the embedding."""
        x = self.cnn.conv1(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.global_pool(x)
        x = self.cnn.classifier(x)
        return x

    def forward_cnn_only(self, image: torch.Tensor) -> torch.Tensor:
        """Fallback CNN-only forward for when strokes are unavailable."""
        return self.forward(image, None, None, None)


# ---------------------------------------------------------------------------
# Utility: batch graph construction for training
# ---------------------------------------------------------------------------

def collate_graph_batch(
    batch_strokes: List[List[List[dict]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Build a batched graph from a list of stroke sets.

    Args:
        batch_strokes: List of stroke sequences (one per batch item)

    Returns:
        (node_feats, edge_index, edge_feats, batch_vector)
        where batch_vector[i] gives the batch index of node i.
        Returns (None, None, None, None) if all stroke sets are empty.
    """
    all_nodes = []
    all_edges_src = []
    all_edges_dst = []
    all_edge_feats = []
    batch_vector = []

    node_offset = 0

    for batch_idx, strokes in enumerate(batch_strokes):
        node_feats, edge_index, edge_feats, _ = build_graph_from_strokes(strokes)
        if node_feats is None:
            continue

        n_nodes = node_feats.size(0)
        all_nodes.append(node_feats)
        batch_vector.extend([batch_idx] * n_nodes)

        if edge_index is not None:
            all_edges_src.append(edge_index[0] + node_offset)
            all_edges_dst.append(edge_index[1] + node_offset)
        if edge_feats is not None:
            all_edge_feats.append(edge_feats)

        node_offset += n_nodes

    if not all_nodes:
        return None, None, None, torch.empty(0, dtype=torch.long)

    node_feats = torch.cat(all_nodes, dim=0)
    edge_index = torch.stack([
        torch.cat(all_edges_src, dim=0),
        torch.cat(all_edges_dst, dim=0),
    ], dim=0)
    edge_feats = torch.cat(all_edge_feats, dim=0) if all_edge_feats else None
    batch_vector = torch.tensor(batch_vector, dtype=torch.long)

    return node_feats, edge_index, edge_feats, batch_vector


if __name__ == "__main__":
    # Quick sanity check
    strokes = [
        [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}, {"x": 0.3, "y": 0.1}],
        [{"x": 0.2, "y": 0.3}, {"x": 0.25, "y": 0.35}],
    ]
    nf, ei, ef, _ = build_graph_from_strokes(strokes)
    print(f"Nodes: {nf.shape}, Edges: {ei.shape}, Edge feats: {ef.shape}")

    gnn = StrokeGNN()
    embed = gnn(nf, ei, ef)
    print(f"GNN embed: {embed.shape}")

    # Test hybrid model
    hybrid = HybridQuickDrawModel(num_classes=6)
    img = torch.zeros(1, 1, 28, 28)
    logits = hybrid(img, nf, ei, ef)
    print(f"Hybrid logits: {logits.shape}")

    logits_cnn_only = hybrid.forward_cnn_only(img)
    print(f"CNN-only logits: {logits_cnn_only.shape}")
