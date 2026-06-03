"""
Diagnostic script for CNN bias analysis.

Evaluates the trained model on:
  1. Real QuickDraw validation data (per-class accuracy + confusion matrix)
  2. Synthetic validation data (to detect overfitting)
  3. Frontend-preprocessed images vs training data statistics

Usage:
    python model/diagnostics.py
"""

import sys
import os
from pathlib import Path

# Fix GBK encoding errors on Windows consoles (chcp 65001 equivalent)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

from model.model import create_model
from model.dataset import load_quickdraw_data, generate_synthetic_data, DEFAULT_CLASS_NAMES
from model.preprocessing import normalize_image


def load_model_checkpoint(path: Path):
    """Load the trained model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "resnet")
        num_classes = checkpoint.get("num_classes", len(DEFAULT_CLASS_NAMES))
        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        val_acc = checkpoint.get("val_acc", 0.0)
    else:
        state_dict = checkpoint
        model_name = "cnn"
        num_classes = len(DEFAULT_CLASS_NAMES)
        class_names = DEFAULT_CLASS_NAMES
        val_acc = 0.0

    model = create_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, class_names, val_acc, model_name


def evaluate_on_data(model, images, labels, class_names, dataset_name="Data"):
    """Evaluate model on numpy image array and print diagnostics."""
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {len(images):,}")

    # Normalize images
    normalized = images.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(1)  # (N, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()

    labels_arr = np.array(labels)

    # Per-class accuracy
    print(f"\n{'Class':<12} {'Count':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        mask = labels_arr == i
        count = mask.sum()
        correct = (preds[mask] == i).sum()
        acc = correct / count if count > 0 else 0
        print(f"{name:<12} {count:<8} {correct:<8} {acc:<10.4f}")

    # Overall accuracy
    overall = (preds == labels_arr).mean()
    print(f"\nOverall accuracy: {overall:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds, labels=list(range(len(class_names))))
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = "{:>10}" * len(class_names)
    print(" " * 12 + header.format(*class_names))
    for i, name in enumerate(class_names):
        row = "{:>10}" * len(class_names)
        print(f"{name:<12}" + row.format(*cm[i]))

    # Class distribution in predictions
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(preds, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        pct = count / len(preds) * 100
        print(f"  {class_names[cls_idx]:<10}: {count:>6} ({pct:>5.1f}%)")

    # Bias detection: which classes are over-predicted?
    expected_pct = 100.0 / len(class_names)
    print(f"\nBias check (expected ~{expected_pct:.1f}% per class):")
    for cls_idx, count in zip(unique, counts):
        pct = count / len(preds) * 100
        deviation = pct - expected_pct
        flag = " <<< BIAS" if pct > expected_pct * 1.3 else ""
        print(f"  {class_names[cls_idx]:<10}: {pct:>5.1f}%  ({deviation:>+5.1f}%{flag})")

    return overall, cm, preds


def compare_preprocessing_statistics():
    """Compare statistics between QuickDraw data and synthetic data."""
    print(f"\n{'='*60}")
    print("Preprocessing Consistency Check")
    print(f"{'='*60}")

    data_dir = PROJECT_ROOT / "data" / "raw"

    # Load a few real QuickDraw images
    real_images = []
    for name in DEFAULT_CLASS_NAMES:
        filepath = data_dir / f"{name}.npy"
        if filepath.exists():
            try:
                raw = np.load(filepath)
                imgs = raw[:100].reshape(-1, 28, 28).astype(np.uint8)
                real_images.append(imgs)
            except Exception as e:
                print(f"  [{name}] Could not load: {e}")

    if real_images:
        real_all = np.concatenate(real_images, axis=0)
        print(f"\nReal QuickDraw stats ({len(real_all)} samples):")
        print(f"  Mean pixel value: {real_all.mean():.4f}")
        print(f"  Std pixel value:  {real_all.std():.4f}")
        print(f"  Non-zero pixels:  {(real_all > 0).mean() * 100:.2f}%")

    # Generate synthetic images
    synthetic_images, synthetic_labels = generate_synthetic_data(
        DEFAULT_CLASS_NAMES, samples_per_class=1000
    )
    print(f"\nSynthetic data stats ({len(synthetic_images)} samples):")
    print(f"  Mean pixel value: {synthetic_images.mean():.4f}")
    print(f"  Std pixel value:  {synthetic_images.std():.4f}")
    print(f"  Non-zero pixels:  {(synthetic_images > 0).mean() * 100:.2f}%")

    if real_images:
        print(f"\nDifference (Synthetic - Real):")
        print(f"  Mean diff: {synthetic_images.mean() - real_all.mean():.4f}")
        print(f"  Std diff:  {synthetic_images.std() - real_all.std():.4f}")


def diagnose_frontend_backend_mismatch():
    """Analyze potential preprocessing mismatches."""
    print(f"\n{'='*60}")
    print("Frontend-Backend Preprocessing Mismatch Analysis")
    print(f"{'='*60}")

    issues = []

    # Issue 1: DPR-dependent stroke thickness
    issues.append(
        "[FIXED] DPR-dependent stroke thickness: "
        "Frontend canvas was read at internal DPR-scaled resolution, "
        "causing Retina screens to produce thinner strokes in 28x28 space."
    )

    # Issue 2: Center of mass vs bbox centering
    issues.append(
        "[FIXED] Center-of-mass centering: "
        "QuickDraw uses center-of-mass centering, frontend was using bbox center. "
        "Fixed by computing pixel centroid and centering crop on it."
    )

    # Issue 3: Synthetic vs real data training
    issues.append(
        "[CRITICAL] Model trained on synthetic data: "
        "QuickDraw download failed (corrupted .npy). Model trained on 12K synthetic "
        "geometric shapes with 100% val accuracy in 1 epoch — severe overfitting."
    )

    # Issue 4: No augmentation matching frontend
    issues.append(
        "[WARNING] Frontend preprocessing includes bilinear anti-aliasing and "
        "dynamic bounding box extraction. Training augmentation (RandomRotation, "
        "RandomResizedCrop) may not fully cover real drawing variations."
    )

    for issue in issues:
        print(f"  • {issue}")


def simulate_frontend_pipeline():
    """
    Simulate old vs new frontend preprocessing to verify the ~96% → ~30% density fix.

    Draws sketch-like patterns at 400×400 virtual canvas resolution with 4px strokes
    (matching the new pen lineWidth), concentrated in a central region (~200×200),
    mimicking how users actually draw on the canvas.
    """
    print(f"\n{'='*60}")
    print("Frontend Pipeline Simulation (Old vs New)")
    print(f"{'='*60}")

    from PIL import Image, ImageDraw
    import math

    rng = np.random.RandomState(42)
    n_samples = 300
    results_old = []
    results_new = []

    # Sketch templates that produce more realistic stroke density
    def draw_sketch(draw, cx, cy, size, rng):
        """Draw a sketch-like pattern centered at (cx, cy) with given size."""
        # Draw 4-10 connected/overlapping strokes in a confined area
        n_strokes = rng.randint(4, 11)
        x, y = cx + rng.randint(-size, size), cy + rng.randint(-size, size)
        for _ in range(n_strokes):
            nx = x + rng.randint(-size//2, size//2)
            ny = y + rng.randint(-size//2, size//2)
            # Clamp to central region
            nx = max(50, min(350, nx))
            ny = max(50, min(350, ny))
            draw.line([(x, y), (nx, ny)], fill=0, width=4)
            x, y = nx, ny

    for _ in range(n_samples):
        img = Image.new('L', (400, 400), color=255)
        draw = ImageDraw.Draw(img)

        # Each sketch in a random central region
        cx = rng.randint(140, 260)
        cy = rng.randint(140, 260)
        size = rng.randint(40, 100)
        draw_sketch(draw, cx, cy, size, rng)

        # --- Bbox + COM in one pass ---
        arr = np.array(img)
        mask = arr < 250
        if mask.sum() < 20:  # skip nearly-empty
            continue
        ys, xs = mask.nonzero()
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        com_y = ys.mean()
        com_x = xs.mean()

        # --- Square crop centered on COM ---
        bbox_h = max_y - min_y + 1
        bbox_w = max_x - min_x + 1
        content = max(bbox_w, bbox_h)
        pad = max(1, int(content * 0.15))
        crop_size = content + pad * 2

        crop_src_x = int(com_x - crop_size / 2)
        crop_src_y = int(com_y - crop_size / 2)

        crop = Image.new('L', (crop_size, crop_size), color=255)
        sx = max(0, crop_src_x)
        sy = max(0, crop_src_y)
        box = (sx, sy, min(400, sx + crop_size), min(400, sy + crop_size))
        region = img.crop(box)
        dx = max(0, -crop_src_x)
        dy = max(0, -crop_src_y)
        crop.paste(region, (dx, dy))

        # --- OLD pipeline: BILINEAR, no binarization, black fill ---
        old_28 = Image.new('L', (28, 28), color=255)
        old_28.paste(crop.resize((20, 20), Image.BILINEAR), (4, 4))
        old_arr = (255 - np.array(old_28)).astype(np.float32) / 255.0
        results_old.append(old_arr)

        # --- NEW pipeline: NEAREST + BINARIZATION (thresh=80), white fill ---
        new_28 = Image.new('L', (28, 28), color=255)
        new_28.paste(crop.resize((20, 20), Image.NEAREST), (4, 4))
        new_arr = np.array(new_28)
        # Strict: dark pixels (≤80) → stroke (1.0), light pixels → bg (0.0)
        binary = (new_arr <= 80).astype(np.float32)
        results_new.append(binary)

    old_all = np.stack(results_old)
    new_all = np.stack(results_new)

    print(f"\n{'Metric':<30} {'Real QuickDraw':<18} {'OLD (bilinear)':<18} {'NEW (nearest+bin)':<18}")
    print("-" * 84)
    print(f"{'Mean pixel value':<30} {'45.18':<18} {old_all.mean():<18.4f} {new_all.mean():<18.4f}")
    print(f"{'Non-zero pixels %':<30} {'29.56%':<18} {(old_all > 0.01).mean()*100:<18.2f}% {(new_all > 0.01).mean()*100:<18.2f}%")

    # Bias test: run through the model
    print(f"\n--- Model prediction distribution (old vs new pipeline) ---")
    model, class_names, _, _ = load_model_checkpoint(
        PROJECT_ROOT / 'model' / 'quickdraw_cnn.pth'
    )

    for label, data in [("OLD (bilinear)", old_all), ("NEW (nearest+bin)", new_all)]:
        tensor = torch.from_numpy(data).unsqueeze(1)  # (N, 1, 28, 28)
        with torch.no_grad():
            logits = model(tensor)
            preds = torch.argmax(logits, dim=1).numpy()
        unique, counts = np.unique(preds, return_counts=True)
        print(f"\n  {label}:")
        for cls_idx, count in zip(unique, counts):
            pct = count / len(preds) * 100
            flag = " <<< BIAS" if pct > 40 else ""
            print(f"    {class_names[cls_idx]:<10}: {count:>4} ({pct:>5.1f}%){flag}")


def test_hybrid_end_to_end():
    """
    Verify the hybrid CNN+GNN inference pipeline end-to-end.

    Creates synthetic pixel + stroke data and tests:
      1. Graph construction (build_graph_from_strokes)
      2. Hybrid model forward pass
      3. CNN-only fallback when strokes are empty
    """
    print(f"\n{'='*60}")
    print("Hybrid CNN+GNN End-to-End Integration Test")
    print(f"{'='*60}")

    try:
        from model.stroke_graph import build_graph_from_strokes, HybridQuickDrawModel
        from model.model import create_model
    except ImportError as e:
        print(f"  SKIP: Hybrid modules not available ({e})")
        return

    # Synthetic stroke data simulating a simple sketch
    strokes = [
        [{"x": 0.2, "y": 0.3}, {"x": 0.3, "y": 0.35}, {"x": 0.4, "y": 0.32},
         {"x": 0.5, "y": 0.28}, {"x": 0.55, "y": 0.25}],
        [{"x": 0.3, "y": 0.5}, {"x": 0.35, "y": 0.55}, {"x": 0.4, "y": 0.6}],
    ]

    # Test 1: Graph construction
    node_feats, edge_index, edge_feats, _ = build_graph_from_strokes(strokes)
    assert node_feats is not None, "Graph construction failed!"
    assert node_feats.shape[1] == 5, f"Expected 5 node features, got {node_feats.shape[1]}"
    print(f"  [PASS] Graph construction: {node_feats.shape[0]} nodes, "
          f"{edge_index.shape[1]} edges")

    # Test 2: Empty strokes return None (graceful degradation)
    nf, ei, ef, _ = build_graph_from_strokes([])
    assert nf is None, "Empty strokes should return None"
    print("  [PASS] Empty strokes → None (graceful fallback)")

    # Test 3: Validate strokes function
    from api.predict import validate_strokes
    try:
        validate_strokes(strokes)
        print("  [PASS] validate_strokes() accepted valid input")
    except ValueError as e:
        print(f"  [FAIL] validate_strokes() rejected valid input: {e}")

    # Test invalid strokes
    for bad_input, desc in [
        ([{"x": 0.5}], "missing 'y'"),
        ([{"x": 1.5, "y": 0.5}], "x out of range"),
        ("not_a_list", "not a list"),
    ]:
        try:
            validate_strokes(bad_input)
            print(f"  [FAIL] validate_strokes() should reject: {desc}")
        except ValueError:
            print(f"  [PASS] validate_strokes() correctly rejected: {desc}")

    # Test 4: Hybrid model forward pass
    print("\n  --- Hybrid model forward pass ---")
    checkpoint_path = PROJECT_ROOT / 'model' / 'quickdraw_cnn.pth'
    if not checkpoint_path.exists():
        print("  SKIP: No checkpoint found")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_classes = checkpoint.get('num_classes', 6)
        base_name = checkpoint.get('model_name', 'resnet')
    else:
        num_classes = 6
        base_name = 'cnn'

    cnn_backbone = create_model(base_name, num_classes=num_classes)

    # Load CNN weights if available
    try:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        cnn_backbone.load_state_dict(state_dict, strict=True)
        print(f"  [INFO] CNN backbone weights loaded from checkpoint")
    except Exception as e:
        print(f"  [WARN] Could not load CNN weights: {e}")

    hybrid = HybridQuickDrawModel(num_classes=num_classes, cnn_backbone=cnn_backbone)
    hybrid.eval()

    # Forward with strokes
    img = torch.randn(1, 1, 28, 28)  # dummy pixel input
    with torch.no_grad():
        logits_hybrid = hybrid(img, node_feats, edge_index, edge_feats)
        logits_cnn = hybrid.forward_cnn_only(img)

    assert logits_hybrid.shape == (1, num_classes), \
        f"Hybrid output shape {logits_hybrid.shape} != (1, {num_classes})"
    assert logits_cnn.shape == (1, num_classes), \
        f"CNN-only output shape {logits_cnn.shape} != (1, {num_classes})"

    print(f"  [PASS] Hybrid forward:    shape={logits_hybrid.shape}")
    print(f"  [PASS] CNN-only fallback:  shape={logits_cnn.shape}")

    # Verify hybrid and CNN-only outputs differ (GNN branch contributes)
    diff = (logits_hybrid - logits_cnn).abs().max().item()
    print(f"  [INFO] Max |hybrid - cnn_only| = {diff:.6f} "
          f"({'GNN contributes' if diff > 1e-6 else 'GNN inactive (check weights)'})")

    print("\n  Hybrid pipeline summary:")
    print(f"    CNN embedding:  128D  (ResNet conv1→layer1-3→global_pool→embed)")
    print(f"    GNN embedding:   64D  (StrokeGNN: 3×GraphConv→mean+max pool→proj)")
    print(f"    Concatenation:  192D  (128 ⊕ 64)")
    print(f"    Joint MLP:      192D → 128 → {num_classes} classes")
    print(f"    Fallback:       CNN-only when strokes absent or graph build fails")
    MODEL_PATH = PROJECT_ROOT / "model" / "quickdraw_cnn.pth"
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    print("=" * 60)
    print("CNN Bias Diagnostic Report")
    print("=" * 60)

    # Load model
    model, class_names, val_acc, model_name = load_model_checkpoint(MODEL_PATH)
    print(f"\nLoaded model: {model_name}")
    print(f"Checkpoint val_acc: {val_acc:.4f}")
    print(f"Classes: {class_names}")

    # Evaluate on synthetic data
    syn_images, syn_labels = generate_synthetic_data(class_names, samples_per_class=500)
    evaluate_on_data(model, syn_images, syn_labels, class_names, "Synthetic Data")

    # Evaluate on real QuickDraw data
    try:
        real_images, real_labels = load_quickdraw_data(
            class_names, DATA_DIR, max_samples_per_class=2000, download=False
        )
        evaluate_on_data(model, real_images, real_labels, class_names, "Real QuickDraw Data")
    except Exception as e:
        print(f"\n[WARNING] Could not evaluate on real QuickDraw data: {e}")
        print("  This confirms the model was NOT trained on real data.")

    # Preprocessing statistics
    compare_preprocessing_statistics()

    # Mismatch analysis
    diagnose_frontend_backend_mismatch()

    # ── NEW: Frontend pipeline simulation (old vs new) ─────────────────
    simulate_frontend_pipeline()

    # ── NEW: Hybrid CNN+GNN end-to-end test ───────────────────────────
    test_hybrid_end_to_end()

    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}")
    print("  1. Re-download corrupted QuickDraw .npy files")
    print("  2. Retrain CNN on REAL QuickDraw data (not synthetic)")
    print("  3. Add stronger augmentation to match frontend variations")
    print("  4. Re-run diagnostics after retraining")
    print("=" * 60)


if __name__ == "__main__":
    main()
