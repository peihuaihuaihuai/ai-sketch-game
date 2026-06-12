"""
Flask backend for the QuickDraw AI sketch recognition web app.

This application serves the trained PyTorch model via a REST API and
hosts the static frontend files.
"""

import os
import sys
import logging

import torch
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flask app initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Enable CORS for local development (restrict in production)
CORS(app, origins='*')


# ---------------------------------------------------------------------------
# PyTorch configuration
# ---------------------------------------------------------------------------

# Limit PyTorch to single-threaded CPU inference to avoid thread-pool
# contention when running inside Flask's request handler.
# NOTE: threaded=False in app.run() ensures single-request latency is
# predictable for local single-user use. For multi-user deployment,
# use a WSGI server like Gunicorn with multiple workers.
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Model loading (at import time)
# ---------------------------------------------------------------------------

# Import prediction module after torch config is set
try:
    import api.predict as predict_module
    model = predict_module.get_model()
    logger.info("Model loaded successfully from %s", predict_module.MODEL_PATH)
except FileNotFoundError as e:
    logger.error(str(e))
    sys.exit(1)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    sys.exit(1)


# Warm-up inference: measure and log first-prediction latency
# PyTorch lazy initialization makes the first inference slower;
# warming up ensures subsequent requests are fast.
try:
    warmup_ms = predict_module.warm_up_model()
    logger.info("Model warm-up complete: %.2f ms avg over 100 runs", warmup_ms)
except Exception as e:
    logger.warning("Model warm-up failed: %s", e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Prediction endpoint: accept a preprocessed sketch and return Top-5 predictions.

    Request Body (JSON):
        {
            "pixels": [0.0, 0.12, ..., 0.98]  // 784 floats in [0, 1]
        }

    Response (200 OK):
        {
            "top5": [
                {"label": "cat", "probability": 0.92},
                ...
            ],
            "latency_ms": 2.145,
            "model": "resnet"
        }

    Response (400 Bad Request): Invalid input
    Response (500 Internal Server Error): Model inference failure
    """
    # --- Client-side validation: return 400 for malformed requests ---

    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON body'}), 400

    if 'pixels' not in data:
        return jsonify({'error': "Missing required field: 'pixels'"}), 400

    pixels = data['pixels']
    strokes = data.get('strokes')

    # Validate input type and content before passing to model
    try:
        predict_module.validate_pixels(pixels)
        if strokes is not None:
            predict_module.validate_strokes(strokes)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    # --- Server-side inference ---
    try:
        result = predict_module.predict(pixels, strokes)
        return jsonify(result)
    except Exception as e:
        logger.exception("Model inference failed")
        return jsonify({
            'error': 'Model inference failed',
            'detail': str(e),
        }), 500


@app.route('/repredict', methods=['POST'])
def repredict_endpoint():
    """
    Re-prediction endpoint: accept a sketch and excluded labels,
    return Top-5 predictions with excluded labels removed and probabilities re-normalized.

    Request Body (JSON):
        {
            "pixels": [0.0, 0.12, ..., 0.98],  // 784 floats in [0, 1]
            "strokes": [...],                     // optional stroke data
            "exclude_labels": ["cat"]            // labels to exclude
        }

    Response (200 OK):
        {
            "top5": [
                {"label": "dog", "probability": 0.65},
                ...
            ],
            "latency_ms": 2.145,
            "model": "resnet",
            "excluded": ["cat"]
        }
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON body'}), 400

    if 'pixels' not in data:
        return jsonify({'error': "Missing required field: 'pixels'"}), 400

    pixels = data['pixels']
    strokes = data.get('strokes')
    exclude_labels = data.get('exclude_labels', [])

    # Validate input
    try:
        predict_module.validate_pixels(pixels)
        if strokes is not None:
            predict_module.validate_strokes(strokes)
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    try:
        result = predict_module.repredict(pixels, strokes, exclude_labels)
        return jsonify(result)
    except Exception as e:
        logger.exception("Re-prediction failed")
        return jsonify({
            'error': 'Re-prediction failed',
            'detail': str(e),
        }), 500


@app.route('/reset', methods=['POST'])
def reset_endpoint():
    """
    Reset endpoint: clear server-side prediction state.

    Response (200 OK):
        {
            "status": "reset",
            "message": "Prediction state cleared successfully"
        }
    """
    try:
        result = predict_module.reset_state()
        return jsonify(result)
    except Exception as e:
        logger.exception("Reset failed")
        return jsonify({
            'error': 'Reset failed',
            'detail': str(e),
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint for monitoring.

    Response (200 OK):
        {
            "status": "ok",
            "model_loaded": true,
            "model_name": "resnet",
            "predictions_served": 42
        }
    """
    try:
        stats = predict_module.get_stats()
        return jsonify({
            'status': 'ok',
            'model_loaded': model is not None,
            **stats,
        })
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({
            'status': 'error',
            'detail': str(e),
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Hugging Face Spaces / cloud platforms set $PORT dynamically;
    # default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Flask server on http://0.0.0.0:%d", port)
    app.run(host='0.0.0.0', port=port, threaded=False)
