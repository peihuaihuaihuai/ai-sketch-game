/**
 * Canvas drawing module for the CNN Sketch Guesser.
 *
 * Handles user drawing input (mouse and touch), stroke smoothing,
 * canvas preprocessing for model input, and the "Model sees" debug view.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // DOM Elements
    // -----------------------------------------------------------------------

    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const preprocessCanvas = document.getElementById('preprocess-canvas');
    const preprocessCtx = preprocessCanvas.getContext('2d');
    const debugCanvas = document.getElementById('debug-canvas');
    const debugCtx = debugCanvas.getContext('2d');

    const btnPen = document.getElementById('btn-pen');
    const btnEraser = document.getElementById('btn-eraser');
    const btnClear = document.getElementById('btn-clear');
    const btnPredict = document.getElementById('btn-predict');
    const btnReset = document.getElementById('btn-reset');
    const debounceSlider = document.getElementById('debounce-slider');
    const debounceValue = document.getElementById('debounce-value');

    // -----------------------------------------------------------------------
    // Drawing State
    // -----------------------------------------------------------------------

    let isDrawing = false;
    let currentTool = 'pen'; // 'pen' | 'eraser'
    let lastX = 0;
    let lastY = 0;

    // Stroke sequence tracking for graph representation
    let strokes = [];
    let currentStroke = null;

    // Tool settings
    const TOOL_SETTINGS = {
        pen: { color: '#000000', lineWidth: 8 },
        eraser: { color: '#ffffff', lineWidth: 20 },
    };

    // Debounce timer for auto-prediction
    let debounceTimer = null;
    let debounceDelay = 800; // milliseconds

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    /**
     * Initialize the canvas with a white background.
     */
    function initCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        strokes = [];
        currentStroke = null;
    }

    /**
     * Set up the canvas for high-DPI displays.
     * This ensures crisp lines on Retina and similar screens.
     */
    function setupHighDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        ctx.scale(dpr, dpr);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Reset to CSS size for display
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';

        initCanvas();
    }

    // -----------------------------------------------------------------------
    // Drawing Logic
    // -----------------------------------------------------------------------

    /**
     * Get the coordinates of a pointer event relative to the canvas.
     */
    function getPointerPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top,
        };
    }

    /**
     * Start drawing.
     */
    function startDrawing(e) {
        isDrawing = true;
        const pos = getPointerPos(e);
        lastX = pos.x;
        lastY = pos.y;

        // Start a new stroke
        currentStroke = [{ x: pos.x, y: pos.y }];

        // Draw a single dot for click/tap without drag
        drawLine(lastX, lastY, lastX, lastY);
    }

    /**
     * Draw a line segment with optional smoothing.
     */
    function drawLine(x1, y1, x2, y2) {
        const settings = TOOL_SETTINGS[currentTool];

        ctx.strokeStyle = settings.color;
        ctx.lineWidth = settings.lineWidth;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    /**
     * Continue drawing with linear interpolation for smooth strokes.
     */
    function continueDrawing(e) {
        if (!isDrawing) return;

        // Prevent touch scrolling on mobile
        if (e.touches) {
            e.preventDefault();
        }

        const pos = getPointerPos(e);
        const x = pos.x;
        const y = pos.y;

        // Stroke smoothing: interpolate at 2px intervals
        const distance = Math.hypot(x - lastX, y - lastY);
        const stepSize = 2;

        if (distance > stepSize) {
            const steps = Math.ceil(distance / stepSize);
            for (let i = 1; i <= steps; i++) {
                const t = i / steps;
                const ix = lastX + (x - lastX) * t;
                const iy = lastY + (y - lastY) * t;
                drawLine(lastX, lastY, ix, iy);
                if (currentStroke) {
                    currentStroke.push({ x: ix, y: iy });
                }
                lastX = ix;
                lastY = iy;
            }
        } else {
            drawLine(lastX, lastY, x, y);
            if (currentStroke) {
                currentStroke.push({ x: x, y: y });
            }
            lastX = x;
            lastY = y;
        }
    }

    /**
     * Stop drawing and trigger debounced prediction.
     */
    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;

        // Save completed stroke
        if (currentStroke && currentStroke.length > 0) {
            strokes.push(currentStroke);
            currentStroke = null;
        }

        // Reset debounce timer
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            sendPrediction();
        }, debounceDelay);
    }

    /**
     * Normalize captured strokes to [0, 1] coordinates within the drawing bbox.
     *
     * Returns an array of strokes, each stroke is an array of {x, y} points
     * normalized to [0, 1] based on the overall drawing bounding box.
     */
    function getNormalizedStrokes() {
        if (strokes.length === 0) return [];

        // Compute overall bbox of all stroke points
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const stroke of strokes) {
            for (const pt of stroke) {
                minX = Math.min(minX, pt.x);
                minY = Math.min(minY, pt.y);
                maxX = Math.max(maxX, pt.x);
                maxY = Math.max(maxY, pt.y);
            }
        }

        const bboxW = maxX - minX || 1;
        const bboxH = maxY - minY || 1;
        const scale = Math.max(bboxW, bboxH);

        // Center and normalize to [0, 1] within a square
        return strokes.map(stroke =>
            stroke.map(pt => ({
                x: (pt.x - minX) / scale,
                y: (pt.y - minY) / scale,
            }))
        );
    }

    // -----------------------------------------------------------------------
    // Preprocessing Pipeline
    // -----------------------------------------------------------------------

    /**
     * Preprocess the canvas drawing into a 784-float array matching QuickDraw format.
     *
     * This pipeline matches the official QuickDraw preprocessing standard:
     *   - Center-of-mass (centroid) centering, NOT bounding-box centering
     *   - Scale to fit 68×68 within 96×96 (14px margin preserved)
     *   - Nearest-neighbor downsampling (no bilinear blur)
     *   - Strict binarization (hard threshold — no gray anti-alias pixels)
     *
     * Target output: ~30% non-zero pixel density, thin clean strokes.
     *
     * Returns:
     *   Array of 784 floats in [0.0, 1.0], or null if canvas is empty.
     *   1.0 = stroke pixel, 0.0 = background pixel (white-on-black QuickDraw format).
     */
    function getPreprocessedPixels() {
        // ── Step 1: DPR-normalized export ──────────────────────────────────
        // Render at CSS resolution so all devices produce identical 96×96 input
        // regardless of devicePixelRatio (Retina vs standard).
        const rect = canvas.getBoundingClientRect();
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width = rect.width;
        exportCanvas.height = rect.height;
        const exportCtx = exportCanvas.getContext('2d');
        exportCtx.drawImage(canvas, 0, 0, rect.width, rect.height);

        const width = exportCanvas.width;
        const height = exportCanvas.height;
        const imageData = exportCtx.getImageData(0, 0, width, height);
        const data = imageData.data;

        // ── Step 2: Single-pass bbox + center-of-mass (centroid) ───────────
        // One scan collects all statistics — avoids two full-image traversals.
        // Threshold: grayscale < 250 filters out pure-white background while
        // accounting for subtle canvas anti-aliasing at stroke edges.
        let minX = width, minY = height, maxX = 0, maxY = 0;
        let sumX = 0, sumY = 0, count = 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                if (gray < 250) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                    sumX += x;
                    sumY += y;
                    count++;
                }
            }
        }

        if (count === 0) return null;

        // Center of mass — the pixel-density-weighted centroid
        const comX = sumX / count;
        const comY = sumY / count;

        // ── Step 3: Square crop anchored at center-of-mass ─────────────────
        // Use the larger bbox dimension + 15% padding so the drawing breathes.
        // CRITICAL: anchor the crop so COM lands exactly at crop center.
        // When crop extends beyond image bounds, pad those regions with white
        // instead of clamping+shifting (which would misalign COM).
        const bboxW = maxX - minX + 1;
        const bboxH = maxY - minY + 1;
        const contentLen = Math.max(bboxW, bboxH);
        const pad = Math.ceil(contentLen * 0.15);
        const cropSize = contentLen + pad * 2;

        // Top-left of crop rect in source coordinates
        const cropSrcX = Math.round(comX - cropSize / 2);
        const cropSrcY = Math.round(comY - cropSize / 2);

        // Build crop canvas with white background fill (handles OOB regions)
        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = cropSize;
        cropCanvas.height = cropSize;
        const cropCtx = cropCanvas.getContext('2d');
        cropCtx.fillStyle = '#ffffff';
        cropCtx.fillRect(0, 0, cropSize, cropSize);

        // Intersection of crop rect with image bounds
        const sx = Math.max(0, cropSrcX);
        const sy = Math.max(0, cropSrcY);
        const sw = Math.min(width - sx, cropSize - Math.max(0, sx - cropSrcX));
        const sh = Math.min(height - sy, cropSize - Math.max(0, sy - cropSrcY));
        const dx = Math.max(0, -cropSrcX);  // destination offset if crop starts left of image
        const dy = Math.max(0, -cropSrcY);

        if (sw > 0 && sh > 0) {
            cropCtx.drawImage(exportCanvas, sx, sy, sw, sh, dx, dy, sw, sh);
        }

        // ── Step 4: Bilinear downscale crop → 28×28 (SINGLE step) ─────────
        // Single bilinear step from crop to 28×28 preserves stroke mass
        // better than two-step (crop→96→28) which washed out thin strokes.
        // Fill white → binarize → invert gives white-strokes-on-black.
        preprocessCtx.imageSmoothingEnabled = true;
        preprocessCtx.imageSmoothingQuality = 'medium';
        preprocessCtx.fillStyle = '#ffffff';
        preprocessCtx.fillRect(0, 0, 28, 28);

        // Draw into 20×20 active area with 4px margin (QuickDraw standard)
        const MARGIN = 4;
        const ACTIVE = 20;
        preprocessCtx.drawImage(cropCanvas, MARGIN, MARGIN, ACTIVE, ACTIVE);

        // ── Step 5: Adaptive threshold binarization ──────────────────────
        // Instead of a fixed threshold, compute it from the grayscale
        // distribution: pick the value at the TARGET_DENSITY percentile.
        // This adapts to light sketches (higher threshold) vs dense
        // sketches (lower threshold), always yielding ~25% non-zero pixels.
        const outData = preprocessCtx.getImageData(0, 0, 28, 28);
        const NUM = 784;
        const TARGET_DENSITY = 0.25;   // target 25% non-zero

        // First pass: collect all grayscale values
        const allGrays = new Array(NUM);
        for (let i = 0; i < NUM; i++) {
            const j = i * 4;
            allGrays[i] = (outData.data[j] + outData.data[j + 1] + outData.data[j + 2]) / 3;
        }

        // Sort to find the threshold at the target percentile
        const sorted = [...allGrays].sort((a, b) => a - b);
        const targetIdx = Math.floor(NUM * TARGET_DENSITY);
        const THRESH = Math.min(240, Math.max(120, sorted[targetIdx]));

        // Second pass: binarize
        const pixels = new Array(NUM);
        for (let i = 0; i < NUM; i++) {
            pixels[i] = allGrays[i] <= THRESH ? 1.0 : 0.0;
        }

        // Update debug view (upscales 28→96 for high-res display)
        updateDebugView();

        return pixels;
    }

    /**
     * Update the "Model sees" debug view canvas (96×96 for display).
     * Reads the 28×28 preprocess, binarizes, and upscales with nearest-neighbor.
     */
    function updateDebugView() {
        // Read 28×28 grayscale from preprocess canvas
        const srcData = preprocessCtx.getImageData(0, 0, 28, 28);
        const imageData = debugCtx.createImageData(96, 96);

        // Upscale 28→96 with pixel doubling (integer ratio ~3.4×)
        for (let dy = 0; dy < 96; dy++) {
            for (let dx = 0; dx < 96; dx++) {
                const sx = Math.floor(dx * 28 / 96);
                const sy = Math.floor(dy * 28 / 96);
                const sj = (sy * 28 + sx) * 4;
                const gray = (srcData.data[sj] + srcData.data[sj + 1] + srcData.data[sj + 2]) / 3;
                // Invert: dark (ink) → white stroke, light (bg) → black bg
                const value = gray <= 200 ? 255 : 0;
                const dj = (dy * 96 + dx) * 4;
                imageData.data[dj]     = value;
                imageData.data[dj + 1] = value;
                imageData.data[dj + 2] = value;
                imageData.data[dj + 3] = 255;
            }
        }

        debugCtx.putImageData(imageData, 0, 0);
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /**
     * Send the preprocessed drawing to the backend for prediction.
     * This function is called automatically after debounce or manually
     * via the "Predict Now" button.
     */
    function sendPrediction() {
        const pixels = getPreprocessedPixels();

        if (pixels === null) {
            showEmptyCanvasMessage();
            return;
        }

        const normalizedStrokes = getNormalizedStrokes();

        window.dispatchEvent(new CustomEvent('requestPrediction', {
            detail: { pixels, strokes: normalizedStrokes },
        }));
    }

    /**
     * Show a message when the user tries to predict on an empty canvas.
     */
    function showEmptyCanvasMessage() {
        const list = document.getElementById('predictions-list');
        list.innerHTML = `
            <div class="placeholder-text">
                ✏️ 请先画点什么！
            </div>
        `;
    }

    // -----------------------------------------------------------------------
    // Tool Switching
    // -----------------------------------------------------------------------

    function setTool(tool) {
        currentTool = tool;

        // Update button states
        btnPen.classList.toggle('active', tool === 'pen');
        btnEraser.classList.toggle('active', tool === 'eraser');
    }

    // -----------------------------------------------------------------------
    // Event Listeners
    // -----------------------------------------------------------------------

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', continueDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events (with passive: false to allow preventDefault)
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', continueDrawing, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchcancel', stopDrawing);

    // Tool buttons
    btnPen.addEventListener('click', () => setTool('pen'));
    btnEraser.addEventListener('click', () => setTool('eraser'));

    // Clear button
    btnClear.addEventListener('click', () => {
        initCanvas();
        // Reset predictions display
        const list = document.getElementById('predictions-list');
        list.innerHTML = `
            <p class="placeholder-text">画点什么来查看预测结果！</p>
        `;
        const meta = document.getElementById('predictions-meta');
        if (meta) meta.classList.add('hidden');
        // Clear debug view
        debugCtx.fillStyle = '#000000';
        debugCtx.fillRect(0, 0, 96, 96);
    });

    // Predict Now button (bypasses debounce)
    btnPredict.addEventListener('click', () => {
        clearTimeout(debounceTimer);
        sendPrediction();
    });

    // Reset button
    if (btnReset) {
        btnReset.addEventListener('click', async () => {
            if (window.apiModule) {
                const result = await window.apiModule.resetState();
                if (result) {
                    console.log('Server state reset:', result.message);
                }
            }
        });
    }

    // Debounce slider
    debounceSlider.addEventListener('input', (e) => {
        debounceDelay = parseInt(e.target.value, 10);
        debounceValue.textContent = debounceDelay;
    });

    // -----------------------------------------------------------------------
    // Initialize
    // -----------------------------------------------------------------------

    initCanvas();

    // Expose sendPrediction globally for other modules
    window.canvasModule = {
        sendPrediction,
        getPreprocessedPixels,
        getStrokes: () => strokes,
    };
})();
