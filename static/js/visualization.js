/**
 * Visualization module for the CNN Sketch Guesser.
 *
 * Renders prediction results as a Top-5 list with animated probability bars
 * and a confidence trend chart showing the last 10 predictions.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // DOM Elements
    // -----------------------------------------------------------------------

    const predictionsList = document.getElementById('predictions-list');
    const trendChart = document.getElementById('trend-chart');

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    // Store the last 10 predictions for the trend chart
    const trendData = [];
    const MAX_TREND_POINTS = 10;

    // Track excluded labels for re-prediction
    let excludedLabels = [];

    // Store current pixels & strokes for re-prediction requests
    let currentPixels = null;
    let currentStrokes = null;

    // Color mapping for each category
    const CATEGORY_COLORS = {
        airplane: '#e94560',
        car: '#ff9800',
        cat: '#4caf50',
        dog: '#2196f3',
        house: '#9c27b0',
        tree: '#009688',
    };

    // Chinese labels
    const CHINESE_LABELS = {
        airplane: '飞机',
        car: '汽车',
        cat: '猫',
        dog: '狗',
        house: '房子',
        tree: '树',
    };

    // -----------------------------------------------------------------------
    // Predictions List
    // -----------------------------------------------------------------------

    const predictionsMeta = document.getElementById('predictions-meta');

    /**
     * Render the Top-3 predictions with animated probability bars.
     *
     * @param {Object} result - Prediction result from the backend
     * @param {Array} result.top5 - Array of {label, probability} objects
     */
    function renderPredictions(result) {
        const { top5, latency_ms, model, excluded } = result;
        const top3 = (top5 || []).slice(0, 3);
    
        // Update excluded labels from backend response (if this is a repredict)
        if (excluded && excluded.length > 0) {
            excludedLabels = excluded;
        } else if (!excluded) {
            // Fresh prediction — reset excluded labels
            excludedLabels = [];
        }
    
        if (top3.length === 0) {
            predictionsList.innerHTML = `
                <p class="placeholder-text">未收到预测结果</p>
            `;
            return;
        }
    
        // Determine if we can still exclude more labels
        const remainingLabels = Object.keys(CHINESE_LABELS).filter(
            label => !excludedLabels.includes(label)
        );
        const canExcludeMore = remainingLabels.length > 1 && top3.length > 0;
    
        // Build HTML for predictions
        const html = top3.map((item, index) => {
            const percentage = (item.probability * 100).toFixed(1);
            const color = CATEGORY_COLORS[item.label] || '#e94560';
            const cnLabel = CHINESE_LABELS[item.label] || item.label;
            const rank = index + 1;
            const isTop = index === 0;
            const highlightClass = isTop ? 'prediction-item top-prediction' : 'prediction-item';
    
            // Add "预测错了" button only to the #1 prediction, if there are still labels to exclude
            const wrongBtn = (isTop && canExcludeMore) ?
                `<button class="wrong-prediction-btn" data-label="${escapeHtml(item.label)}" title="排除此项，重新预测">
                    ✖ 预测错了
                </button>` : '';
    
            return `
                <div class="${highlightClass}">
                    <span class="prediction-rank">#${rank}</span>
                    <span class="prediction-label">${escapeHtml(cnLabel)} (${escapeHtml(item.label)})</span>
                    ${wrongBtn}
                    <div class="prediction-bar-container">
                        <div class="prediction-bar" style="width: 0%; background: ${color};"></div>
                    </div>
                    <span class="prediction-probability">${percentage}%</span>
                </div>
            `;
        }).join('');
    
        // Add excluded info line above predictions if any labels were excluded
        const excludedInfoLine = (excludedLabels.length > 0) ?
            `<div class="excluded-info-line">
                <span class="excluded-badge">🚫 已排除: ${excludedLabels.map(l => escapeHtml(CHINESE_LABELS[l] || l)).join(', ')}</span>
                <button class="reset-exclude-btn" title="恢复全部标签重新预测">↩ 恢复</button>
            </div>` : '';
    
        predictionsList.innerHTML = excludedInfoLine + html;
    
        // Attach click handler for "预测错了" button
        const wrongBtns = predictionsList.querySelectorAll('.wrong-prediction-btn');
        wrongBtns.forEach(btn => {
            btn.addEventListener('click', handleWrongPrediction);
        });
    
        // Attach click handler for "恢复" button
        const resetBtn = predictionsList.querySelector('.reset-exclude-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', handleResetExclude);
        }
    
        // Animate bars after a short delay to trigger CSS transition
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const bars = predictionsList.querySelectorAll('.prediction-bar');
                top3.forEach((item, index) => {
                    const percentage = item.probability * 100;
                    if (bars[index]) {
                        bars[index].style.width = `${percentage}%`;
                    }
                });
            });
        });
    
        // Update meta info (latency, model name)
        if (predictionsMeta) {
            const latency = latency_ms !== undefined ? `${latency_ms.toFixed(1)}ms` : '--';
            const modelName = model || 'unknown';
            const excludedStr = excludedLabels.length > 0 ? ` · 排除: ${excludedLabels.join(', ')}` : '';
            predictionsMeta.textContent = `模型: ${modelName} · 推理延迟: ${latency}${excludedStr}`;
            predictionsMeta.classList.remove('hidden');
        }
    }
    
    /**
     * Handle click on "预测错了" button — exclude the label and request re-prediction.
     */
    function handleWrongPrediction(e) {
        const label = e.currentTarget.getAttribute('data-label');
        if (!label || excludedLabels.includes(label)) return;
    
        // Add label to excluded list
        excludedLabels.push(label);
    
        // Dispatch re-prediction request
        if (currentPixels) {
            window.dispatchEvent(new CustomEvent('requestRepredict', {
                detail: {
                    pixels: currentPixels,
                    strokes: currentStrokes,
                    excludeLabels: [...excludedLabels],
                },
            }));
        }
    }
    
    /**
     * Handle click on "恢复" button — reset excluded labels and re-predict normally.
     */
    function handleResetExclude() {
        excludedLabels = [];
    
        if (currentPixels) {
            // Re-predict with no exclusions (same as original prediction)
            window.dispatchEvent(new CustomEvent('requestRepredict', {
                detail: {
                    pixels: currentPixels,
                    strokes: currentStrokes,
                    excludeLabels: [],
                },
            }));
        }
    }

    // -----------------------------------------------------------------------
    // Confidence Trend Chart
    // -----------------------------------------------------------------------

    /**
     * Update the confidence trend chart with the latest prediction.
     *
     * @param {Object} result - Prediction result from the backend
     */
    function updateTrendChart(result) {
        const topPrediction = result.top5 ? result.top5[0] : null;
        if (!topPrediction) return;

        // Add new data point
        trendData.push({
            label: topPrediction.label,
            probability: topPrediction.probability,
            timestamp: new Date(),
        });

        // Keep only the last N points
        if (trendData.length > MAX_TREND_POINTS) {
            trendData.shift();
        }

        renderTrendChart();
    }

    /**
     * Render the confidence trend as a simple SVG line chart.
     */
    function renderTrendChart() {
        if (trendData.length < 2) {
            trendChart.innerHTML = `
                <p class="placeholder-text">
                    Confidence trend will appear after 2+ predictions.
                </p>
            `;
            return;
        }

        const width = 400;
        const height = 150;
        const padding = 30;

        // Calculate scales
        const maxProb = 1.0;
        const xStep = (width - 2 * padding) / (MAX_TREND_POINTS - 1);

        // Build SVG
        let svgHtml = `
            <svg viewBox="0 0 ${width} ${height}" class="trend-svg"
                 style="width:100%; height:auto;">
                <!-- Background grid -->
                <line x1="${padding}" y1="${padding}" x2="${width - padding}" y2="${padding}"
                      stroke="#2a2a4a" stroke-dasharray="4" />
                <line x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}"
                      stroke="#2a2a4a" stroke-dasharray="4" />
                <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}"
                      stroke="#2a2a4a" stroke-dasharray="4" />

                <!-- Y-axis labels -->
                <text x="${padding - 5}" y="${padding + 4}" text-anchor="end" fill="#a0a0a0" font-size="10">100%</text>
                <text x="${padding - 5}" y="${height / 2 + 4}" text-anchor="end" fill="#a0a0a0" font-size="10">50%</text>
                <text x="${padding - 5}" y="${height - padding + 4}" text-anchor="end" fill="#a0a0a0" font-size="10">0%</text>
        `;

        // Build polyline points
        const points = trendData.map((point, index) => {
            const x = padding + (MAX_TREND_POINTS - trendData.length + index) * xStep;
            const y = (height - padding) - (point.probability * (height - 2 * padding));
            return `${x},${y}`;
        }).join(' ');

        // Draw line
        svgHtml += `
            <polyline points="${points}" fill="none" stroke="#e94560"
                      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        `;

        // Draw data points
        trendData.forEach((point, index) => {
            const x = padding + (MAX_TREND_POINTS - trendData.length + index) * xStep;
            const y = (height - padding) - (point.probability * (height - 2 * padding));
            const color = CATEGORY_COLORS[point.label] || '#e94560';

            svgHtml += `
                <circle cx="${x}" cy="${y}" r="4" fill="${color}" stroke="#1a1a2e" stroke-width="1" />
            `;
        });

        svgHtml += '</svg>';

        trendChart.innerHTML = svgHtml;
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /**
     * Escape HTML special characters to prevent XSS.
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // Event Handling
    // -----------------------------------------------------------------------

    // Listen for raw prediction request to capture pixels/strokes for re-prediction
    window.addEventListener('requestPrediction', (e) => {
        currentPixels = e.detail.pixels;
        currentStrokes = e.detail.strokes;
        // Reset excluded labels on fresh prediction
        excludedLabels = [];
    });

    window.addEventListener('predictionResult', (e) => {
        const result = e.detail;
        renderPredictions(result);
        updateTrendChart(result);
    });

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    window.visualizationModule = {
        renderPredictions,
        updateTrendChart,
        getExcludedLabels: () => [...excludedLabels],
        resetExcludedLabels: () => { excludedLabels = []; },
    };
})();
