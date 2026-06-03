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
        const { top5, latency_ms, model } = result;
        const top3 = (top5 || []).slice(0, 3);

        if (top3.length === 0) {
            predictionsList.innerHTML = `
                <p class="placeholder-text">未收到预测结果</p>
            `;
            return;
        }

        // Build HTML for predictions
        const html = top3.map((item, index) => {
            const percentage = (item.probability * 100).toFixed(1);
            const color = CATEGORY_COLORS[item.label] || '#e94560';
            const cnLabel = CHINESE_LABELS[item.label] || item.label;
            const rank = index + 1;
            const isTop = index === 0;
            const highlightClass = isTop ? 'prediction-item top-prediction' : 'prediction-item';

            return `
                <div class="${highlightClass}">
                    <span class="prediction-rank">#${rank}</span>
                    <span class="prediction-label">${escapeHtml(cnLabel)} (${escapeHtml(item.label)})</span>
                    <div class="prediction-bar-container">
                        <div class="prediction-bar" style="width: 0%; background: ${color};"></div>
                    </div>
                    <span class="prediction-probability">${percentage}%</span>
                </div>
            `;
        }).join('');

        predictionsList.innerHTML = html;

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
            predictionsMeta.textContent = `模型: ${modelName} · 推理延迟: ${latency}`;
            predictionsMeta.classList.remove('hidden');
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
    };
})();
