/**
 * API module for the CNN Sketch Guesser.
 *
 * Handles communication with the Flask backend prediction endpoint,
 * including request serialization, response parsing, and retry logic.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    // Base URL for the Flask backend API
    const API_BASE = 'http://localhost:5000';

    // Maximum number of retry attempts on failure
    const MAX_RETRIES = 3;

    // Delay between retries (exponential backoff: 500ms, 1000ms, 2000ms)
    const RETRY_DELAYS = [500, 1000, 2000];

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    let retryCount = 0;
    let isBackendAvailable = true;

    // -----------------------------------------------------------------------
    // DOM References
    // -----------------------------------------------------------------------

    const statusEl = document.getElementById('backend-status');
    const loadingEl = document.getElementById('predictions-loading');
    const predictionsList = document.getElementById('predictions-list');

    // -----------------------------------------------------------------------
    // Backend Status UI
    // -----------------------------------------------------------------------

    /**
     * Show the backend unavailable warning.
     */
    function showBackendError(message) {
        isBackendAvailable = false;
        statusEl.textContent = `⚠️ ${message}`;
        statusEl.classList.remove('hidden');
    }

    /**
     * Show backend connected status.
     */
    function showBackendConnected() {
        isBackendAvailable = true;
        statusEl.textContent = '✅ 模型已连接，正在等待绘制';
        statusEl.classList.remove('hidden');
        statusEl.style.backgroundColor = '#4caf50';
        statusEl.style.color = '#fff';
    }

    /**
     * Hide the backend status message.
     */
    function hideBackendError() {
        isBackendAvailable = true;
        statusEl.classList.add('hidden');
    }

    // -----------------------------------------------------------------------
    // Prediction Request
    // -----------------------------------------------------------------------

    /**
     * Show loading indicator.
     */
    function showLoading() {
        if (loadingEl) loadingEl.classList.remove('hidden');
        if (predictionsList) predictionsList.style.opacity = '0.5';
    }

    /**
     * Hide loading indicator.
     */
    function hideLoading() {
        if (loadingEl) loadingEl.classList.add('hidden');
        if (predictionsList) predictionsList.style.opacity = '1';
    }

    /**
     * Send a prediction request to the backend.
     *
     * @param {number[]} pixels - Array of 784 floats in [0, 1]
     * @param {Array<Array<{x:number, y:number}>>} [strokes] - Optional stroke sequences
     * @returns {Promise<Object>} - Prediction result with top5 array
     */
    async function fetchPrediction(pixels, strokes) {
        showLoading();
        try {
            const payload = { pixels };
            if (strokes && strokes.length > 0) {
                payload.strokes = strokes;
            }
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(
                    errorData.error || `Server returned ${response.status}`
                );
            }

            return await response.json();
        } finally {
            hideLoading();
        }
    }

    /**
     * Request a prediction with automatic retry on failure.
     *
     * @param {number[]} pixels - Array of 784 floats in [0, 1]
     * @param {Array<Array<{x:number, y:number}>>} [strokes] - Optional stroke sequences
     * @returns {Promise<Object|null>} - Prediction result or null on failure
     */
    async function requestPredictionWithRetry(pixels, strokes) {
        retryCount = 0;

        while (retryCount <= MAX_RETRIES) {
            try {
                const result = await fetchPrediction(pixels, strokes);
                showBackendConnected();
                return result;
            } catch (error) {
                console.warn(`Prediction attempt ${retryCount + 1} failed:`, error);

                if (retryCount < MAX_RETRIES) {
                    const delay = RETRY_DELAYS[retryCount] || 2000;
                    showBackendError(
                        `后端连接失败，${delay}ms后重试... (${retryCount + 1}/${MAX_RETRIES})`
                    );
                    await sleep(delay);
                    retryCount++;
                } else {
                    showBackendError(
                        '后端连接失败，请确保服务器在 localhost:5000 运行'
                    );
                    return null;
                }
            }
        }

        return null;
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /**
     * Sleep for a given number of milliseconds.
     */
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // -----------------------------------------------------------------------
    // Event Handling
    // -----------------------------------------------------------------------

    // Listen for prediction requests from the canvas module
    window.addEventListener('requestPrediction', async (e) => {
        const { pixels, strokes } = e.detail;
        const result = await requestPredictionWithRetry(pixels, strokes);

        if (result) {
            // Dispatch result to visualization module
            window.dispatchEvent(new CustomEvent('predictionResult', {
                detail: result,
            }));

            // Dispatch result to history module
            window.dispatchEvent(new CustomEvent('saveToHistory', {
                detail: {
                    pixels,
                    prediction: result,
                },
            }));
        }
    });

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Reset server-side prediction state.
     */
    async function resetState() {
        try {
            const response = await fetch(`${API_BASE}/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            if (!response.ok) throw new Error('Reset failed');
            return await response.json();
        } catch (error) {
            console.warn('Reset failed:', error);
            return null;
        }
    }

    window.apiModule = {
        requestPredictionWithRetry,
        resetState,
        isBackendAvailable: () => isBackendAvailable,
    };
})();
