/**
 * History module for the CNN Sketch Guesser.
 *
 * Manages drawing history in browser localStorage, including saving,
 * loading, displaying, and deleting past drawings with their predictions.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    // localStorage key for history data
    const STORAGE_KEY = 'sketch_guesser_history';

    // Maximum number of history entries to keep
    const MAX_HISTORY_ENTRIES = 50;

    // Thumbnail dimensions
    const THUMB_WIDTH = 64;
    const THUMB_HEIGHT = 64;

    // JPEG quality for thumbnail compression (0.0 - 1.0)
    const THUMB_QUALITY = 0.6;

    // -----------------------------------------------------------------------
    // DOM Elements
    // -----------------------------------------------------------------------

    const historyGallery = document.getElementById('history-gallery');
    const btnClearHistory = document.getElementById('btn-clear-history');

    // -----------------------------------------------------------------------
    // Data Management
    // -----------------------------------------------------------------------

    /**
     * Load history from localStorage.
     *
     * @returns {Array} Array of history entry objects
     */
    function loadHistory() {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (error) {
            console.error('Failed to load history:', error);
            return [];
        }
    }

    /**
     * Save history to localStorage.
     *
     * @param {Array} history - Array of history entry objects
     */
    function saveHistory(history) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
        } catch (error) {
            console.error('Failed to save history:', error);
            // localStorage might be full; show a warning
            if (error.name === 'QuotaExceededError') {
                alert('History storage is full. Please clear some entries.');
            }
        }
    }

    // -----------------------------------------------------------------------
    // Thumbnail Generation
    // -----------------------------------------------------------------------

    /**
     * Create a thumbnail from a 784-float pixel array.
     *
     * @param {number[]} pixels - Array of 784 floats in [0, 1]
     * @returns {string} Base64-encoded JPEG thumbnail data URL
     */
    function createThumbnail(pixels) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 28;
        canvas.height = 28;

        const imageData = ctx.createImageData(28, 28);
        for (let i = 0; i < pixels.length; i++) {
            const value = Math.round(pixels[i] * 255);
            const idx = i * 4;
            imageData.data[idx] = value;     // R
            imageData.data[idx + 1] = value; // G
            imageData.data[idx + 2] = value; // B
            imageData.data[idx + 3] = 255;   // A
        }
        ctx.putImageData(imageData, 0, 0);

        // Scale up to thumbnail size with nearest-neighbor for pixel art look
        const thumbCanvas = document.createElement('canvas');
        const thumbCtx = thumbCanvas.getContext('2d');
        thumbCanvas.width = THUMB_WIDTH;
        thumbCanvas.height = THUMB_HEIGHT;
        thumbCtx.imageSmoothingEnabled = false;
        thumbCtx.drawImage(canvas, 0, 0, THUMB_WIDTH, THUMB_HEIGHT);

        return thumbCanvas.toDataURL('image/jpeg', THUMB_QUALITY);
    }

    // -----------------------------------------------------------------------
    // History Operations
    // -----------------------------------------------------------------------

    /**
     * Add a new entry to the history.
     *
     * @param {Object} data
     * @param {number[]} data.pixels - Preprocessed 784-float array
     * @param {Object} data.prediction - Backend prediction result
     */
    function addHistoryEntry(data) {
        const { pixels, prediction } = data;

        const history = loadHistory();

        // Create thumbnail
        const thumbnail = createThumbnail(pixels);

        // Build entry
        const entry = {
            id: generateId(),
            timestamp: new Date().toISOString(),
            thumbnail,
            topLabel: prediction.top5[0]?.label || 'unknown',
            topProbability: prediction.top5[0]?.probability || 0,
            top5: prediction.top5,
        };

        // Add to beginning (most recent first)
        history.unshift(entry);

        // Enforce maximum size
        if (history.length > MAX_HISTORY_ENTRIES) {
            history.pop();
        }

        saveHistory(history);
        renderHistory();
    }

    /**
     * Delete a specific history entry by ID.
     *
     * @param {string} id - Entry ID to delete
     */
    function deleteHistoryEntry(id) {
        const history = loadHistory().filter(entry => entry.id !== id);
        saveHistory(history);
        renderHistory();
    }

    /**
     * Clear all history entries.
     */
    function clearAllHistory() {
        if (!confirm('Are you sure you want to clear all history?')) {
            return;
        }
        localStorage.removeItem(STORAGE_KEY);
        renderHistory();
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    /**
     * Render the history gallery grid.
     */
    function renderHistory() {
        const history = loadHistory();

        if (history.length === 0) {
            historyGallery.innerHTML = `
                <p class="placeholder-text">Your drawings will appear here.</p>
            `;
            return;
        }

        historyGallery.innerHTML = history.map(entry => `
            <div class="history-item" data-id="${entry.id}">
                <img src="${entry.thumbnail}" alt="${escapeHtml(entry.topLabel)}">
                <div class="history-label">${escapeHtml(entry.topLabel)}</div>
            </div>
        `).join('');

        // Attach click handlers
        historyGallery.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = item.getAttribute('data-id');
                showHistoryDetails(id);
            });
        });
    }

    /**
     * Show details for a specific history entry.
     *
     * @param {string} id - Entry ID to display
     */
    function showHistoryDetails(id) {
        const history = loadHistory();
        const entry = history.find(h => h.id === id);

        if (!entry) return;

        const dateStr = new Date(entry.timestamp).toLocaleString();
        const top5Html = entry.top5.map((item, index) => {
            const percentage = (item.probability * 100).toFixed(1);
            return `${index + 1}. ${item.label}: ${percentage}%`;
        }).join('\n');

        alert(
            `Drawing from ${dateStr}\n\n` +
            `Top Prediction: ${entry.topLabel} (${(entry.topProbability * 100).toFixed(1)}%)\n\n` +
            `Top 5:\n${top5Html}`
        );
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /**
     * Generate a unique ID for history entries.
     */
    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Escape HTML special characters.
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // Event Handling
    // -----------------------------------------------------------------------

    window.addEventListener('saveToHistory', (e) => {
        addHistoryEntry(e.detail);
    });

    btnClearHistory.addEventListener('click', clearAllHistory);

    // -----------------------------------------------------------------------
    // Initialize
    // -----------------------------------------------------------------------

    renderHistory();

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    window.historyModule = {
        loadHistory,
        addHistoryEntry,
        deleteHistoryEntry,
        clearAllHistory,
        renderHistory,
    };
})();
