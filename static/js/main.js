/*!
 * Keystroke Analytics - Professional JavaScript
 * Version 2.0.0
 */

// Global Application State
window.KeystrokeAnalytics = {
    version: '2.0.0',
    apiBaseUrl: '/api',
    sessionId: null,
    currentUser: null,
    config: {
        demo: {
            minKeystrokes: 5,
            maxKeystrokes: 1000,
            qualityThresholds: {
                insufficient: 5,
                low: 15,
                good: 30,
                excellent: 50
            }
        },
        api: {
            timeout: 30000,
            retryAttempts: 3,
            retryDelay: 1000
        },
        ui: {
            animationDuration: 300,
            loadingDelay: 500,
            autoHideAlerts: 5000
        }
    }
};

// Utility Functions
const Utils = {
    /**
     * Generate a random session ID
     */
    generateSessionId() {
        return 'session-' + Math.random().toString(36).substr(2, 9) + '-' + Date.now();
    },

    /**
     * Format number with thousands separators
     */
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    },

    /**
     * Format duration in milliseconds to human readable
     */
    formatDuration(ms) {
        if (ms < 1000) return `${ms.toFixed(1)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
    },

    /**
     * Debounce function calls
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Throttle function calls
     */
    throttle(func, limit) {
        let lastFunc;
        let lastRan;
        return function executedFunction(...args) {
            if (!lastRan) {
                func(...args);
                lastRan = Date.now();
            } else {
                clearTimeout(lastFunc);
                lastFunc = setTimeout(() => {
                    if ((Date.now() - lastRan) >= limit) {
                        func(...args);
                        lastRan = Date.now();
                    }
                }, limit - (Date.now() - lastRan));
            }
        };
    },

    /**
     * Deep clone an object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    /**
     * Check if element is in viewport
     */
    isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    },

    /**
     * Smooth scroll to element
     */
    scrollToElement(element, offset = 0) {
        const targetPosition = element.offsetTop - offset;
        window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
        });
    },

    /**
     * Show loading state on element
     */
    showLoading(element, text = 'Loading...') {
        element.classList.add('loading');
        element.setAttribute('data-original-text', element.textContent);
        element.textContent = text;
        element.disabled = true;
    },

    /**
     * Hide loading state on element
     */
    hideLoading(element) {
        element.classList.remove('loading');
        const originalText = element.getAttribute('data-original-text');
        if (originalText) {
            element.textContent = originalText;
            element.removeAttribute('data-original-text');
        }
        element.disabled = false;
    }
};

// API Helper Functions
const API = {
    /**
     * Make HTTP request with retry logic
     */
    async request(url, options = {}) {
        const config = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': window.KeystrokeAnalytics.sessionId || 'anonymous'
            },
            timeout: window.KeystrokeAnalytics.config.api.timeout,
            ...options
        };

        let lastError;
        const maxRetries = window.KeystrokeAnalytics.config.api.retryAttempts;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);

                const response = await fetch(url, {
                    ...config,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                return await response.json();

            } catch (error) {
                lastError = error;
                
                if (attempt < maxRetries) {
                    console.warn(`API request failed (attempt ${attempt + 1}/${maxRetries + 1}):`, error);
                    await new Promise(resolve => 
                        setTimeout(resolve, window.KeystrokeAnalytics.config.api.retryDelay * (attempt + 1))
                    );
                } else {
                    console.error('API request failed after all retries:', error);
                    throw error;
                }
            }
        }
    },

    /**
     * GET request
     */
    async get(endpoint) {
        return this.request(`${window.KeystrokeAnalytics.apiBaseUrl}${endpoint}`);
    },

    /**
     * POST request
     */
    async post(endpoint, data) {
        return this.request(`${window.KeystrokeAnalytics.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * PUT request
     */
    async put(endpoint, data) {
        return this.request(`${window.KeystrokeAnalytics.apiBaseUrl}${endpoint}`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(`${window.KeystrokeAnalytics.apiBaseUrl}${endpoint}`, {
            method: 'DELETE'
        });
    }
};

// UI Components and Helpers
const UI = {
    /**
     * Show alert message
     */
    showAlert(message, type = 'info', autoHide = true) {
        const alertContainer = document.createElement('div');
        alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.zIndex = '9999';
        alertContainer.style.minWidth = '300px';
        
        alertContainer.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertContainer);

        // Auto-hide after delay
        if (autoHide) {
            setTimeout(() => {
                if (alertContainer.parentNode) {
                    alertContainer.remove();
                }
            }, window.KeystrokeAnalytics.config.ui.autoHideAlerts);
        }

        return alertContainer;
    },

    /**
     * Show loading overlay
     */
    showLoadingOverlay(text = 'Loading...') {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'position-fixed top-0 start-0 w-100 h-100';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        overlay.style.zIndex = '9999';
        overlay.style.display = 'flex';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';

        overlay.innerHTML = `
            <div class="text-center text-white">
                <div class="spinner-border mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div>${text}</div>
            </div>
        `;

        document.body.appendChild(overlay);
        return overlay;
    },

    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    },

    /**
     * Animate counter
     */
    animateCounter(element, targetValue, duration = 1000) {
        const startValue = parseInt(element.textContent.replace(/,/g, '')) || 0;
        const increment = (targetValue - startValue) / (duration / 16);
        let currentValue = startValue;

        const timer = setInterval(() => {
            currentValue += increment;
            if ((increment > 0 && currentValue >= targetValue) || 
                (increment < 0 && currentValue <= targetValue)) {
                currentValue = targetValue;
                clearInterval(timer);
            }
            element.textContent = Utils.formatNumber(Math.floor(currentValue));
        }, 16);
    },

    /**
     * Add fade-in animation to element
     */
    fadeIn(element, duration = 300) {
        element.style.opacity = '0';
        element.style.display = 'block';
        
        const startTime = performance.now();
        
        function animate(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            element.style.opacity = progress.toString();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }
        
        requestAnimationFrame(animate);
    },

    /**
     * Add slide-in animation to element
     */
    slideIn(element, direction = 'up', duration = 300) {
        const translations = {
            up: 'translateY(20px)',
            down: 'translateY(-20px)',
            left: 'translateX(20px)',
            right: 'translateX(-20px)'
        };

        element.style.transform = translations[direction];
        element.style.opacity = '0';
        element.style.transition = `all ${duration}ms ease`;
        element.style.display = 'block';

        requestAnimationFrame(() => {
            element.style.transform = 'translate(0)';
            element.style.opacity = '1';
        });
    },

    /**
     * Create progress bar
     */
    createProgressBar(container, value = 0, label = '') {
        const progressHtml = `
            <div class="mb-2">
                <div class="d-flex justify-content-between">
                    <small>${label}</small>
                    <small class="progress-value">${value.toFixed(1)}%</small>
                </div>
                <div class="progress">
                    <div class="progress-bar" 
                         role="progressbar" 
                         style="width: ${value}%" 
                         aria-valuenow="${value}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = progressHtml;
        return container.querySelector('.progress-bar');
    },

    /**
     * Update progress bar value
     */
    updateProgressBar(progressBar, value, animated = true) {
        const container = progressBar.closest('.mb-2');
        const valueElement = container.querySelector('.progress-value');
        
        if (animated) {
            const currentValue = parseFloat(progressBar.getAttribute('aria-valuenow')) || 0;
            const increment = (value - currentValue) / 20;
            let current = currentValue;
            
            const timer = setInterval(() => {
                current += increment;
                if ((increment > 0 && current >= value) || 
                    (increment < 0 && current <= value)) {
                    current = value;
                    clearInterval(timer);
                }
                
                progressBar.style.width = `${current}%`;
                progressBar.setAttribute('aria-valuenow', current);
                valueElement.textContent = `${current.toFixed(1)}%`;
            }, 50);
        } else {
            progressBar.style.width = `${value}%`;
            progressBar.setAttribute('aria-valuenow', value);
            valueElement.textContent = `${value.toFixed(1)}%`;
        }
    }
};

// Data Analytics and Processing
const Analytics = {
    /**
     * Calculate keystroke statistics
     */
    calculateKeystrokeStats(timingData) {
        if (!timingData || timingData.length === 0) {
            return {};
        }

        const sorted = [...timingData].sort((a, b) => a - b);
        const sum = timingData.reduce((acc, val) => acc + val, 0);
        
        return {
            count: timingData.length,
            mean: sum / timingData.length,
            median: sorted[Math.floor(sorted.length / 2)],
            min: sorted[0],
            max: sorted[sorted.length - 1],
            std: this.calculateStandardDeviation(timingData),
            q25: sorted[Math.floor(sorted.length * 0.25)],
            q75: sorted[Math.floor(sorted.length * 0.75)],
            variance: this.calculateVariance(timingData)
        };
    },

    /**
     * Calculate standard deviation
     */
    calculateStandardDeviation(values) {
        const mean = values.reduce((acc, val) => acc + val, 0) / values.length;
        const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    },

    /**
     * Calculate variance
     */
    calculateVariance(values) {
        const mean = values.reduce((acc, val) => acc + val, 0) / values.length;
        return values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
    },

    /**
     * Calculate data quality score
     */
    calculateQualityScore(timingData, textLength = 0) {
        if (!timingData || timingData.length === 0) return 0;

        let score = 1.0;
        const config = window.KeystrokeAnalytics.config.demo;

        // Penalize short sequences
        if (timingData.length < config.qualityThresholds.low) {
            score *= 0.6;
        } else if (timingData.length < config.qualityThresholds.good) {
            score *= 0.8;
        }

        // Check for reasonable timing values (10ms to 2000ms)
        const reasonableTimings = timingData.filter(t => t >= 10000 && t <= 2000000);
        const reasonableRatio = reasonableTimings.length / timingData.length;
        score *= Math.max(0.5, reasonableRatio);

        // Check for variance (too uniform is suspicious)
        const cv = this.calculateStandardDeviation(timingData) / (timingData.reduce((a, b) => a + b, 0) / timingData.length);
        if (cv < 0.1) score *= 0.6; // Too uniform
        if (cv > 2.0) score *= 0.8; // Too variable

        return Math.max(0, Math.min(1, score));
    },

    /**
     * Get quality level from score
     */
    getQualityLevel(score) {
        if (score >= 0.8) return 'excellent';
        if (score >= 0.6) return 'good';
        if (score >= 0.4) return 'fair';
        return 'poor';
    }
};

// Performance Monitoring
const Performance = {
    marks: new Map(),

    /**
     * Start performance measurement
     */
    mark(name) {
        this.marks.set(name, performance.now());
    },

    /**
     * End performance measurement and return duration
     */
    measure(name) {
        const startTime = this.marks.get(name);
        if (!startTime) {
            console.warn(`Performance mark '${name}' not found`);
            return 0;
        }
        
        const duration = performance.now() - startTime;
        this.marks.delete(name);
        return duration;
    },

    /**
     * Monitor page load performance
     */
    monitorPageLoad() {
        window.addEventListener('load', () => {
            const navigation = performance.getEntriesByType('navigation')[0];
            const paintEntries = performance.getEntriesByType('paint');

            const metrics = {
                domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                firstPaint: paintEntries.find(entry => entry.name === 'first-paint')?.startTime || 0,
                firstContentfulPaint: paintEntries.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0
            };

            console.log('Page Performance Metrics:', metrics);
        });
    }
};

// Error Handling
const ErrorHandler = {
    /**
     * Log error to console and optionally send to server
     */
    logError(error, context = {}) {
        const errorInfo = {
            message: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            userAgent: navigator.userAgent,
            context: context
        };

        console.error('Application Error:', errorInfo);

        // In production, you might want to send errors to a logging service
        if (window.location.hostname !== 'localhost') {
            this.sendErrorToServer(errorInfo);
        }
    },

    /**
     * Send error to server for logging
     */
    async sendErrorToServer(errorInfo) {
        try {
            await API.post('/errors', errorInfo);
        } catch (e) {
            console.error('Failed to send error to server:', e);
        }
    },

    /**
     * Global error handler
     */
    setupGlobalErrorHandlers() {
        window.addEventListener('error', (event) => {
            this.logError(event.error, { type: 'javascript-error' });
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.logError(new Error(event.reason), { type: 'unhandled-promise-rejection' });
        });
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Set up global session
    window.KeystrokeAnalytics.sessionId = Utils.generateSessionId();
    
    // Set up error handling
    ErrorHandler.setupGlobalErrorHandlers();
    
    // Monitor performance
    Performance.monitorPageLoad();
    
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                Utils.scrollToElement(target, 80);
            }
        });
    });

    // Add loading states to forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('[type="submit"]');
            if (submitBtn) {
                Utils.showLoading(submitBtn, 'Processing...');
            }
        });
    });

    // Initialize tooltips
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }

    console.log('Keystroke Analytics v' + window.KeystrokeAnalytics.version + ' initialized');
});

// Export for use in other modules
window.KeystrokeAnalytics.Utils = Utils;
window.KeystrokeAnalytics.API = API;
window.KeystrokeAnalytics.UI = UI;
window.KeystrokeAnalytics.Analytics = Analytics;
window.KeystrokeAnalytics.Performance = Performance;
window.KeystrokeAnalytics.ErrorHandler = ErrorHandler; 