/**
 * Enhanced API utility class with retry logic, loading states, and better error handling
 */
class EnhancedAPIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
        this.requestTimeout = 30000; // 30 seconds
        this.isOffline = false;
        
        // Monitor network status
        this.initNetworkMonitoring();
    }

    initNetworkMonitoring() {
        window.addEventListener('online', () => {
            this.isOffline = false;
            this.showNetworkStatus('back online', 'success');
        });

        window.addEventListener('offline', () => {
            this.isOffline = true;
            this.showNetworkStatus('offline', 'error');
        });
    }

    showNetworkStatus(status, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = `Network is ${status}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#f44336' : '#4caf50'};
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
    }

    async makeRequest(endpoint, options = {}, showLoading = true) {
        if (this.isOffline) {
            throw new Error('No internet connection available');
        }

        const {
            method = 'GET',
            body = null,
            headers = {},
            timeout = this.requestTimeout,
            retries = this.retryAttempts
        } = options;

        const requestOptions = {
            method,
            headers: {
                'Content-Type': 'application/json',
                ...headers
            },
            ...(body && { body: typeof body === 'string' ? body : JSON.stringify(body) })
        };

        // Show loading indicator
        if (showLoading) {
            this.showLoading(endpoint);
        }

        let lastError;
        
        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeout);

                requestOptions.signal = controller.signal;
                
                console.log(`🔄 Enhanced API call attempt ${attempt + 1}/${retries + 1} to: ${this.baseUrl}${endpoint}`);

                const response = await fetch(`${this.baseUrl}${endpoint}`, requestOptions);
                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new APIError(
                        errorData.detail || errorData.error || errorData.message || `HTTP ${response.status}: ${response.statusText}`,
                        response.status,
                        errorData.error_code || 'HTTP_ERROR',
                        errorData.details
                    );
                }

                const data = await response.json();
                console.log(`✅ Enhanced API success for ${endpoint}:`, data);
                
                // Hide loading indicator
                if (showLoading) {
                    this.hideLoading(endpoint);
                }

                return data;

            } catch (error) {
                lastError = error;
                console.warn(`⚠️ Enhanced API attempt ${attempt + 1} failed:`, error.message);
                
                // Don't retry on certain errors (client errors)
                if (error instanceof APIError) {
                    if (error.status === 401 || error.status === 403 || error.status === 404 || error.status === 422) {
                        console.error(`❌ Non-retryable error (${error.status}): ${error.message}`);
                        break;
                    }
                }

                // Don't retry on the last attempt
                if (attempt === retries) {
                    console.error(`❌ All ${retries + 1} enhanced API attempts failed for ${endpoint}`);
                    break;
                }

                // Exponential backoff with jitter
                const delay = this.retryDelay * Math.pow(2, attempt) + Math.random() * 1000;
                console.log(`⏳ Enhanced API retrying in ${Math.round(delay)}ms...`);
                await this.sleep(delay);
            }
        }

        // Hide loading indicator on error
        if (showLoading) {
            this.hideLoading(endpoint);
        }

        throw lastError;
    }

    showLoading(endpoint) {
        const loadingKey = this.getLoadingKey(endpoint);
        let loader = document.getElementById(loadingKey);
        
        if (!loader) {
            loader = document.createElement('div');
            loader.id = loadingKey;
            loader.className = 'api-loader';
            loader.innerHTML = `
                <div class="loader-spinner"></div>
                <span>Loading ${this.getEndpointName(endpoint)}...</span>
            `;
            loader.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 20px;
                border-radius: 8px;
                z-index: 9999;
                display: flex;
                align-items: center;
                gap: 10px;
            `;
            
            document.body.appendChild(loader);
        }
    }

    hideLoading(endpoint) {
        const loadingKey = this.getLoadingKey(endpoint);
        const loader = document.getElementById(loadingKey);
        if (loader) {
            loader.remove();
        }
    }

    getLoadingKey(endpoint) {
        return `loader-${endpoint.replace(/[^a-zA-Z0-9]/g, '-')}`;
    }

    getEndpointName(endpoint) {
        const names = {
            '/api/market_data': 'market data',
            '/api/account_info': 'account info',
            '/api/ml_status': 'ML status',
            '/api/trading_logs': 'trading logs',
            '/api/training_progress': 'training progress'
        };
        return names[endpoint] || 'data';
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Custom API Error class for better error handling
 */
class APIError extends Error {
    constructor(message, status, code, details) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.code = code;
        this.details = details;
    }
}

/**
 * Enhanced Loading State Manager
 */
class LoadingStateManager {
    constructor() {
        this.loadingStates = new Map();
        this.createGlobalLoader();
    }

    createGlobalLoader() {
        const style = document.createElement('style');
        style.textContent = `
            .global-loader {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: transparent;
                z-index: 10001;
                transition: opacity 0.3s;
            }

            .global-loader.loading {
                opacity: 1;
            }

            .global-loader:not(.loading) {
                opacity: 0;
            }

            .global-loader::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                background: linear-gradient(90deg, #00d4ff 0%, #007bff 50%, #00d4ff 100%);
                width: 0%;
                animation: loading-progress 2s infinite;
            }

            @keyframes loading-progress {
                0% { width: 0%; left: 0%; }
                50% { width: 75%; left: 0%; }
                100% { width: 0%; left: 100%; }
            }

            .loader-spinner {
                width: 20px;
                height: 20px;
                border: 2px solid transparent;
                border-top: 2px solid currentColor;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .section-loader {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                color: #666;
                font-style: italic;
            }

            .error-state {
                padding: 20px;
                text-align: center;
                color: #f44336;
                background: rgba(244, 67, 54, 0.1);
                border: 1px solid rgba(244, 67, 54, 0.3);
                border-radius: 4px;
                margin: 10px 0;
            }

            .retry-button {
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
            }

            .retry-button:hover {
                background: #0056b3;
            }
        `;
        document.head.appendChild(style);

        // Create global loader element
        const loader = document.createElement('div');
        loader.id = 'global-loader';
        loader.className = 'global-loader';
        document.body.appendChild(loader);
    }

    setLoading(key, isLoading) {
        if (isLoading) {
            this.loadingStates.set(key, true);
        } else {
            this.loadingStates.delete(key);
        }

        // Update global loader
        const globalLoader = document.getElementById('global-loader');
        if (globalLoader) {
            if (this.loadingStates.size > 0) {
                globalLoader.classList.add('loading');
            } else {
                globalLoader.classList.remove('loading');
            }
        }
    }

    showSectionLoader(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="section-loader">
                    <div class="loader-spinner"></div>
                    <span style="margin-left: 10px;">${message}</span>
                </div>
            `;
        }
    }

    showError(containerId, error, retryCallback) {
        const container = document.getElementById(containerId);
        if (container) {
            const errorMessage = error instanceof APIError 
                ? `${error.message} (${error.code})`
                : error.message || 'An unexpected error occurred';

            container.innerHTML = `
                <div class="error-state">
                    <div style="margin-bottom: 10px;">⚠️ ${errorMessage}</div>
                    ${retryCallback ? '<button class="retry-button" onclick="' + retryCallback + '">Retry</button>' : ''}
                </div>
            `;
        }
    }
}

/**
 * Enhanced Notification System
 */
class NotificationManager {
    constructor() {
        this.notifications = [];
        this.maxNotifications = 5;
        this.defaultDuration = 5000;
        this.createNotificationContainer();
    }

    createNotificationContainer() {
        const style = document.createElement('style');
        style.textContent = `
            .notification-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                pointer-events: none;
            }

            .notification {
                background: #333;
                color: white;
                padding: 12px 20px;
                border-radius: 4px;
                margin-bottom: 10px;
                max-width: 400px;
                word-wrap: break-word;
                pointer-events: auto;
                animation: slideInRight 0.3s ease;
                position: relative;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .notification.success {
                background: #4caf50;
            }

            .notification.error {
                background: #f44336;
            }

            .notification.warning {
                background: #ff9800;
            }

            .notification.info {
                background: #2196f3;
            }

            .notification-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 0;
                margin-left: auto;
                opacity: 0.7;
            }

            .notification-close:hover {
                opacity: 1;
            }

            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            @keyframes slideOutRight {
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);

        const container = document.createElement('div');
        container.id = 'notification-container';
        container.className = 'notification-container';
        document.body.appendChild(container);
    }

    show(message, type = 'info', duration = this.defaultDuration) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icon = this.getIcon(type);
        notification.innerHTML = `
            ${icon}
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;

        const container = document.getElementById('notification-container');
        container.appendChild(notification);

        // Add to tracking array
        this.notifications.push(notification);

        // Remove oldest if too many
        if (this.notifications.length > this.maxNotifications) {
            this.remove(this.notifications[0]);
        }

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(notification), duration);
        }

        // Close button functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.onclick = () => this.remove(notification);

        return notification;
    }

    remove(notification) {
        if (notification && notification.parentNode) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
                const index = this.notifications.indexOf(notification);
                if (index > -1) {
                    this.notifications.splice(index, 1);
                }
            }, 300);
        }
    }

    getIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    clear() {
        this.notifications.forEach(notification => this.remove(notification));
    }
}

// Export for use in main dashboard
window.EnhancedAPIClient = EnhancedAPIClient;
window.APIError = APIError;
window.LoadingStateManager = LoadingStateManager;
window.NotificationManager = NotificationManager;