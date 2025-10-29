// Enhanced dashboard integration example
// This shows how to integrate the new error handling and loading states

// Initialize enhanced components
const apiClient = new EnhancedAPIClient();
const loadingManager = new LoadingStateManager();
const notificationManager = new NotificationManager();

// Example of enhanced API usage in your existing dashboard
class EnhancedTradingDashboard extends TradingDashboard {
    constructor() {
        super();
        this.apiClient = apiClient;
        this.loadingManager = loadingManager;
        this.notifications = notificationManager;
    }

    // Override the existing API call method with enhanced error handling
    async makeApiCall(endpoint, options = {}) {
        try {
            loadingManager.setLoading(endpoint, true);
            const response = await apiClient.makeRequest(endpoint, options);
            
            if (response.success) {
                return response.data || response;
            } else {
                throw new APIError(response.error, 400, response.error_code, response.details);
            }
        } catch (error) {
            if (error instanceof APIError) {
                notificationManager.show(`API Error: ${error.message}`, 'error');
                
                // Handle specific error types
                switch (error.code) {
                    case 'INVALID_CREDENTIALS':
                        notificationManager.show('Please check your API credentials', 'warning');
                        break;
                    case 'INSUFFICIENT_BALANCE':
                        notificationManager.show('Insufficient balance for trading', 'warning');
                        break;
                    case 'MODEL_NOT_READY':
                        notificationManager.show('ML model needs training first', 'info');
                        break;
                }
            } else {
                notificationManager.show('Connection error - please try again', 'error');
            }
            throw error;
        } finally {
            loadingManager.setLoading(endpoint, false);
        }
    }

    // Enhanced market data loading with retry
    async loadMarketData() {
        const containerId = 'market-overview';
        try {
            loadingManager.showSectionLoader(containerId, 'Loading market data...');
            const data = await this.makeApiCall('/api/market_data');
            
            // Update UI with success
            this.updateMarketDataUI(data);
            notificationManager.show('Market data updated', 'success', 2000);
            
        } catch (error) {
            loadingManager.showError(containerId, error, 'dashboard.loadMarketData()');
        }
    }

    // Enhanced trading start with validation
    async startTrading(symbol, mode) {
        try {
            // Show confirmation for production mode
            if (!confirm(`Start ${mode} trading for ${symbol}?`)) {
                return;
            }

            const response = await this.makeApiCall('/api/start_trading', {
                method: 'POST',
                body: { symbol, mode }
            });

            notificationManager.show(`Trading started successfully in ${mode} mode`, 'success');
            return response;

        } catch (error) {
            // Error already handled in makeApiCall
            console.error('Trading start failed:', error);
        }
    }

    // Enhanced training with progress tracking
    async startTraining(symbol, timesteps) {
        try {
            await this.makeApiCall('/api/train_model', {
                method: 'POST',
                body: { symbol, timesteps }
            });

            notificationManager.show('Model training started', 'info');
            
            // Start progress monitoring
            this.monitorTrainingProgress();

        } catch (error) {
            console.error('Training start failed:', error);
        }
    }

    // Monitor training progress with enhanced feedback
    async monitorTrainingProgress() {
        const progressContainer = 'training-progress';
        
        const checkProgress = async () => {
            try {
                const progress = await this.makeApiCall('/api/training_progress', {}, false); // Don't show loading
                
                if (progress && progress.status === 'training') {
                    this.updateTrainingProgressUI(progress);
                    
                    // Continue monitoring
                    setTimeout(checkProgress, 5000);
                } else if (progress && progress.status === 'completed') {
                    notificationManager.show('Model training completed!', 'success');
                    this.updateTrainingProgressUI(progress);
                }
                
            } catch (error) {
                loadingManager.showError(progressContainer, error, 'dashboard.monitorTrainingProgress()');
            }
        };

        checkProgress();
    }

    // Enhanced account info with detailed error handling
    async loadAccountInfo() {
        const containerId = 'account-info';
        try {
            loadingManager.showSectionLoader(containerId, 'Loading account information...');
            const accountData = await this.makeApiCall('/api/account_info');
            
            this.updateAccountInfoUI(accountData);
            
        } catch (error) {
            if (error.code === 'INVALID_CREDENTIALS') {
                loadingManager.showError(
                    containerId, 
                    error, 
                    'window.location.reload()'
                );
            } else {
                loadingManager.showError(containerId, error, 'dashboard.loadAccountInfo()');
            }
        }
    }

    // Bulk data loading with individual error handling
    async loadAllData() {
        const loadTasks = [
            { name: 'Market Data', fn: () => this.loadMarketData() },
            { name: 'Account Info', fn: () => this.loadAccountInfo() },
            { name: 'ML Status', fn: () => this.loadMLStatus() },
            { name: 'Trading Logs', fn: () => this.loadTradingLogs() }
        ];

        for (const task of loadTasks) {
            try {
                await task.fn();
            } catch (error) {
                console.warn(`${task.name} loading failed:`, error);
                // Continue with other tasks
            }
        }
    }
}

// Export for global use
window.EnhancedTradingDashboard = EnhancedTradingDashboard;

// Initialize enhanced dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Replace the existing dashboard with enhanced version
    if (window.dashboard) {
        // Preserve existing functionality and enhance it
        const originalDashboard = window.dashboard;
        window.dashboard = new EnhancedTradingDashboard();
        
        // Copy any custom properties from original
        Object.keys(originalDashboard).forEach(key => {
            if (typeof originalDashboard[key] !== 'function') {
                window.dashboard[key] = originalDashboard[key];
            }
        });
    }
});

// CSS for enhanced components
const enhancedCSS = `
    .global-loader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: transparent;
        z-index: 10001;
    }

    .global-loader.loading::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: linear-gradient(90deg, #00d4ff 0%, #007bff 50%, #00d4ff 100%);
        width: 100%;
        animation: loading-bar 2s infinite;
    }

    @keyframes loading-bar {
        0% { transform: translateX(-100%); }
        50% { transform: translateX(0%); }
        100% { transform: translateX(100%); }
    }

    .enhanced-card {
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .enhanced-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
    }

    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-indicator.online {
        background: #4caf50;
        animation: pulse 2s infinite;
    }

    .status-indicator.offline {
        background: #f44336;
    }

    .status-indicator.warning {
        background: #ff9800;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
`;

// Inject enhanced CSS
const styleSheet = document.createElement('style');
styleSheet.textContent = enhancedCSS;
document.head.appendChild(styleSheet);