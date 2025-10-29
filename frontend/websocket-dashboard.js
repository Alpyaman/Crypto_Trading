/**
 * WebSocket-enabled Trading Dashboard with real-time updates
 */
class WebSocketTradingDashboard extends TradingDashboard {
    constructor() {
        super();
        this.websockets = new Map();
        this.reconnectAttempts = new Map();
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnecting = new Set();
        
        // Initialize WebSocket connections
        this.initializeWebSockets();
    }

    initializeWebSockets() {
        // Connect to training WebSocket
        this.connectTrainingWebSocket();
        
        // Connect to market data WebSocket (if available)
        this.connectMarketWebSocket();
    }

    connectTrainingWebSocket() {
        const wsUrl = `ws://${window.location.host}/api/v1/ml/ws/training`;
        
        if (this.isConnecting.has('training')) {
            console.log('Training WebSocket connection already in progress');
            return;
        }

        this.isConnecting.add('training');
        console.log(`🔌 Connecting to training WebSocket: ${wsUrl}`);

        try {
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('✅ Training WebSocket connected');
                this.websockets.set('training', ws);
                this.reconnectAttempts.set('training', 0);
                this.isConnecting.delete('training');
                
                // Show connection status
                this.updateConnectionStatus('training', true);
                
                // Request initial progress
                ws.send(JSON.stringify({ type: 'get_progress' }));
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'heartbeat') {
                        // Handle heartbeat
                        console.log('💓 Training WebSocket heartbeat received');
                        return;
                    }
                    
                    // Update training progress in real-time
                    this.updateTrainingProgress(data);
                    
                } catch (error) {
                    console.error('Failed to parse training WebSocket message:', error);
                }
            };

            ws.onclose = (event) => {
                console.log(`❌ Training WebSocket closed: ${event.code} - ${event.reason}`);
                this.websockets.delete('training');
                this.isConnecting.delete('training');
                this.updateConnectionStatus('training', false);
                
                // Attempt to reconnect
                this.attemptReconnect('training');
            };

            ws.onerror = (error) => {
                console.error('❌ Training WebSocket error:', error);
                this.isConnecting.delete('training');
                this.updateConnectionStatus('training', false);
            };

        } catch (error) {
            console.error('❌ Failed to create training WebSocket:', error);
            this.isConnecting.delete('training');
            this.updateConnectionStatus('training', false);
        }
    }

    connectMarketWebSocket() {
        const wsUrl = `ws://${window.location.host}/api/v1/ml/ws/market`;
        
        if (this.isConnecting.has('market')) {
            console.log('Market WebSocket connection already in progress');
            return;
        }

        this.isConnecting.add('market');
        console.log(`🔌 Connecting to market WebSocket: ${wsUrl}`);

        try {
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('✅ Market WebSocket connected');
                this.websockets.set('market', ws);
                this.reconnectAttempts.set('market', 0);
                this.isConnecting.delete('market');
                this.updateConnectionStatus('market', true);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'market_update') {
                        // Update market data in real-time
                        this.updateMarketData(data);
                    }
                    
                } catch (error) {
                    console.error('Failed to parse market WebSocket message:', error);
                }
            };

            ws.onclose = (event) => {
                console.log(`❌ Market WebSocket closed: ${event.code} - ${event.reason}`);
                this.websockets.delete('market');
                this.isConnecting.delete('market');
                this.updateConnectionStatus('market', false);
                this.attemptReconnect('market');
            };

            ws.onerror = (error) => {
                console.error('❌ Market WebSocket error:', error);
                this.isConnecting.delete('market');
                this.updateConnectionStatus('market', false);
            };

        } catch (error) {
            console.error('❌ Failed to create market WebSocket:', error);
            this.isConnecting.delete('market');
            this.updateConnectionStatus('market', false);
        }
    }

    attemptReconnect(type) {
        const attempts = this.reconnectAttempts.get(type) || 0;
        
        if (attempts >= this.maxReconnectAttempts) {
            console.log(`❌ Max reconnection attempts reached for ${type} WebSocket`);
            this.showNotification(`${type} real-time updates unavailable`, 'warning', 0);
            return;
        }

        const delay = this.reconnectDelay * Math.pow(2, attempts);
        console.log(`🔄 Attempting to reconnect ${type} WebSocket in ${delay}ms (attempt ${attempts + 1})`);
        
        setTimeout(() => {
            this.reconnectAttempts.set(type, attempts + 1);
            
            if (type === 'training') {
                this.connectTrainingWebSocket();
            } else if (type === 'market') {
                this.connectMarketWebSocket();
            }
        }, delay);
    }

    updateConnectionStatus(type, connected) {
        const statusElement = document.getElementById(`${type}-ws-status`);
        if (statusElement) {
            statusElement.className = `status-indicator ${connected ? 'online' : 'offline'}`;
            statusElement.title = `${type} WebSocket: ${connected ? 'Connected' : 'Disconnected'}`;
        }

        // Update any connection indicators in the UI
        const indicator = document.querySelector(`.connection-status.${type}`);
        if (indicator) {
            indicator.classList.toggle('connected', connected);
            indicator.classList.toggle('disconnected', !connected);
        }
    }

    updateTrainingProgress(progressData) {
        try {
            // Update progress bars
            if (progressData.current_step && progressData.total_steps) {
                const percentage = Math.round((progressData.current_step / progressData.total_steps) * 100);
                
                const progressBar = document.getElementById('training-progress-bar');
                if (progressBar) {
                    progressBar.style.width = `${percentage}%`;
                    progressBar.setAttribute('aria-valuenow', percentage);
                }

                const progressText = document.getElementById('training-progress-text');
                if (progressText) {
                    progressText.textContent = `${progressData.current_step.toLocaleString()} / ${progressData.total_steps.toLocaleString()} steps (${percentage}%)`;
                }
            }

            // Update training status
            const statusElement = document.getElementById('training-status');
            if (statusElement) {
                const status = progressData.status || 'unknown';
                statusElement.textContent = status.replace('_', ' ').toUpperCase();
                statusElement.className = `status-badge ${status}`;
            }

            // Update loss information
            if (progressData.current_loss !== undefined) {
                const lossElement = document.getElementById('current-loss');
                if (lossElement) {
                    lossElement.textContent = progressData.current_loss.toFixed(6);
                }
            }

            // Update time information
            if (progressData.elapsed_time) {
                const timeElement = document.getElementById('training-time');
                if (timeElement) {
                    timeElement.textContent = this.formatDuration(progressData.elapsed_time);
                }
            }

            // Update ETA
            if (progressData.eta) {
                const etaElement = document.getElementById('training-eta');
                if (etaElement) {
                    etaElement.textContent = this.formatDuration(progressData.eta);
                }
            }

            // Show completion notification
            if (progressData.status === 'completed') {
                this.showNotification('🎉 Model training completed successfully!', 'success', 5000);
            } else if (progressData.status === 'error') {
                this.showNotification(`❌ Training error: ${progressData.error || 'Unknown error'}`, 'error', 0);
            }

            console.log('📊 Training progress updated:', progressData);

        } catch (error) {
            console.error('Failed to update training progress:', error);
        }
    }

    updateMarketData(marketData) {
        try {
            // Update price display
            const priceElement = document.getElementById('current-price');
            if (priceElement) {
                priceElement.textContent = `$${marketData.price.toLocaleString()}`;
            }

            // Update 24h change
            const changeElement = document.getElementById('price-change-24h');
            if (changeElement) {
                const change = marketData.change_24h;
                changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
            }

            // Update chart if available
            if (this.charts.priceChart && marketData.price) {
                this.addPricePoint(marketData.price, new Date());
            }

            console.log('📈 Market data updated:', marketData);

        } catch (error) {
            console.error('Failed to update market data:', error);
        }
    }

    addPricePoint(price, timestamp) {
        try {
            if (!this.charts.priceChart) return;

            const chart = this.charts.priceChart;
            const data = chart.data;

            // Add new data point
            data.labels.push(timestamp.toLocaleTimeString());
            data.datasets[0].data.push(price);

            // Keep only last 50 points
            if (data.labels.length > 50) {
                data.labels.shift();
                data.datasets[0].data.shift();
            }

            chart.update('none'); // Update without animation for real-time feel

        } catch (error) {
            console.error('Failed to add price point to chart:', error);
        }
    }

    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    // Override the start training method to use WebSocket updates
    async startTraining(symbol, timesteps) {
        try {
            const response = await this.makeApiCall('/api/train_model', {
                method: 'POST',
                body: { symbol, timesteps }
            });

            this.showNotification('🚀 Model training started - watch progress in real-time!', 'success');
            
            // The WebSocket will automatically start receiving updates
            return response;

        } catch (error) {
            console.error('Training start failed:', error);
            this.showNotification('❌ Failed to start training', 'error');
        }
    }

    // Enhanced cleanup
    cleanup() {
        // Close all WebSocket connections
        for (const [type, ws] of this.websockets) {
            if (ws.readyState === WebSocket.OPEN) {
                console.log(`🔌 Closing ${type} WebSocket`);
                ws.close();
            }
        }
        
        this.websockets.clear();
        this.reconnectAttempts.clear();
        this.isConnecting.clear();

        // Call parent cleanup if available
        if (super.cleanup) {
            super.cleanup();
        }
    }

    // Method to manually reconnect all WebSockets
    reconnectAllWebSockets() {
        console.log('🔄 Manually reconnecting all WebSockets');
        
        this.cleanup();
        
        setTimeout(() => {
            this.initializeWebSockets();
        }, 1000);
    }

    // Get WebSocket connection status
    getWebSocketStatus() {
        const status = {};
        for (const [type, ws] of this.websockets) {
            status[type] = {
                connected: ws.readyState === WebSocket.OPEN,
                readyState: ws.readyState,
                url: ws.url
            };
        }
        return status;
    }
}

// Initialize enhanced dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Replace the existing dashboard with WebSocket-enabled version
    if (window.dashboard) {
        window.dashboard.cleanup?.();
    }
    
    window.dashboard = new WebSocketTradingDashboard();
    
    // Add WebSocket status indicators to the UI
    addWebSocketStatusIndicators();
    
    console.log('🚀 WebSocket Trading Dashboard initialized');
});

function addWebSocketStatusIndicators() {
    // Add connection status indicators to the header
    const header = document.querySelector('.header') || document.querySelector('.nav-container');
    if (header) {
        const statusContainer = document.createElement('div');
        statusContainer.className = 'websocket-status-container';
        statusContainer.innerHTML = `
            <div class="status-item">
                <span class="status-indicator" id="training-ws-status"></span>
                <span>Training</span>
            </div>
            <div class="status-item">
                <span class="status-indicator" id="market-ws-status"></span>
                <span>Market</span>
            </div>
        `;
        
        header.appendChild(statusContainer);
    }
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard && window.dashboard.cleanup) {
        window.dashboard.cleanup();
    }
});

// Export for global use
window.WebSocketTradingDashboard = WebSocketTradingDashboard;