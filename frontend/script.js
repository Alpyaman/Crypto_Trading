// Crypto Trading AI Dashboard - Main JavaScript
class TradingDashboard {
    constructor() {
        this.apiUrl = '';  // Use relative URLs since served from same origin
        this.currentSymbol = 'BTCUSDT';
        this.currentTimeframe = '1h';  // Default timeframe
        this.updateInterval = 5000;
        this.charts = {};
        this.activeIntervals = [];
        this.init();
    }

    init() {
        this.initializeNavigation();
        this.initializeCharts();
        this.startDataUpdates();
        this.initializeEventListeners();
        this.loadInitialData();
    }

    // Navigation System
    initializeNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const pages = document.querySelectorAll('.page');

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const targetPage = item.getAttribute('data-page');
                const targetPageId = targetPage + '-page'; // Convert to actual page ID
                
                console.log(`🔄 Switching to page: ${targetPage} (ID: ${targetPageId})`);
                
                // Update navigation
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');

                // Update pages
                pages.forEach(page => page.classList.remove('active'));
                const targetElement = document.getElementById(targetPageId);
                if (targetElement) {
                    targetElement.classList.add('active');
                    console.log(`✅ Successfully switched to ${targetPage} page`);
                } else {
                    console.error(`❌ Page element not found: ${targetPageId}`);
                }
            });
        });
    }

    // Chart Initialization
    initializeCharts() {
        // Price Chart
        const priceCtx = document.getElementById('price-chart');
        console.log('📊 Initializing price chart...', priceCtx ? '✅ Found canvas' : '❌ Canvas not found');
        
        if (priceCtx) {
            this.charts.price = new Chart(priceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Bitcoin Price (USDT)',
                        data: [],
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                color: '#ffffff',
                                font: {
                                    size: 14
                                }
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#ffffff',
                            bodyColor: '#ffffff',
                            borderColor: '#00d4aa',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#64748b',
                                maxTicksLimit: 10
                            }
                        },
                        y: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#64748b',
                                callback: function(value) {
                                    if (value >= 1000) {
                                        return '$' + (value / 1000).toFixed(1) + 'K';
                                    }
                                    return '$' + value.toLocaleString();
                                }
                            },
                            beginAtZero: false
                        }
                    }
                }
            });
            console.log('✅ Price chart initialized successfully');
        }

        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart');
        if (performanceCtx) {
            this.charts.performance = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#64748b'
                            }
                        },
                        y: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#64748b'
                            }
                        }
                    }
                }
            });
        }
    }

    // Event Listeners
    initializeEventListeners() {
        // Symbol selector
        const symbolSelect = document.getElementById('symbol-select');
        if (symbolSelect) {
            symbolSelect.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value;
                this.updateSymbolData();
            });
        }

        // Trading buttons
        document.getElementById('start-trading')?.addEventListener('click', () => this.startTrading());
        document.getElementById('stop-trading')?.addEventListener('click', () => this.stopTrading());

        // Training buttons
        document.getElementById('start-training')?.addEventListener('click', () => this.startTraining());
        document.getElementById('stop-training-btn')?.addEventListener('click', () => this.stopTraining());
        document.getElementById('test-model')?.addEventListener('click', () => this.testModel());

        // Settings form
        document.getElementById('settingsForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSettings();
        });

        // Refresh button
        document.getElementById('refresh-data')?.addEventListener('click', () => {
            console.log('🔄 Manual refresh triggered');
            this.loadInitialData();
        });

        // Chart controls
        const chartControls = document.querySelectorAll('.chart-controls .btn');
        chartControls.forEach(btn => {
            btn.addEventListener('click', () => {
                console.log('📊 Chart timeframe changed to:', btn.getAttribute('data-timeframe'));
                chartControls.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.updateChartTimeframe(btn.getAttribute('data-timeframe'));
            });
        });
    }

    // API Calls
    async apiCall(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            if (data) {
                options.body = JSON.stringify(data);
            }

            console.log(`Making API call to: ${this.apiUrl}${endpoint}`); // Debug log
            const response = await fetch(`${this.apiUrl}${endpoint}`, options);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log(`API response for ${endpoint}:`, result); // Debug log
            return result;
        } catch (error) {
            console.error('API call failed:', error);
            console.error('Endpoint:', endpoint);
            console.error('Method:', method);
            this.showNotification('API Error: ' + error.message, 'error');
            throw error;
        }
    }

    // Data Loading
    async loadInitialData() {
        try {
            console.log('🚀 Loading initial dashboard data...');
            
            // Load each section individually with error handling
            try {
                console.log('📊 Loading system status...');
                await this.updateSystemStatus();
                console.log('✅ System status loaded');
            } catch (error) {
                console.error('❌ System status failed:', error);
                // Continue with other data
            }
            
            try {
                console.log('💹 Loading market data...');
                await this.updateMarketData();
                console.log('✅ Market data loaded');
            } catch (error) {
                console.error('❌ Market data failed:', error);
                // Continue with other data
            }
            
            try {
                console.log('💰 Loading account balance...');
                await this.updateAccountBalance();
                console.log('✅ Account balance loaded');
            } catch (error) {
                console.error('❌ Account balance failed:', error);
                // Continue with other data
            }
            
            try {
                console.log('🤖 Loading model status...');
                await this.updateModelStatus();
                console.log('✅ Model status loaded');
            } catch (error) {
                console.error('❌ Model status failed:', error);
                // Continue with other data
            }

            try {
                console.log('📈 Loading technical indicators...');
                await this.updateTechnicalIndicators();
                console.log('✅ Technical indicators loaded');
            } catch (error) {
                console.error('❌ Technical indicators failed:', error);
                // Continue with other data
            }

            try {
                console.log('🧠 Checking training status...');
                await this.checkTrainingStatus();
                console.log('✅ Training status checked');
            } catch (error) {
                console.error('❌ Training status check failed:', error);
                // Continue with other data
            }

            console.log('✅ Dashboard loading completed (some sections may have failed)');
            this.showNotification('Dashboard data loaded', 'success');
            
        } catch (error) {
            console.error('❌ Critical error in loadInitialData:', error);
            this.showNotification('Some dashboard data failed to load', 'error');
        }
    }

    async updateSystemStatus() {
        try {
            console.log('🔍 Fetching system status...');
            const status = await this.apiCall('/api/status');
            console.log('📊 System status received:', status);
            
            const statusTextElement = document.getElementById('status-text');
            
            if (statusTextElement) {
                statusTextElement.textContent = status.status || 'Online';
            }
            
            console.log('✅ System status updated successfully');
        } catch (error) {
            console.error('❌ Failed to update system status:', error);
            // Set fallback values
            const statusTextElement = document.getElementById('status-text');
            if (statusTextElement) statusTextElement.textContent = 'Error';
        }
    }

    async checkTrainingStatus() {
        try {
            const progress = await this.apiCall('/api/ml/training-progress');
            
            if (progress.isTraining) {
                console.log('🧠 Training in progress detected, starting monitor...');
                this.updateTrainingProgress(progress);
                this.startTrainingMonitor();
            } else {
                this.updateTrainingProgress(progress);
            }
        } catch (error) {
            console.error('❌ Failed to check training status:', error);
        }
    }

    async updateMarketData() {
        try {
            console.log('💹 Fetching market data...');
            const marketData = await this.apiCall(`/api/market-data/${this.currentSymbol}?timeframe=${this.currentTimeframe}`);
            
            console.log('📊 Market data received:', marketData);
            
            // Update price display
            const price = marketData.price || 0;
            const currentPriceElement = document.getElementById('current-price');
            if (currentPriceElement) {
                currentPriceElement.textContent = `$${price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            }
            
            const changeElement = document.getElementById('price-change');
            const change = marketData.change || 0;
            if (changeElement) {
                changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
            }

            // Update market info elements only if they exist
            const marketRegimeElement = document.getElementById('market-regime');
            const volatilityElement = document.getElementById('volatility');
            
            if (marketRegimeElement) marketRegimeElement.textContent = 'Trending'; // Demo value
            if (volatilityElement) volatilityElement.textContent = 'Medium'; // Demo value

            // Update charts with error handling
            try {
                this.updatePriceChart(marketData.priceHistory || []);
            } catch (chartError) {
                console.error('📊 Chart update failed:', chartError);
            }
            
            console.log('✅ Market data updated successfully');
        } catch (error) {
            console.error('❌ Failed to update market data:', error);
            // Set error placeholder values
            const currentPriceElement = document.getElementById('current-price');
            const changeElement = document.getElementById('price-change');
            
            if (currentPriceElement) currentPriceElement.textContent = 'Error loading price';
            if (changeElement) changeElement.textContent = 'N/A';
        }
    }

    async updateAccountBalance() {
        try {
            console.log('💰 Fetching account balance...');
            const balance = await this.apiCall('/api/account/balance');
            console.log('💳 Account balance received:', balance);
            
            // Update header balance if element exists
            const totalBalanceElement = document.getElementById('totalBalance');
            const availableBalanceElement = document.getElementById('availableBalance');
            const lockedBalanceElement = document.getElementById('lockedBalance');
            
            if (totalBalanceElement) totalBalanceElement.textContent = `$${balance.total?.toFixed(2) || '0.00'}`;
            if (availableBalanceElement) availableBalanceElement.textContent = `$${balance.available?.toFixed(2) || '0.00'}`;
            if (lockedBalanceElement) lockedBalanceElement.textContent = `$${balance.locked?.toFixed(2) || '0.00'}`;
            
            // Update trading page balance display
            const tradingAvailableElement = document.getElementById('available-balance');
            const usedMarginElement = document.getElementById('used-margin');
            const unrealizedPnlElement = document.getElementById('unrealized-pnl');
            
            if (tradingAvailableElement) tradingAvailableElement.textContent = `$${balance.available?.toFixed(2) || '0.00'}`;
            if (usedMarginElement) usedMarginElement.textContent = `$${balance.locked?.toFixed(2) || '0.00'}`;
            if (unrealizedPnlElement) unrealizedPnlElement.textContent = `$0.00`; // Demo value
            
            console.log('✅ Account balance updated successfully');
        } catch (error) {
            console.error('❌ Failed to update account balance:', error);
        }
    }

    async updateModelStatus() {
        try {
            console.log('🤖 Fetching model status...');
            const modelStatus = await this.apiCall('/api/ml/status');
            console.log('🧠 Model status received:', modelStatus);
            
            const modelStatusElement = document.getElementById('model-status');
            
            if (modelStatusElement) {
                modelStatusElement.textContent = modelStatus.status || 'Not Loaded';
            }
            
            // Update model details in the dashboard
            const lastTrainingElement = document.getElementById('last-training');
            const modelConfidenceElement = document.getElementById('model-confidence');
            
            if (lastTrainingElement) lastTrainingElement.textContent = modelStatus.lastTrained || 'Never';
            if (modelConfidenceElement) modelConfidenceElement.textContent = `${modelStatus.confidence?.toFixed(1) || '0.0'}%`;
            
            console.log('✅ Model status updated successfully');
        } catch (error) {
            console.error('❌ Failed to update model status:', error);
            const modelStatusElement = document.getElementById('model-status');
            if (modelStatusElement) modelStatusElement.textContent = 'Error';
        }
    }

    async updateTechnicalIndicators() {
        try {
            console.log('📈 Fetching technical indicators...');
            const indicators = await this.apiCall(`/api/indicators/${this.currentSymbol}`);
            console.log('📊 Technical indicators received:', indicators);
            
            // Update RSI
            const rsi = indicators.rsi || 0;
            const rsiValueElement = document.getElementById('rsi-value');
            const rsiProgressElement = document.getElementById('rsi-progress');
            if (rsiValueElement) rsiValueElement.textContent = rsi.toFixed(2);
            if (rsiProgressElement) rsiProgressElement.style.width = `${rsi}%`;
            
            // Update MACD
            const macdValueElement = document.getElementById('macd-value');
            if (macdValueElement) macdValueElement.textContent = indicators.macd?.toFixed(4) || '0.0000';
            
            // Update other indicators with fallback to demo values
            const bbPositionElement = document.getElementById('bb-position');
            const volumeRatioElement = document.getElementById('volume-ratio');
            if (bbPositionElement) bbPositionElement.textContent = '-'; // Demo placeholder
            if (volumeRatioElement) volumeRatioElement.textContent = '-'; // Demo placeholder
            
            console.log('✅ Technical indicators updated successfully');
        } catch (error) {
            console.error('❌ Failed to update technical indicators:', error);
        }
    }

    // Chart Updates
    updatePriceChart(priceHistory) {
        if (!this.charts.price || !priceHistory || !priceHistory.length) {
            console.log('⚠️ Cannot update price chart: missing chart or data');
            return;
        }

        console.log('📈 Updating price chart with', priceHistory.length, 'data points');
        console.log('🔍 Sample data:', priceHistory.slice(0, 3));

        // Format data for chart with timeframe-specific labels
        const labels = priceHistory.map(item => {
            const date = new Date(item.timestamp);
            
            // Different label formats based on timeframe
            if (this.currentTimeframe === '1h') {
                return date.toLocaleTimeString('en-US', { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    hour12: false 
                });
            } else if (this.currentTimeframe === '4h') {
                return date.toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric',
                    hour: '2-digit'
                });
            } else { // 1d
                return date.toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric'
                });
            }
        });
        
        const prices = priceHistory.map(item => {
            const price = parseFloat(item.price);
            console.log('💰 Processing price:', price);
            return price;
        });

        console.log('📊 Processed prices range:', Math.min(...prices), '-', Math.max(...prices));

        // Update chart data
        this.charts.price.data.labels = labels;
        this.charts.price.data.datasets[0].data = prices;
        this.charts.price.data.datasets[0].label = `${this.currentSymbol} Price (${this.currentTimeframe.toUpperCase()})`;
        
        // Force the chart to recalculate scales
        this.charts.price.options.scales.y.min = undefined;
        this.charts.price.options.scales.y.max = undefined;
        
        // Update the chart with animation
        this.charts.price.update('active');
        
        console.log('✅ Price chart updated successfully');
    }

    updatePerformanceChart(performanceData) {
        if (!this.charts.performance || !performanceData.length) return;

        const labels = performanceData.map(item => new Date(item.timestamp).toLocaleDateString());
        const values = performanceData.map(item => item.value);

        this.charts.performance.data.labels = labels;
        this.charts.performance.data.datasets[0].data = values;
        this.charts.performance.update('none');
    }

    updateChartTimeframe(timeframe) {
        console.log(`📊 Updating chart timeframe to: ${timeframe}`);
        this.currentTimeframe = timeframe;
        
        // Update active button styling
        document.querySelectorAll('.chart-controls button').forEach(btn => {
            btn.classList.remove('active');
            if (btn.getAttribute('data-timeframe') === timeframe) {
                btn.classList.add('active');
            }
        });
        
        // Fetch new data for the selected timeframe
        this.updateMarketData();
    }

    // Trading Functions
    async startTrading() {
        const symbol = document.getElementById('trading-symbol')?.value || 'BTCUSDT';
        const mode = document.getElementById('trading-mode')?.value || 'balanced';
        const positionSize = document.getElementById('position-size')?.value || 1000;

        try {
            console.log(`🚀 Starting trading: ${symbol} in ${mode} mode with $${positionSize}`);
            
            const tradeData = {
                symbol: symbol,
                mode: mode,
                position_size: parseFloat(positionSize)
            };

            const result = await this.apiCall('/api/trading/start', 'POST', tradeData);
            
            if (result.success) {
                this.showNotification(`Trading started for ${symbol}`, 'success');
                
                // Update button states
                const startBtn = document.getElementById('start-trading');
                const stopBtn = document.getElementById('stop-trading');
                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                
            } else {
                this.showNotification(`Failed to start trading: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('❌ Trading start failed:', error);
            this.showNotification('Failed to start trading', 'error');
        }
    }

    async stopTrading() {
        try {
            console.log('🛑 Stopping trading...');
            const result = await this.apiCall('/api/trading/stop', 'POST');
            
            if (result.success) {
                this.showNotification('Trading stopped successfully', 'success');
                
                // Update button states
                const startBtn = document.getElementById('start-trading');
                const stopBtn = document.getElementById('stop-trading');
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                
            } else {
                this.showNotification('Failed to stop trading', 'error');
            }
        } catch (error) {
            console.error('❌ Error stopping trading:', error);
            this.showNotification('Error stopping trading', 'error');
        }
    }

    // ML Training Functions
    async startTraining() {
        try {
            const symbol = document.getElementById('train-symbol')?.value || 'BTCUSDT';
            const steps = parseInt(document.getElementById('training-steps')?.value) || 200000;
            const algorithm = document.getElementById('algorithm')?.value || 'PPO';
            
            console.log(`🧠 Starting ML training: ${algorithm} for ${symbol} with ${steps} steps`);
            
            const trainingParams = {
                symbol: symbol,
                timesteps: steps,
                algorithm: algorithm
            };

            const result = await this.apiCall('/api/ml/train', 'POST', trainingParams);
            
            if (result.success) {
                this.showNotification('Training started successfully', 'success');
                this.startTrainingMonitor();
                
                // Update training status
                const statusElement = document.getElementById('training-status');
                if (statusElement) statusElement.textContent = 'Training...';
                
            } else {
                this.showNotification(`Failed to start training: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('❌ Training start failed:', error);
            this.showNotification('Training start failed', 'error');
        }
    }

    async testModel() {
        try {
            console.log('🧪 Testing ML model...');
            const result = await this.apiCall('/api/ml/test', 'POST');
            
            if (result.success) {
                this.showNotification(`Model test completed. Accuracy: ${result.accuracy}%`, 'success');
            } else {
                this.showNotification('Model test failed', 'error');
            }
        } catch (error) {
            console.error('❌ Model test failed:', error);
            this.showNotification('Model test failed', 'error');
        }
    }

    startTrainingMonitor() {
        const trainingInterval = setInterval(async () => {
            try {
                const progress = await this.apiCall('/api/ml/training-progress');
                this.updateTrainingProgress(progress);
                
                if (!progress.isTraining && progress.status === 'completed') {
                    clearInterval(trainingInterval);
                    this.showNotification('Training completed successfully', 'success');
                    this.hideTrainingDetails();
                } else if (!progress.isTraining && progress.status === 'failed') {
                    clearInterval(trainingInterval);
                    this.showNotification(`Training failed: ${progress.errorMessage || 'Unknown error'}`, 'error');
                    this.hideTrainingDetails();
                } else if (!progress.isTraining && progress.status === 'stopped') {
                    clearInterval(trainingInterval);
                    this.showNotification('Training stopped by user', 'warning');
                    this.hideTrainingDetails();
                }
            } catch (error) {
                console.error('❌ Failed to update training progress:', error);
                clearInterval(trainingInterval);
            }
        }, 2000); // Update every 2 seconds

        this.activeIntervals.push(trainingInterval);
    }

    updateTrainingProgress(progress) {
        try {
            // Update status
            const statusElement = document.getElementById('training-status');
            if (statusElement) statusElement.textContent = this.formatTrainingStatus(progress.status, progress.isTraining);

            // Update progress bar and percentage
            const progressElement = document.getElementById('training-progress');
            const progressFill = document.getElementById('progress-fill');
            if (progressElement) progressElement.textContent = `${progress.progress.toFixed(1)}%`;
            if (progressFill) progressFill.style.width = `${progress.progress}%`;

            if (progress.isTraining) {
                this.showTrainingDetails();
                
                // Update training details
                const algorithmElement = document.getElementById('training-algorithm');
                const symbolElement = document.getElementById('training-symbol');
                const timeElapsedElement = document.getElementById('time-elapsed');
                const timeRemainingElement = document.getElementById('time-remaining');
                const timestepsElement = document.getElementById('training-timesteps');
                const episodesElement = document.getElementById('training-episodes');
                
                if (algorithmElement) algorithmElement.textContent = progress.algorithm || 'PPO';
                if (symbolElement) symbolElement.textContent = progress.symbol || 'BTCUSDT';
                if (timeElapsedElement) timeElapsedElement.textContent = progress.timeElapsed || '00:00:00';
                if (timeRemainingElement) timeRemainingElement.textContent = progress.timeRemaining || '00:00:00';
                if (timestepsElement) timestepsElement.textContent = `${progress.currentTimestep.toLocaleString()} / ${progress.totalTimesteps.toLocaleString()}`;
                if (episodesElement) episodesElement.textContent = progress.episodes.toLocaleString();

                // Update training metrics
                const lossElement = document.getElementById('training-loss');
                const rewardElement = document.getElementById('training-reward');
                const learningRateElement = document.getElementById('learning-rate');
                const episodeLengthElement = document.getElementById('episode-length');
                
                if (lossElement) lossElement.textContent = progress.loss > 0 ? progress.loss.toFixed(6) : '-';
                if (rewardElement) rewardElement.textContent = progress.meanReward !== 0 ? progress.meanReward.toFixed(3) : '-';
                if (learningRateElement) learningRateElement.textContent = progress.learningRate > 0 ? progress.learningRate.toExponential(2) : '-';
                if (episodeLengthElement) episodeLengthElement.textContent = progress.episodeLength > 0 ? progress.episodeLength.toString() : '-';

                // Show stop button
                const stopButton = document.getElementById('stop-training-btn');
                if (stopButton) stopButton.style.display = 'inline-flex';
            } else {
                // Hide stop button
                const stopButton = document.getElementById('stop-training-btn');
                if (stopButton) stopButton.style.display = 'none';
            }
        } catch (error) {
            console.error('❌ Error updating training progress display:', error);
        }
    }

    formatTrainingStatus(status, isTraining) {
        if (isTraining) {
            return status === 'starting' ? 'Starting...' : 'Training';
        }
        
        switch (status) {
            case 'completed': return 'Completed';
            case 'failed': return 'Failed';
            case 'stopped': return 'Stopped';
            case 'idle': return 'Ready';
            default: return 'Ready';
        }
    }

    showTrainingDetails() {
        const detailsElement = document.getElementById('training-details');
        const metricsElement = document.getElementById('training-metrics');
        if (detailsElement) detailsElement.style.display = 'block';
        if (metricsElement) metricsElement.style.display = 'grid';
    }

    hideTrainingDetails() {
        const detailsElement = document.getElementById('training-details');
        const metricsElement = document.getElementById('training-metrics');
        if (detailsElement) detailsElement.style.display = 'none';
        if (metricsElement) metricsElement.style.display = 'none';
    }

    async stopTraining() {
        try {
            console.log('🛑 Stopping training...');
            const result = await this.apiCall('/api/ml/train/stop', 'POST');
            
            if (result.success) {
                this.showNotification('Training stop requested', 'success');
            } else {
                this.showNotification('Failed to stop training', 'error');
            }
        } catch (error) {
            console.error('❌ Failed to stop training:', error);
            this.showNotification('Failed to stop training', 'error');
        }
    }

    stopTrainingMonitor() {
        this.activeIntervals.forEach(interval => clearInterval(interval));
        this.activeIntervals = [];
    }

    // Settings
    saveSettings() {
        const settings = {
            apiKey: document.getElementById('apiKey').value,
            secretKey: document.getElementById('secretKey').value,
            autoTrading: document.getElementById('autoTrading').checked,
            riskLevel: document.getElementById('riskLevel').value,
            maxPositionSize: parseFloat(document.getElementById('maxPositionSize').value),
            stopLoss: parseFloat(document.getElementById('stopLoss').value),
            takeProfit: parseFloat(document.getElementById('takeProfit').value)
        };

        localStorage.setItem('tradingSettings', JSON.stringify(settings));
        this.showNotification('Settings saved successfully', 'success');
    }

    loadSettings() {
        const savedSettings = localStorage.getItem('tradingSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            Object.keys(settings).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    if (element.type === 'checkbox') {
                        element.checked = settings[key];
                    } else {
                        element.value = settings[key];
                    }
                }
            });
        }
    }

    // Data Updates
    startDataUpdates() {
        // Update market data every 5 seconds
        const marketInterval = setInterval(() => {
            this.updateMarketData();
            this.updateTechnicalIndicators();
        }, this.updateInterval);

        // Update system status every 10 seconds
        const systemInterval = setInterval(() => {
            this.updateSystemStatus();
        }, 10000);

        // Update account balance every 30 seconds
        const balanceInterval = setInterval(() => {
            this.updateAccountBalance();
        }, 30000);

        this.activeIntervals.push(marketInterval, systemInterval, balanceInterval);
    }

    // Symbol Updates
    async updateSymbolData() {
        await this.updateMarketData();
        await this.updateTechnicalIndicators();
    }

    // Utility Functions
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: ${type === 'success' ? '#4ade80' : type === 'error' ? '#ef4444' : '#00d4aa'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    formatNumber(num, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(num);
    }

    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    }

    // Cleanup
    cleanup() {
        this.activeIntervals.forEach(interval => clearInterval(interval));
        this.activeIntervals = [];
    }
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
`;
document.head.appendChild(style);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
    
    // Load saved settings
    window.dashboard.loadSettings();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.cleanup();
    }
});