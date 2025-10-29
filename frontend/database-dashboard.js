/**
 * Database Dashboard Integration
 * Frontend components for displaying database-stored information
 */

class DatabaseDashboard {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.updateInterval = 30000; // Update every 30 seconds
        this.charts = {};
        this.initializeDashboard();
    }

    initializeDashboard() {
        this.setupEventListeners();
        this.loadDashboardData();
        this.startAutoUpdate();
    }

    setupEventListeners() {
        // Database controls
        const refreshBtn = document.getElementById('refresh-database');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDashboardData());
        }

        // Export data controls
        const exportTradesBtn = document.getElementById('export-trades');
        if (exportTradesBtn) {
            exportTradesBtn.addEventListener('click', () => this.exportTradeData());
        }
    }

    async loadDashboardData() {
        try {
            console.log('📊 Loading database dashboard data...');
            
            // Load comprehensive dashboard data
            const dashboardData = await this.apiClient.makeRequest('/api/database/analytics/dashboard');
            
            if (dashboardData.success) {
                this.updateDashboardDisplay(dashboardData.data);
            }
            
            // Load performance summary
            await this.loadPerformanceSummary();
            
        } catch (error) {
            console.error('❌ Failed to load database dashboard:', error);
            this.showNotification('Failed to load database data', 'error');
        }
    }

    updateDashboardDisplay(data) {
        console.log('📈 Updating dashboard with database data:', data);

        // Update recent trades
        this.displayRecentTrades(data.recent_trades);
        
        // Update trading statistics
        this.displayTradingStats(data.trading_stats);
        
        // Update training overview
        this.displayTrainingOverview(data.training_overview);
        
        // Update account balance
        this.displayAccountBalance(data.account_balance);
    }

    displayRecentTrades(tradesData) {
        const container = document.getElementById('recent-trades-list');
        if (!container || !tradesData.trades) return;

        const trades = tradesData.trades.slice(0, 10); // Show last 10 trades
        
        container.innerHTML = `
            <div class="database-section">
                <h3>📊 Recent Trades (Database)</h3>
                <div class="trades-summary">
                    <span class="trade-count">${tradesData.count} total trades</span>
                </div>
                <div class="trades-list">
                    ${trades.length > 0 ? trades.map(trade => `
                        <div class="trade-item ${trade.pnl > 0 ? 'profit' : 'loss'}">
                            <div class="trade-header">
                                <span class="symbol">${trade.symbol}</span>
                                <span class="side ${trade.side.toLowerCase()}">${trade.side}</span>
                                <span class="timestamp">${new Date(trade.timestamp).toLocaleString()}</span>
                            </div>
                            <div class="trade-details">
                                <span class="quantity">${trade.quantity} @ $${trade.price}</span>
                                <span class="pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">
                                    ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(4)}
                                </span>
                            </div>
                            ${trade.trading_mode ? `<div class="trade-mode">${trade.trading_mode}</div>` : ''}
                        </div>
                    `).join('') : '<div class="no-data">No trades recorded yet</div>'}
                </div>
            </div>
        `;
    }

    displayTradingStats(stats) {
        const container = document.getElementById('trading-statistics');
        if (!container || !stats) return;

        container.innerHTML = `
            <div class="database-section">
                <h3>📈 Trading Performance (Database)</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Trades</div>
                        <div class="stat-value">${stats.total_trades}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value ${stats.win_rate >= 50 ? 'positive' : 'negative'}">
                            ${stats.win_rate.toFixed(1)}%
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total P&L</div>
                        <div class="stat-value ${stats.total_pnl >= 0 ? 'positive' : 'negative'}">
                            ${stats.total_pnl >= 0 ? '+' : ''}$${stats.total_pnl.toFixed(4)}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Avg Profit</div>
                        <div class="stat-value ${stats.avg_profit >= 0 ? 'positive' : 'negative'}">
                            ${stats.avg_profit >= 0 ? '+' : ''}$${stats.avg_profit.toFixed(4)}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Best Trade</div>
                        <div class="stat-value positive">+$${stats.max_profit.toFixed(4)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Worst Trade</div>
                        <div class="stat-value negative">$${stats.max_loss.toFixed(4)}</div>
                    </div>
                </div>
            </div>
        `;
    }

    displayTrainingOverview(trainingData) {
        const container = document.getElementById('training-overview');
        if (!container || !trainingData) return;

        const recentSessions = trainingData.recent_sessions || [];
        const activeSessions = trainingData.active_sessions || [];

        container.innerHTML = `
            <div class="database-section">
                <h3>🤖 ML Training Overview (Database)</h3>
                
                <div class="training-summary">
                    <div class="summary-card">
                        <div class="summary-label">Active Sessions</div>
                        <div class="summary-value">${trainingData.active_count}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Recent Sessions</div>
                        <div class="summary-value">${recentSessions.length}</div>
                    </div>
                </div>

                ${activeSessions.length > 0 ? `
                    <div class="active-sessions">
                        <h4>🔄 Active Training Sessions</h4>
                        ${activeSessions.map(session => `
                            <div class="session-item active">
                                <div class="session-header">
                                    <span class="algorithm">${session.algorithm}</span>
                                    <span class="symbol">${session.symbol}</span>
                                    <span class="status running">RUNNING</span>
                                </div>
                                <div class="session-progress">
                                    <span>Progress: ${session.current_progress.toFixed(1)}%</span>
                                    <span>Episode: ${session.episode_count}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}

                <div class="recent-sessions">
                    <h4>📋 Recent Training Sessions</h4>
                    ${recentSessions.length > 0 ? recentSessions.slice(0, 5).map(session => `
                        <div class="session-item">
                            <div class="session-header">
                                <span class="algorithm">${session.algorithm}</span>
                                <span class="symbol">${session.symbol}</span>
                                <span class="status ${session.status.toLowerCase()}">${session.status}</span>
                            </div>
                            <div class="session-details">
                                <span>Timesteps: ${session.timesteps.toLocaleString()}</span>
                                ${session.final_reward ? `<span>Reward: ${session.final_reward.toFixed(2)}</span>` : ''}
                                <span>Started: ${new Date(session.started_at).toLocaleDateString()}</span>
                            </div>
                        </div>
                    `).join('') : '<div class="no-data">No training sessions found</div>'}
                </div>
            </div>
        `;
    }

    displayAccountBalance(balanceData) {
        const container = document.getElementById('account-balance-db');
        if (!container) return;

        if (!balanceData) {
            container.innerHTML = `
                <div class="database-section">
                    <h3>💳 Account Balance (Database)</h3>
                    <div class="no-data">No balance data recorded</div>
                </div>
            `;
            return;
        }

        const assets = balanceData.assets || [];

        container.innerHTML = `
            <div class="database-section">
                <h3>💳 Account Balance (Database)</h3>
                <div class="balance-overview">
                    <div class="balance-card">
                        <div class="balance-label">Total Balance</div>
                        <div class="balance-value">$${balanceData.total_balance.toFixed(2)}</div>
                    </div>
                    <div class="balance-card">
                        <div class="balance-label">Available</div>
                        <div class="balance-value">$${balanceData.available_balance.toFixed(2)}</div>
                    </div>
                    <div class="balance-card">
                        <div class="balance-label">Unrealized P&L</div>
                        <div class="balance-value ${balanceData.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                            ${balanceData.unrealized_pnl >= 0 ? '+' : ''}$${balanceData.unrealized_pnl.toFixed(2)}
                        </div>
                    </div>
                </div>
                
                ${assets.length > 0 ? `
                    <div class="assets-breakdown">
                        <h4>Asset Breakdown</h4>
                        ${assets.map(asset => `
                            <div class="asset-item">
                                <span class="asset-symbol">${asset.asset}</span>
                                <span class="asset-amount">${asset.free}</span>
                                ${asset.locked > 0 ? `<span class="asset-locked">(${asset.locked} locked)</span>` : ''}
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <div class="balance-timestamp">
                    Last updated: ${new Date(balanceData.timestamp).toLocaleString()}
                </div>
            </div>
        `;
    }

    async loadPerformanceSummary(days = 30) {
        try {
            const summary = await this.apiClient.makeRequest(`/api/database/analytics/performance?days_back=${days}`);
            
            if (summary.success) {
                this.displayPerformanceSummary(summary.data);
            }
        } catch (error) {
            console.error('❌ Failed to load performance summary:', error);
        }
    }

    displayPerformanceSummary(data) {
        const container = document.getElementById('performance-summary');
        if (!container) return;

        container.innerHTML = `
            <div class="database-section">
                <h3>📊 Performance Summary (${data.period_days} days)</h3>
                
                <div class="performance-grid">
                    <div class="performance-section">
                        <h4>Trading Performance</h4>
                        <div class="performance-stats">
                            <div class="perf-stat">
                                <span class="label">Total Trades:</span>
                                <span class="value">${data.trading.total_trades}</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Win Rate:</span>
                                <span class="value ${data.trading.win_rate >= 50 ? 'positive' : 'negative'}">
                                    ${data.trading.win_rate}%
                                </span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Total P&L:</span>
                                <span class="value ${data.trading.total_pnl >= 0 ? 'positive' : 'negative'}">
                                    $${data.trading.total_pnl}
                                </span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="performance-section">
                        <h4>Training Performance</h4>
                        <div class="performance-stats">
                            <div class="perf-stat">
                                <span class="label">Total Sessions:</span>
                                <span class="value">${data.training.total_sessions}</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Completed:</span>
                                <span class="value">${data.training.completed_sessions}</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Success Rate:</span>
                                <span class="value">${data.training.success_rate}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="summary-timestamp">
                    Generated: ${new Date(data.generated_at).toLocaleString()}
                </div>
            </div>
        `;
    }

    async exportTradeData() {
        try {
            console.log('📤 Exporting trade data...');
            
            const trades = await this.apiClient.makeRequest('/api/database/trades?limit=1000');
            
            if (trades.success && trades.data.trades.length > 0) {
                this.downloadCSV(trades.data.trades, 'trading_history.csv');
                this.showNotification('Trade data exported successfully', 'success');
            } else {
                this.showNotification('No trade data to export', 'warning');
            }
        } catch (error) {
            console.error('❌ Export failed:', error);
            this.showNotification('Failed to export trade data', 'error');
        }
    }

    downloadCSV(data, filename) {
        const csvContent = this.convertToCSV(data);
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    convertToCSV(data) {
        if (!data || data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvRows = [headers.join(',')];
        
        for (const row of data) {
            const values = headers.map(header => {
                const value = row[header];
                return typeof value === 'string' ? `"${value}"` : value;
            });
            csvRows.push(values.join(','));
        }
        
        return csvRows.join('\n');
    }

    startAutoUpdate() {
        setInterval(() => {
            this.loadDashboardData();
        }, this.updateInterval);
    }

    showNotification(message, type) {
        // Use the main dashboard's notification system
        if (window.dashboard && window.dashboard.showNotification) {
            window.dashboard.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
}

// Export for use in main dashboard
window.DatabaseDashboard = DatabaseDashboard;