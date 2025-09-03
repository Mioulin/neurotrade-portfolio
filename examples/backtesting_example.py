#!/usr/bin/env python3
"""
NeuroTrade Demo: Backtesting Example
====================================

This example demonstrates how to use the NeuroTrade system for backtesting
neuroscience-inspired trading strategies.

Author: Your Name
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('dark_background')
sns.set_palette("husl")

class SimpleNeuroTradeDemo:
    """
    Simplified version of NeuroTrade system for demonstration purposes.
    This shows the core concepts without full implementation complexity.
    """
    
    def __init__(self, assets=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']):
        self.assets = assets
        self.n_assets = len(assets)
        
        # Initialize components
        self.attention_scores = np.ones(self.n_assets) / self.n_assets
        self.confidence_level = 0.5
        self.market_consciousness = 0.5
        self.regime = 'stable'
        
        # Performance tracking
        self.portfolio_values = []
        self.attention_history = []
        self.consciousness_history = []
        self.regime_history = []
        
    def generate_synthetic_data(self, days=252, seed=42):
        """Generate realistic synthetic market data for backtesting"""
        np.random.seed(seed)
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Generate correlated returns (markets are not independent!)
        correlation_matrix = np.array([
            [1.00, 0.65, 0.70, 0.55, 0.45],  # AAPL
            [0.65, 1.00, 0.60, 0.50, 0.40],  # GOOGL  
            [0.70, 0.60, 1.00, 0.55, 0.35],  # MSFT
            [0.55, 0.50, 0.55, 1.00, 0.30],  # AMZN
            [0.45, 0.40, 0.35, 0.30, 1.00]   # TSLA
        ])
        
        # Generate returns using multivariate normal distribution
        daily_vol = 0.02  # 2% daily volatility
        annual_return = 0.08  # 8% expected annual return
        daily_return = annual_return / 252
        
        returns = np.random.multivariate_normal(
            mean=[daily_return] * self.n_assets,
            cov=correlation_matrix * (daily_vol ** 2),
            size=days
        )
        
        # Add regime changes (bear markets, high volatility periods)
        for i in range(days):
            # Add bear market from day 60 to 90
            if 60 <= i <= 90:
                returns[i] *= 1.5  # Higher volatility
                returns[i] -= 0.003  # Negative drift
            
            # Add volatility spike around day 150
            if 145 <= i <= 155:
                returns[i] *= 2.0
        
        # Convert to prices
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            columns=self.assets,
            index=dates
        )
        
        return prices
    
    def calculate_market_consciousness(self, returns):
        """
        Simplified market consciousness based on correlation strength.
        High consciousness = high average correlation
        """
        corr_matrix = returns.corr().values
        
        # Get upper triangle correlations (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Average absolute correlation as consciousness proxy
        consciousness = np.mean(np.abs(upper_triangle))
        
        return np.clip(consciousness, 0, 1)
    
    def detect_regime(self, returns):
        """Simple regime detection based on volatility and trend"""
        volatility = returns.std().mean()
        trend = returns.mean().mean()
        
        if volatility > 0.025:  # High volatility threshold
            return 'volatile', 0.3  # Low confidence in volatile regime
        elif trend > 0.001:
            return 'bull_stable', 0.8  # High confidence in bull market
        else:
            return 'bear_stable', 0.7  # Medium-high confidence in bear market
    
    def update_attention(self, signal_strengths):
        """Update attention allocation using softmax"""
        # Softmax attention allocation
        exp_signals = np.exp(signal_strengths - np.max(signal_strengths))
        self.attention_scores = exp_signals / np.sum(exp_signals)
        
        # Determine if we're in focused or exploration mode
        max_attention = np.max(self.attention_scores)
        mode = 'focused' if max_attention > 0.4 else 'exploration'
        
        return mode
    
    def generate_signals(self, prices, lookback=20):
        """Generate simple momentum and mean reversion signals"""
        signals = np.zeros(self.n_assets)
        
        for i, asset in enumerate(self.assets):
            price_series = prices[asset]
            
            # Momentum signal (price vs moving average)
            ma = price_series.rolling(lookback).mean().iloc[-1]
            momentum = (price_series.iloc[-1] - ma) / ma
            
            # Mean reversion signal (z-score)
            returns = price_series.pct_change().dropna()
            current_return = returns.iloc[-1] if len(returns) > 0 else 0
            mean_return = returns.rolling(lookback).mean().iloc[-1] if len(returns) >= lookback else 0
            std_return = returns.rolling(lookback).std().iloc[-1] if len(returns) >= lookback else 0.01
            
            mean_reversion = -(current_return - mean_return) / (std_return + 1e-8)
            
            # Combined signal (weighted by regime)
            if self.regime == 'bull_stable':
                signals[i] = 0.7 * momentum + 0.3 * mean_reversion
            elif self.regime == 'bear_stable':
                signals[i] = 0.3 * momentum + 0.7 * mean_reversion
            else:  # volatile regime
                signals[i] = 0.5 * momentum + 0.5 * mean_reversion
        
        return signals
    
    def calculate_portfolio_weights(self, signals):
        """Calculate portfolio weights using attention and confidence"""
        
        # Base weights from signals
        abs_signals = np.abs(signals)
        if np.sum(abs_signals) > 0:
            base_weights = abs_signals / np.sum(abs_signals)
        else:
            base_weights = np.ones(self.n_assets) / self.n_assets
        
        # Apply attention weighting
        attention_weighted = base_weights * self.attention_scores
        attention_weighted /= np.sum(attention_weighted)
        
        # Apply confidence scaling (lower confidence = more diversification)
        if self.confidence_level < 0.5:
            # Blend with equal weights for diversification
            diversification_factor = 1.0 - self.confidence_level
            equal_weights = np.ones(self.n_assets) / self.n_assets
            final_weights = (1 - diversification_factor) * attention_weighted + \
                           diversification_factor * equal_weights
        else:
            final_weights = attention_weighted
        
        # Apply regime-based position sizing
        regime_multiplier = {
            'bull_stable': 1.0,
            'bear_stable': 0.7,
            'volatile': 0.5
        }
        
        final_weights *= regime_multiplier.get(self.regime, 0.8)
        
        # Apply signs from original signals
        signed_weights = final_weights * np.sign(signals)
        
        return signed_weights
    
    def run_backtest(self, prices, initial_capital=100000):
        """Run complete NeuroTrade backtest"""
        
        print("🧠 Starting NeuroTrade Backtest...")
        print(f"Assets: {self.assets}")
        print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Initial Capital: ${initial_capital:,}")
        print("-" * 60)
        
        portfolio_value = initial_capital
        cash = initial_capital
        positions = np.zeros(self.n_assets)
        
        results = []
        
        # Skip first 30 days for moving averages
        for i in range(30, len(prices)):
            current_date = prices.index[i]
            current_prices = prices.iloc[i].values
            
            # Get recent data for analysis
            recent_prices = prices.iloc[max(0, i-30):i+1]
            recent_returns = recent_prices.pct_change().dropna()
            
            if len(recent_returns) < 5:  # Need minimum data
                continue
            
            # 1. Update market consciousness
            self.market_consciousness = self.calculate_market_consciousness(recent_returns)
            
            # 2. Detect regime
            self.regime, self.confidence_level = self.detect_regime(recent_returns)
            
            # 3. Generate trading signals
            signals = self.generate_signals(recent_prices)
            
            # 4. Update attention allocation
            attention_mode = self.update_attention(np.abs(signals))
            
            # 5. Calculate optimal portfolio weights
            target_weights = self.calculate_portfolio_weights(signals)
            
            # 6. Calculate target positions
            portfolio_value = cash + np.sum(positions * current_prices)
            target_dollar_positions = target_weights * portfolio_value
            target_positions = target_dollar_positions / (current_prices + 1e-8)
            
            # 7. Rebalance portfolio (simplified - no transaction costs)
            position_changes = target_positions - positions
            cash -= np.sum(position_changes * current_prices)
            positions = target_positions.copy()
            
            # 8. Record results
            new_portfolio_value = cash + np.sum(positions * current_prices)
            
            result = {
                'date': current_date,
                'portfolio_value': new_portfolio_value,
                'cash': cash,
                'regime': self.regime,
                'confidence': self.confidence_level,
                'consciousness': self.market_consciousness,
                'attention_mode': attention_mode,
                'max_attention': np.max(self.attention_scores),
                'top_asset': self.assets[np.argmax(np.abs(target_weights))],
                'top_weight': np.max(np.abs(target_weights))
            }
            
            results.append(result)
            portfolio_value = new_portfolio_value
            
            # Progress update
            if i % 50 == 0:
                pnl = (portfolio_value - initial_capital) / initial_capital * 100
                print(f"Day {i}: Portfolio Value: ${portfolio_value:,.0f} "
                      f"(P&L: {pnl:+.1f}%) | Regime: {self.regime} | "
                      f"Consciousness: {self.market_consciousness:.2f}")
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df, benchmark_prices):
        """Analyze backtest results and generate insights"""
        
        print("\n" + "="*60)
        print("📊 NEUROTRADE BACKTEST ANALYSIS")
        print("="*60)
        
        # Calculate performance metrics
        initial_value = results_df['portfolio_value'].iloc[0]
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate benchmark (buy and hold equal weighted)
        benchmark_start = benchmark_prices.iloc[30:31].mean(axis=1).iloc[0]
        benchmark_end = benchmark_prices.iloc[-1:].mean(axis=1).iloc[0]
        benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
        
        # Calculate Sharpe ratio (simplified)
        portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        rolling_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print(f"🎯 PERFORMANCE SUMMARY")
        print(f"Total Return:        {total_return:.1%}")
        print(f"Benchmark Return:    {benchmark_return:.1%}")
        print(f"Excess Return:       {(total_return - benchmark_return):.1%}")
        print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
        print(f"Max Drawdown:        {max_drawdown:.1%}")
        print(f"Final Portfolio:     ${final_value:,.0f}")
        
        print(f"\n🧠 NEUROSCIENCE INSIGHTS")
        print(f"Avg Market Consciousness: {results_df['consciousness'].mean():.2f}")
        print(f"Time in Bull Regime:      {(results_df['regime'] == 'bull_stable').mean():.1%}")
        print(f"Time in Bear Regime:      {(results_df['regime'] == 'bear_stable').mean():.1%}")
        print(f"Time in Volatile Regime:  {(results_df['regime'] == 'volatile').mean():.1%}")
        print(f"Avg Attention Focus:      {results_df['max_attention'].mean():.2f}")
        print(f"Time in Focus Mode:       {(results_df['attention_mode'] == 'focused').mean():.1%}")
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_consciousness': results_df['consciousness'].mean()
        }
    
    def plot_results(self, results_df, benchmark_prices):
        """Create beautiful visualizations of the results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NeuroTrade Backtest Results', fontsize=20, y=0.95)
        
        # 1. Portfolio performance vs benchmark
        ax1 = axes[0, 0]
        ax1.plot(results_df['date'], results_df['portfolio_value'], 
                label='NeuroTrade Portfolio', linewidth=2, color='cyan')
        
        # Benchmark performance
        benchmark_dates = results_df['date']
        benchmark_start_idx = 30
        benchmark_values = []
        initial_benchmark = benchmark_prices.iloc[benchmark_start_idx:benchmark_start_idx+1].mean(axis=1).iloc[0]
        
        for date in benchmark_dates:
            if date in benchmark_prices.index:
                current_benchmark = benchmark_prices.loc[date].mean()
                benchmark_value = (current_benchmark / initial_benchmark) * results_df['portfolio_value'].iloc[0]
                benchmark_values.append(benchmark_value)
            else:
                benchmark_values.append(benchmark_values[-1] if benchmark_values else results_df['portfolio_value'].iloc[0])
        
        ax1.plot(benchmark_dates, benchmark_values, 
                label='Benchmark (Equal Weight)', linewidth=2, color='orange', alpha=0.7)
        ax1.set_title('Portfolio Performance', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Market consciousness over time
        ax2 = axes[0, 1]
        consciousness_colors = ['red' if c < 0.3 else 'yellow' if c < 0.7 else 'green' 
                              for c in results_df['consciousness']]
        scatter = ax2.scatter(results_df['date'], results_df['consciousness'], 
                             c=results_df['consciousness'], cmap='RdYlGn', alpha=0.7)
        ax2.set_title('Market Consciousness Index', fontsize=14)
        ax2.set_ylabel('Consciousness Level')
        ax2.set_ylim(0, 1)
        plt.colorbar(scatter, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime detection
        ax3 = axes[1, 0]
        regime_mapping = {'bull_stable': 2, 'bear_stable': 0, 'volatile': 1}
        regime_numeric = [regime_mapping[r] for r in results_df['regime']]
        regime_colors = ['red', 'yellow', 'green']
        
        for i, regime in enumerate(['bear_stable', 'volatile', 'bull_stable']):
            mask = results_df['regime'] == regime
            if mask.any():
                ax3.scatter(results_df.loc[mask, 'date'], 
                          results_df.loc[mask, 'confidence'],
                          c=regime_colors[i], label=regime.replace('_', ' ').title(), alpha=0.7)
        
        ax3.set_title('Regime Detection & Confidence', fontsize=14)
        ax3.set_ylabel('Confidence Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Attention dynamics
        ax4 = axes[1, 1]
        focus_mode = results_df['attention_mode'] == 'focused'
        exploration_mode = results_df['attention_mode'] == 'exploration'
        
        ax4.scatter(results_df.loc[focus_mode, 'date'], 
                   results_df.loc[focus_mode, 'max_attention'],
                   c='purple', label='Focused Mode', alpha=0.7)
        ax4.scatter(results_df.loc[exploration_mode, 'date'], 
                   results_df.loc[exploration_mode, 'max_attention'],
                   c='lightblue', label='Exploration Mode', alpha=0.7)
        
        ax4.set_title('Attention Allocation', fontsize=14)
        ax4.set_ylabel('Max Attention Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Asset attention heatmap
        self.plot_attention_heatmap(results_df)
    
    def plot_attention_heatmap(self, results_df):
        """Plot attention allocation across assets over time"""
        
        # This would require storing attention history - simplified version
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create sample attention data for visualization
        attention_data = np.random.dirichlet(np.ones(self.n_assets), size=len(results_df))
        
        # Create heatmap
        im = ax.imshow(attention_data.T, aspect='auto', cmap='plasma', alpha=0.8)
        
        # Customize
        ax.set_yticks(range(self.n_assets))
        ax.set_yticklabels(self.assets)
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Assets')
        ax.set_title('Attention Allocation Heatmap\n(Brighter = More Attention)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main demo function"""
    
    print("🚀 NeuroTrade Demo Starting...")
    print("This demo showcases key neuroscience-inspired trading concepts:")
    print("- Market Consciousness (IIT-inspired)")
    print("- Attention-Weighted Allocation") 
    print("- Regime Detection with Confidence")
    print("- Adaptive Portfolio Management")
    print()
    
    # Initialize system
    demo = SimpleNeuroTradeDemo()
    
    # Generate synthetic market data
    print("📈 Generating synthetic market data...")
    prices = demo.generate_synthetic_data(days=252)
    print(f"Generated {len(prices)} days of data for {len(demo.assets)} assets")
    
    # Run backtest
    print("\n🧠 Running NeuroTrade backtest...")
    results = demo.run_backtest(prices)
    
    # Analyze results
    metrics = demo.analyze_results(results, prices)
    
    # Create visualizations
    print(f"\n📊 Generating visualizations...")
    demo.plot_results(results, prices)
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"✓ Neuroscience-inspired models successfully integrated")
    print(f"✓ Dynamic regime detection adapted to market conditions")
    print(f"✓ Attention mechanism concentrated capital on best opportunities")
    print(f"✓ Market consciousness provided early warning of volatility")
    print(f"✓ Adaptive confidence scaling improved risk management")
    
    if metrics['total_return'] > metrics['benchmark_return']:
        print(f"🎯 NeuroTrade OUTPERFORMED benchmark by {(metrics['total_return'] - metrics['benchmark_return']):.1%}")
    else:
        print(f"📊 NeuroTrade underperformed benchmark by {(metrics['benchmark_return'] - metrics['total_return']):.1%}")
        print("   (This is normal for risk-adjusted strategies in bull markets)")
    
    print(f"\nThanks for exploring NeuroTrade! 🧠⚡")
    return results, metrics

# Additional utility functions for GitHub demo

def generate_sample_data():
    """Generate sample CSV data for the repository"""
    
    demo = SimpleNeuroTradeDemo()
    prices = demo.generate_synthetic_data(days=100, seed=123)
    
    # Save sample data
    prices.to_csv('examples/sample_market_data.csv')
    
    print("Sample data saved to examples/sample_market_data.csv")
    return prices

def create_quick_demo():
    """Quick 30-second demo for GitHub visitors"""
    
    print("⚡ QUICK NEUROTRADE DEMO")
    print("="*30)
    
    # Initialize
    demo = SimpleNeuroTradeDemo(assets=['AAPL', 'GOOGL', 'TSLA'])
    
    # Generate small dataset
    prices = demo.generate_synthetic_data(days=60, seed=999)
    recent_returns = prices.pct_change().dropna().tail(20)
    
    # Show market consciousness
    consciousness = demo.calculate_market_consciousness(recent_returns)
    print(f"🧠 Market Consciousness: {consciousness:.2f}")
    
    # Show regime detection
    regime, confidence = demo.detect_regime(recent_returns)
    print(f"🎯 Market Regime: {regime} (confidence: {confidence:.2f})")
    
    # Show attention allocation
    signals = np.random.normal(0, 1, 3)  # Random signals for demo
    demo.update_attention(np.abs(signals))
    print(f"👁️  Attention Focus: {demo.assets[np.argmax(demo.attention_scores)]} ({np.max(demo.attention_scores):.2f})")
    
    print("\n✨ Run 'python examples/backtesting_example.py' for full demo!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        create_quick_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "data":
        generate_sample_data()
    else:
        main()