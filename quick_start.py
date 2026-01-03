#!/usr/bin/env python3
"""
Quick Start Script for Portfolio Optimization Demo

This script provides a simple way to run the portfolio optimization demonstration
without needing to install the full package.

DISCLAIMER: This is for research and educational purposes only. NOT investment advice.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Run the portfolio optimization demo."""
    print("="*80)
    print("Portfolio Optimization with AI - Quick Start Demo")
    print("="*80)
    print()
    print("⚠️  DISCLAIMER: This is for research and educational purposes only.")
    print("   NOT investment advice. All results are hypothetical.")
    print()
    
    try:
        # Import required modules
        from src.data import DataLoader
        from src.models import GeneticAlgorithmOptimizer, MeanVarianceOptimizer, RiskParityOptimizer
        from src.backtest import BacktestEngine
        from src.utils import set_seed, format_percentage
        
        print("✅ All modules imported successfully!")
        print()
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Load sample data
        print("📊 Loading sample data...")
        loader = DataLoader()
        data = loader.load_sample_data()
        
        # Prepare returns data
        print("📈 Preparing returns data...")
        close_prices = {}
        for col in data["market_data"].columns:
            if col.endswith('_Close') or col.endswith('_Adj Close'):
                asset = col.split('_')[0]
                close_prices[asset] = data["market_data"][col]
        
        returns = pd.DataFrame(close_prices).pct_change().dropna()
        
        print(f"   Data shape: {returns.shape}")
        print(f"   Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"   Assets: {list(returns.columns)}")
        print()
        
        # Run different optimization methods
        print("🔧 Running portfolio optimization...")
        
        # 1. Genetic Algorithm
        print("   1. Genetic Algorithm...")
        ga_optimizer = GeneticAlgorithmOptimizer({
            "genetic_algorithm": {
                "population_size": 100,
                "generations": 200,
                "mutation_rate": 0.1
            }
        })
        ga_weights = ga_optimizer.optimize(returns)
        
        # 2. Mean-Variance
        print("   2. Mean-Variance Optimization...")
        mv_optimizer = MeanVarianceOptimizer({
            "mean_variance": {
                "risk_aversion": 1.0,
                "constraints": {"max_weight": 0.4}
            }
        })
        mv_weights = mv_optimizer.optimize(returns)
        
        # 3. Risk Parity
        print("   3. Risk Parity...")
        rp_optimizer = RiskParityOptimizer({
            "risk_parity": {
                "target_risk": 0.15
            }
        })
        rp_weights = rp_optimizer.optimize(returns)
        
        print("✅ Optimization completed!")
        print()
        
        # Compare results
        print("📊 Optimization Results Comparison:")
        print("-" * 60)
        
        methods = {
            "Genetic Algorithm": ga_weights,
            "Mean-Variance": mv_weights,
            "Risk Parity": rp_weights
        }
        
        for method_name, weights in methods.items():
            portfolio_returns = returns.dot(weights)
            
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            print(f"\n{method_name}:")
            print(f"  Total Return: {format_percentage(total_return)}")
            print(f"  Annualized Return: {format_percentage(annualized_return)}")
            print(f"  Volatility: {format_percentage(volatility)}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"  Portfolio Weights:")
            for i, asset in enumerate(returns.columns):
                print(f"    {asset}: {format_percentage(weights[i])}")
        
        print()
        
        # Run backtest on best method (highest Sharpe ratio)
        best_method = max(methods.keys(), 
                         key=lambda x: returns.dot(methods[x]).mean() / returns.dot(methods[x]).std() * np.sqrt(252))
        best_weights = methods[best_method]
        
        print(f"🚀 Running backtest on {best_method}...")
        
        backtest_engine = BacktestEngine({
            "backtesting": {
                "initial_capital": 1000000,
                "transaction_costs": {"commission": 0.001}
            }
        })
        
        results = backtest_engine.run_backtest(returns, best_weights, rebalancing_frequency="monthly")
        
        print("✅ Backtest completed!")
        print()
        
        # Display backtest results
        metrics = results["performance_metrics"]
        print("📈 Backtest Results:")
        print("-" * 40)
        print(f"Initial Capital: $1,000,000")
        print(f"Final Value: ${1000000 * (1 + metrics['total_return']):,.0f}")
        print(f"Total Return: {format_percentage(metrics['total_return'])}")
        print(f"Annualized Return: {format_percentage(metrics['annualized_return'])}")
        print(f"Volatility: {format_percentage(metrics['volatility'])}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {format_percentage(metrics['max_drawdown'])}")
        
        print()
        print("="*80)
        print("Demo completed successfully!")
        print()
        print("Next steps:")
        print("1. Run 'python main.py' for the full demonstration")
        print("2. Run 'streamlit run demo/app.py' for the interactive web interface")
        print("3. Explore the 'assets' directory for generated plots and reports")
        print()
        print("Remember: This is for educational purposes only. NOT investment advice.")
        print("="*80)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print()
        print("Please install required dependencies:")
        print("pip install numpy pandas matplotlib seaborn plotly streamlit yfinance scikit-learn cvxpy")
        print()
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Please check your Python environment and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    # Import pandas here to avoid import errors in the main function
    import pandas as pd
    import numpy as np
    
    sys.exit(main())
