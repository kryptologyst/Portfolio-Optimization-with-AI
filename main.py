#!/usr/bin/env python3
"""
Modern Portfolio Optimization with AI - Main Script

This script demonstrates the modernized portfolio optimization framework
with multiple optimization methods, risk management, and comprehensive backtesting.

DISCLAIMER: This is for research and educational purposes only. NOT investment advice.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Optional
import argparse
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import our modules
from src.data import DataLoader
from src.models import (
    GeneticAlgorithmOptimizer, 
    MeanVarianceOptimizer, 
    RiskParityOptimizer,
    MachineLearningOptimizer
)
from src.backtest import BacktestEngine, WalkForwardAnalysis
from src.utils import (
    set_seed, load_config, ensure_directory, 
    format_currency, format_percentage
)


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """Load sample data for demonstration.
    
    Returns:
        Dictionary containing market data, fundamentals, and labels.
    """
    print("Loading sample data...")
    loader = DataLoader()
    data = loader.load_sample_data()
    
    print(f"Market data shape: {data['market_data'].shape}")
    print(f"Fundamentals shape: {data['fundamentals'].shape}")
    print(f"Labels shape: {data['labels'].shape}")
    
    return data


def prepare_returns_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare returns data from market data.
    
    Args:
        market_data: Market data DataFrame.
        
    Returns:
        Returns DataFrame.
    """
    print("Preparing returns data...")
    
    # Extract close prices for each asset
    close_prices = {}
    for col in market_data.columns:
        if col.endswith('_Close') or col.endswith('_Adj Close'):
            asset = col.split('_')[0]
            close_prices[asset] = market_data[col]
    
    close_df = pd.DataFrame(close_prices)
    
    # Calculate returns
    returns = close_df.pct_change().dropna()
    
    print(f"Returns data shape: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns


def run_optimization_comparison(returns: pd.DataFrame, config: Dict) -> Dict[str, np.ndarray]:
    """Run multiple optimization methods and compare results.
    
    Args:
        returns: Returns DataFrame.
        config: Configuration dictionary.
        
    Returns:
        Dictionary mapping method names to optimal weights.
    """
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Genetic Algorithm
    print("\n1. Genetic Algorithm Optimization...")
    ga_optimizer = GeneticAlgorithmOptimizer(config)
    ga_weights = ga_optimizer.optimize(returns)
    results["genetic_algorithm"] = ga_weights
    
    # 2. Mean-Variance Optimization
    print("2. Mean-Variance Optimization...")
    mv_optimizer = MeanVarianceOptimizer(config)
    mv_weights = mv_optimizer.optimize(returns)
    results["mean_variance"] = mv_weights
    
    # 3. Risk Parity
    print("3. Risk Parity Optimization...")
    rp_optimizer = RiskParityOptimizer(config)
    rp_weights = rp_optimizer.optimize(returns)
    results["risk_parity"] = rp_weights
    
    # 4. Machine Learning (if features available)
    print("4. Machine Learning Optimization...")
    try:
        ml_optimizer = MachineLearningOptimizer(config)
        ml_weights = ml_optimizer.optimize(returns)
        results["machine_learning"] = ml_weights
    except Exception as e:
        print(f"   ML optimization failed: {e}")
        results["machine_learning"] = None
    
    return results


def compare_optimization_results(returns: pd.DataFrame, weights_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare results from different optimization methods.
    
    Args:
        returns: Returns DataFrame.
        weights_dict: Dictionary of optimization results.
        
    Returns:
        DataFrame with comparison results.
    """
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for method, weights in weights_dict.items():
        if weights is not None:
            # Calculate portfolio performance
            portfolio_returns = returns.dot(weights)
            
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            comparison_data.append({
                "Method": method.replace("_", " ").title(),
                "Total Return": total_return,
                "Annualized Return": annualized_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Weights": weights
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display results
    print("\nPerformance Comparison:")
    print("-" * 80)
    display_cols = ["Method", "Total Return", "Annualized Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]
    
    for _, row in comparison_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Total Return: {format_percentage(row['Total Return'])}")
        print(f"  Annualized Return: {format_percentage(row['Annualized Return'])}")
        print(f"  Volatility: {format_percentage(row['Volatility'])}")
        print(f"  Sharpe Ratio: {row['Sharpe Ratio']:.3f}")
        print(f"  Max Drawdown: {format_percentage(row['Max Drawdown'])}")
        
        # Display weights
        print(f"  Portfolio Weights:")
        for i, weight in enumerate(row['Weights']):
            asset = returns.columns[i]
            print(f"    {asset}: {format_percentage(weight)}")
    
    return comparison_df


def run_comprehensive_backtest(returns: pd.DataFrame, weights_dict: Dict[str, np.ndarray], config: Dict) -> None:
    """Run comprehensive backtesting for all optimization methods.
    
    Args:
        returns: Returns DataFrame.
        weights_dict: Dictionary of optimization results.
        config: Configuration dictionary.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE BACKTESTING")
    print("="*60)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Create benchmark (equal-weighted portfolio)
    benchmark_weights = np.ones(len(returns.columns)) / len(returns.columns)
    benchmark_returns = returns.dot(benchmark_weights)
    
    # Run backtests for each method
    backtest_results = {}
    
    for method, weights in weights_dict.items():
        if weights is not None:
            print(f"\nBacktesting {method.replace('_', ' ').title()}...")
            
            try:
                results = backtest_engine.run_backtest(
                    returns, 
                    weights, 
                    benchmark_returns=benchmark_returns,
                    rebalancing_frequency="monthly"
                )
                backtest_results[method] = results
                
                # Print summary
                metrics = results["performance_metrics"]
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {format_percentage(metrics['max_drawdown'])}")
                print(f"  Total Return: {format_percentage(metrics['total_return'])}")
                
            except Exception as e:
                print(f"  Backtest failed: {e}")
    
    # Generate plots for the best performing method
    if backtest_results:
        best_method = max(
            backtest_results.keys(),
            key=lambda x: backtest_results[x]["performance_metrics"]["sharpe_ratio"]
        )
        
        print(f"\nGenerating plots for best method: {best_method.replace('_', ' ').title()}")
        
        # Set up the backtest engine with best results
        backtest_engine.results = backtest_results[best_method]
        
        # Create output directory
        output_dir = ensure_directory("assets")
        
        # Generate plots
        backtest_engine.plot_results(save_path=output_dir / f"{best_method}_backtest.png")
        
        # Generate report
        report = backtest_engine.generate_report()
        print("\n" + report)
        
        # Save report
        with open(output_dir / f"{best_method}_report.txt", "w") as f:
            f.write(report)


def run_walk_forward_analysis(returns: pd.DataFrame, config: Dict) -> None:
    """Run walk-forward analysis for strategy validation.
    
    Args:
        returns: Returns DataFrame.
        config: Configuration dictionary.
    """
    print("\n" + "="*60)
    print("WALK-FORWARD ANALYSIS")
    print("="*60)
    
    # Initialize walk-forward analysis
    wf_analysis = WalkForwardAnalysis(config)
    
    # Use genetic algorithm for walk-forward analysis
    ga_optimizer = GeneticAlgorithmOptimizer(config)
    
    print("Running walk-forward analysis with Genetic Algorithm...")
    wf_results = wf_analysis.run_analysis(returns, ga_optimizer)
    
    # Display summary
    summary = wf_results["summary_metrics"]
    print(f"\nWalk-Forward Analysis Summary:")
    print(f"  Total Periods: {summary['total_periods']}")
    print(f"  Total Return: {format_percentage(summary['total_return'])}")
    print(f"  Annualized Return: {format_percentage(summary['annualized_return'])}")
    print(f"  Volatility: {format_percentage(summary['volatility'])}")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {format_percentage(summary['max_drawdown'])}")
    print(f"  Average Period Return: {format_percentage(summary['avg_period_return'])}")
    print(f"  Average Period Sharpe: {summary['avg_period_sharpe']:.3f}")


def create_visualization_dashboard(returns: pd.DataFrame, weights_dict: Dict[str, np.ndarray]) -> None:
    """Create comprehensive visualization dashboard.
    
    Args:
        returns: Returns DataFrame.
        weights_dict: Dictionary of optimization results.
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATION DASHBOARD")
    print("="*60)
    
    # Create output directory
    output_dir = ensure_directory("assets")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Portfolio Optimization Dashboard", fontsize=16, fontweight='bold')
    
    # 1. Portfolio Weights Comparison
    ax1 = axes[0, 0]
    methods = []
    weight_data = []
    
    for method, weights in weights_dict.items():
        if weights is not None:
            methods.append(method.replace("_", " ").title())
            weight_data.append(weights)
    
    if weight_data:
        weight_df = pd.DataFrame(weight_data, index=methods, columns=returns.columns)
        weight_df.plot(kind='bar', ax=ax1, stacked=True)
        ax1.set_title("Portfolio Weights Comparison")
        ax1.set_ylabel("Weight")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cumulative Returns Comparison
    ax2 = axes[0, 1]
    for method, weights in weights_dict.items():
        if weights is not None:
            portfolio_returns = returns.dot(weights)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            ax2.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=method.replace("_", " ").title(), linewidth=2)
    
    ax2.set_title("Cumulative Returns Comparison")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk-Return Scatter Plot
    ax3 = axes[1, 0]
    returns_data = []
    volatility_data = []
    method_names = []
    
    for method, weights in weights_dict.items():
        if weights is not None:
            portfolio_returns = returns.dot(weights)
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            returns_data.append(annual_return)
            volatility_data.append(annual_volatility)
            method_names.append(method.replace("_", " ").title())
    
    scatter = ax3.scatter(volatility_data, returns_data, s=100, alpha=0.7)
    ax3.set_xlabel("Volatility")
    ax3.set_ylabel("Annualized Return")
    ax3.set_title("Risk-Return Profile")
    ax3.grid(True, alpha=0.3)
    
    # Add labels to points
    for i, method in enumerate(method_names):
        ax3.annotate(method, (volatility_data[i], returns_data[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Correlation Matrix
    ax4 = axes[1, 1]
    correlation_matrix = returns.corr()
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(returns.columns)))
    ax4.set_yticks(range(len(returns.columns)))
    ax4.set_xticklabels(returns.columns, rotation=45)
    ax4.set_yticklabels(returns.columns)
    ax4.set_title("Asset Correlation Matrix")
    
    # Add colorbar
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_dashboard.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Dashboard saved to: {output_dir / 'optimization_dashboard.png'}")


def main():
    """Main function to run the portfolio optimization demonstration."""
    parser = argparse.ArgumentParser(description="Portfolio Optimization with AI")
    parser.add_argument("--config", type=str, default="configs/optimization.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--skip-backtest", action="store_true", 
                       help="Skip backtesting")
    parser.add_argument("--skip-walk-forward", action="store_true", 
                       help="Skip walk-forward analysis")
    parser.add_argument("--skip-visualization", action="store_true", 
                       help="Skip visualization dashboard")
    
    args = parser.parse_args()
    
    # Print disclaimer
    print("="*80)
    print("DISCLAIMER: This is a research and educational demonstration only.")
    print("NOT investment advice. All results are hypothetical.")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {}
    
    # Load data
    data = load_sample_data()
    returns = prepare_returns_data(data["market_data"])
    
    # Run optimization comparison
    weights_dict = run_optimization_comparison(returns, config)
    
    # Compare results
    comparison_df = compare_optimization_results(returns, weights_dict)
    
    # Run comprehensive backtesting
    if not args.skip_backtest:
        run_comprehensive_backtest(returns, weights_dict, config)
    
    # Run walk-forward analysis
    if not args.skip_walk_forward:
        run_walk_forward_analysis(returns, config)
    
    # Create visualization dashboard
    if not args.skip_visualization:
        create_visualization_dashboard(returns, weights_dict)
    
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nAll results have been saved to the 'assets' directory.")
    print("\nRemember: This is for research and educational purposes only.")
    print("NOT investment advice. Consult qualified financial professionals.")


if __name__ == "__main__":
    main()
