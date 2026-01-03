"""Backtesting framework for portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils import (
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_var, calculate_cvar, format_currency, format_percentage
)


class BacktestEngine:
    """Backtesting engine for portfolio strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.backtest_config = self.config.get("backtesting", {})
        
        self.initial_capital = self.backtest_config.get("initial_capital", 1000000)
        self.commission = self.backtest_config.get("transaction_costs", {}).get("commission", 0.001)
        self.slippage = self.backtest_config.get("transaction_costs", {}).get("slippage", 0.0005)
        self.market_impact = self.backtest_config.get("transaction_costs", {}).get("market_impact", 0.0001)
        
        self.results = None
        self.trades = None
        self.performance_metrics = None
    
    def run_backtest(
        self,
        returns: pd.DataFrame,
        weights: Union[np.ndarray, pd.DataFrame],
        benchmark_returns: Optional[pd.Series] = None,
        rebalancing_frequency: str = "monthly"
    ) -> Dict[str, Any]:
        """Run backtest.
        
        Args:
            returns: Returns DataFrame.
            weights: Portfolio weights (array or DataFrame with time-varying weights).
            benchmark_returns: Benchmark returns for comparison.
            rebalancing_frequency: Rebalancing frequency.
            
        Returns:
            Dictionary containing backtest results.
        """
        if isinstance(weights, np.ndarray):
            # Static weights
            weights_df = pd.DataFrame(
                np.tile(weights, (len(returns), 1)),
                index=returns.index,
                columns=returns.columns
            )
        else:
            weights_df = weights
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(returns, weights_df, rebalancing_frequency)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_returns, benchmark_returns)
        
        # Generate trades
        trades = self._generate_trades(returns, weights_df, rebalancing_frequency)
        
        # Store results
        self.results = {
            "portfolio_returns": portfolio_returns,
            "weights": weights_df,
            "trades": trades,
            "performance_metrics": performance_metrics,
            "benchmark_returns": benchmark_returns
        }
        
        return self.results
    
    def _calculate_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        rebalancing_frequency: str
    ) -> pd.Series:
        """Calculate portfolio returns with rebalancing.
        
        Args:
            returns: Returns DataFrame.
            weights: Portfolio weights DataFrame.
            rebalancing_frequency: Rebalancing frequency.
            
        Returns:
            Portfolio returns series.
        """
        # Align weights with returns
        weights_aligned = weights.reindex(returns.index, method="ffill")
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights_aligned).sum(axis=1)
        
        # Apply transaction costs for rebalancing
        if rebalancing_frequency != "daily":
            portfolio_returns = self._apply_transaction_costs(
                portfolio_returns, weights_aligned, rebalancing_frequency
            )
        
        return portfolio_returns
    
    def _apply_transaction_costs(
        self,
        portfolio_returns: pd.Series,
        weights: pd.DataFrame,
        rebalancing_frequency: str
    ) -> pd.Series:
        """Apply transaction costs for rebalancing.
        
        Args:
            portfolio_returns: Portfolio returns.
            weights: Portfolio weights.
            rebalancing_frequency: Rebalancing frequency.
            
        Returns:
            Portfolio returns with transaction costs.
        """
        # Determine rebalancing dates
        if rebalancing_frequency == "monthly":
            rebalance_dates = weights.groupby(weights.index.to_period("M")).first().index
        elif rebalancing_frequency == "weekly":
            rebalance_dates = weights.groupby(weights.index.to_period("W")).first().index
        elif rebalancing_frequency == "quarterly":
            rebalance_dates = weights.groupby(weights.index.to_period("Q")).first().index
        else:
            rebalance_dates = weights.index
        
        # Calculate turnover and transaction costs
        adjusted_returns = portfolio_returns.copy()
        
        for i, date in enumerate(rebalance_dates):
            if i > 0:
                prev_date = rebalance_dates[i-1]
                prev_weights = weights.loc[prev_date]
                curr_weights = weights.loc[date]
                
                # Calculate turnover
                turnover = np.sum(np.abs(curr_weights - prev_weights))
                
                # Calculate transaction costs
                transaction_cost = turnover * (self.commission + self.slippage + self.market_impact)
                
                # Apply cost to return
                if date in adjusted_returns.index:
                    adjusted_returns.loc[date] -= transaction_cost
        
        return adjusted_returns
    
    def _calculate_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            portfolio_returns: Portfolio returns.
            benchmark_returns: Benchmark returns.
            
        Returns:
            Dictionary of performance metrics.
        """
        metrics = {}
        
        # Return metrics
        metrics["total_return"] = (1 + portfolio_returns).prod() - 1
        metrics["annualized_return"] = portfolio_returns.mean() * 252
        metrics["cumulative_return"] = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        
        # Risk metrics
        metrics["volatility"] = portfolio_returns.std() * np.sqrt(252)
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(portfolio_returns)
        metrics["max_drawdown"] = calculate_max_drawdown(portfolio_returns)[0]
        metrics["var_95"] = calculate_var(portfolio_returns, 0.05)
        metrics["cvar_95"] = calculate_cvar(portfolio_returns, 0.05)
        
        # Additional risk metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        metrics["sortino_ratio"] = (
            metrics["annualized_return"] / (downside_returns.std() * np.sqrt(252))
            if len(downside_returns) > 0 else np.nan
        )
        
        # Calmar ratio
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / abs(metrics["max_drawdown"])
            if metrics["max_drawdown"] != 0 else np.nan
        )
        
        # Trade metrics
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        
        metrics["win_rate"] = len(positive_returns) / len(portfolio_returns)
        metrics["avg_winning_return"] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics["avg_losing_return"] = negative_returns.mean() if len(negative_returns) > 0 else 0
        metrics["profit_factor"] = (
            abs(positive_returns.sum() / negative_returns.sum())
            if len(negative_returns) > 0 and negative_returns.sum() != 0 else np.inf
        )
        
        # Benchmark comparison
        if benchmark_returns is not None:
            # Align benchmark returns
            benchmark_aligned = benchmark_returns.reindex(portfolio_returns.index, method="ffill")
            
            # Excess returns
            excess_returns = portfolio_returns - benchmark_aligned
            metrics["excess_return"] = excess_returns.mean() * 252
            
            # Tracking error
            metrics["tracking_error"] = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            metrics["information_ratio"] = (
                metrics["excess_return"] / metrics["tracking_error"]
                if metrics["tracking_error"] != 0 else np.nan
            )
            
            # Beta
            covariance = np.cov(portfolio_returns, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            metrics["beta"] = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
            
            # Alpha
            risk_free_rate = 0.02
            metrics["alpha"] = (
                metrics["annualized_return"] - risk_free_rate - 
                metrics["beta"] * (benchmark_aligned.mean() * 252 - risk_free_rate)
            )
        
        return metrics
    
    def _generate_trades(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        rebalancing_frequency: str
    ) -> pd.DataFrame:
        """Generate trade log.
        
        Args:
            returns: Returns DataFrame.
            weights: Portfolio weights DataFrame.
            rebalancing_frequency: Rebalancing frequency.
            
        Returns:
            DataFrame with trade information.
        """
        trades = []
        
        # Determine rebalancing dates
        if rebalancing_frequency == "monthly":
            rebalance_dates = weights.groupby(weights.index.to_period("M")).first().index
        elif rebalancing_frequency == "weekly":
            rebalance_dates = weights.groupby(weights.index.to_period("W")).first().index
        elif rebalancing_frequency == "quarterly":
            rebalance_dates = weights.groupby(weights.index.to_period("Q")).first().index
        else:
            rebalance_dates = weights.index
        
        for i, date in enumerate(rebalance_dates):
            if i > 0:
                prev_date = rebalance_dates[i-1]
                prev_weights = weights.loc[prev_date]
                curr_weights = weights.loc[date]
                
                # Calculate weight changes
                weight_changes = curr_weights - prev_weights
                
                # Record trades
                for asset, change in weight_changes.items():
                    if abs(change) > 1e-6:  # Only record significant changes
                        trades.append({
                            "date": date,
                            "asset": asset,
                            "weight_change": change,
                            "prev_weight": prev_weights[asset],
                            "curr_weight": curr_weights[asset],
                            "trade_size": abs(change),
                            "direction": "buy" if change > 0 else "sell"
                        })
        
        return pd.DataFrame(trades)
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot backtest results.
        
        Args:
            save_path: Path to save plots.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        portfolio_returns = self.results["portfolio_returns"]
        benchmark_returns = self.results.get("benchmark_returns")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Portfolio Backtest Results", fontsize=16)
        
        # Equity curve
        ax1 = axes[0, 0]
        portfolio_value = (1 + portfolio_returns).cumprod() * self.initial_capital
        ax1.plot(portfolio_value.index, portfolio_value.values, label="Portfolio", linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_aligned = benchmark_returns.reindex(portfolio_returns.index, method="ffill")
            benchmark_value = (1 + benchmark_aligned).cumprod() * self.initial_capital
            ax1.plot(benchmark_value.index, benchmark_value.values, label="Benchmark", linewidth=2)
        
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[0, 1]
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=2)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax3 = axes[1, 0]
        rolling_sharpe = portfolio_returns.rolling(252).apply(
            lambda x: calculate_sharpe_ratio(x), raw=False
        )
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax3.set_title("Rolling Sharpe Ratio (252 days)")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax4 = axes[1, 1]
        rolling_vol = portfolio_returns.rolling(252).std() * np.sqrt(252)
        ax4.plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        ax4.set_title("Rolling Volatility (252 days)")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate text report of backtest results.
        
        Returns:
            Formatted report string.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        metrics = self.results["performance_metrics"]
        
        report = f"""
Portfolio Backtest Report
========================

Initial Capital: {format_currency(self.initial_capital)}
Final Value: {format_currency(self.initial_capital * (1 + metrics["total_return"]))}

Return Metrics:
--------------
Total Return: {format_percentage(metrics["total_return"])}
Annualized Return: {format_percentage(metrics["annualized_return"])}
Cumulative Return: {format_percentage(metrics["cumulative_return"])}

Risk Metrics:
-------------
Volatility: {format_percentage(metrics["volatility"])}
Sharpe Ratio: {metrics["sharpe_ratio"]:.3f}
Sortino Ratio: {metrics["sortino_ratio"]:.3f}
Calmar Ratio: {metrics["calmar_ratio"]:.3f}
Maximum Drawdown: {format_percentage(metrics["max_drawdown"])}
VaR (95%): {format_percentage(metrics["var_95"])}
CVaR (95%): {format_percentage(metrics["cvar_95"])}

Trade Metrics:
--------------
Win Rate: {format_percentage(metrics["win_rate"])}
Average Winning Return: {format_percentage(metrics["avg_winning_return"])}
Average Losing Return: {format_percentage(metrics["avg_losing_return"])}
Profit Factor: {metrics["profit_factor"]:.3f}
"""
        
        if "excess_return" in metrics:
            report += f"""
Benchmark Comparison:
--------------------
Excess Return: {format_percentage(metrics["excess_return"])}
Tracking Error: {format_percentage(metrics["tracking_error"])}
Information Ratio: {metrics["information_ratio"]:.3f}
Beta: {metrics["beta"]:.3f}
Alpha: {format_percentage(metrics["alpha"])}
"""
        
        return report
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save backtest results to file.
        
        Args:
            filepath: Output file path.
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == ".csv":
            # Save portfolio returns
            self.results["portfolio_returns"].to_csv(filepath)
        elif filepath.suffix == ".json":
            import json
            # Convert results to JSON-serializable format
            results_json = {
                "performance_metrics": self.results["performance_metrics"],
                "portfolio_returns": self.results["portfolio_returns"].to_dict(),
                "trades": self.results["trades"].to_dict("records") if self.results["trades"] is not None else []
            }
            with open(filepath, 'w') as f:
                json.dump(results_json, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy validation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize walk-forward analysis.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.wf_config = self.config.get("walk_forward", {})
        
        self.train_period = self.wf_config.get("train_period", 252)
        self.test_period = self.wf_config.get("test_period", 63)
        self.step_size = self.wf_config.get("step_size", 21)
        self.min_train_period = self.wf_config.get("min_train_period", 126)
    
    def run_analysis(
        self,
        returns: pd.DataFrame,
        optimizer,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """Run walk-forward analysis.
        
        Args:
            returns: Returns DataFrame.
            optimizer: Portfolio optimizer instance.
            **optimizer_kwargs: Additional optimizer parameters.
            
        Returns:
            Dictionary containing walk-forward results.
        """
        results = []
        
        # Generate walk-forward periods
        periods = self._generate_periods(returns.index)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            print(f"Walk-forward period {i+1}/{len(periods)}: {train_start} to {test_end}")
            
            # Split data
            train_data = returns.loc[train_start:train_end]
            test_data = returns.loc[test_start:test_end]
            
            if len(train_data) < self.min_train_period:
                continue
            
            # Optimize on training data
            try:
                weights = optimizer.optimize(train_data, **optimizer_kwargs)
                
                # Test on out-of-sample data
                test_returns = test_data.dot(weights)
                
                # Calculate performance metrics
                metrics = self._calculate_period_metrics(test_returns)
                
                results.append({
                    "period": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "weights": weights,
                    "test_returns": test_returns,
                    "metrics": metrics
                })
                
            except Exception as e:
                print(f"Error in period {i+1}: {e}")
                continue
        
        return {
            "periods": results,
            "summary_metrics": self._calculate_summary_metrics(results)
        }
    
    def _generate_periods(self, index: pd.DatetimeIndex) -> List[Tuple]:
        """Generate walk-forward periods.
        
        Args:
            index: Date index.
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples.
        """
        periods = []
        
        start_date = index[0]
        end_date = index[-1]
        
        current_date = start_date
        
        while current_date < end_date:
            train_start = current_date
            train_end = current_date + pd.Timedelta(days=self.train_period)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=self.test_period)
            
            if test_end <= end_date:
                periods.append((train_start, train_end, test_start, test_end))
            
            current_date += pd.Timedelta(days=self.step_size)
        
        return periods
    
    def _calculate_period_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate metrics for a single period.
        
        Args:
            returns: Returns series.
            
        Returns:
            Dictionary of metrics.
        """
        return {
            "total_return": (1 + returns).prod() - 1,
            "annualized_return": returns.mean() * 252,
            "volatility": returns.std() * np.sqrt(252),
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "max_drawdown": calculate_max_drawdown(returns)[0]
        }
    
    def _calculate_summary_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate summary metrics across all periods.
        
        Args:
            results: List of period results.
            
        Returns:
            Dictionary of summary metrics.
        """
        if not results:
            return {}
        
        # Aggregate returns
        all_returns = pd.concat([r["test_returns"] for r in results])
        
        return {
            "total_periods": len(results),
            "total_return": (1 + all_returns).prod() - 1,
            "annualized_return": all_returns.mean() * 252,
            "volatility": all_returns.std() * np.sqrt(252),
            "sharpe_ratio": calculate_sharpe_ratio(all_returns),
            "max_drawdown": calculate_max_drawdown(all_returns)[0],
            "avg_period_return": np.mean([r["metrics"]["total_return"] for r in results]),
            "avg_period_sharpe": np.mean([r["metrics"]["sharpe_ratio"] for r in results])
        }
