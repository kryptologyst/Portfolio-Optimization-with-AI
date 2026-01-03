"""Tests for portfolio optimization framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import modules to test
from src.data import DataLoader
from src.models import (
    GeneticAlgorithmOptimizer, 
    MeanVarianceOptimizer, 
    RiskParityOptimizer
)
from src.backtest import BacktestEngine
from src.utils import (
    set_seed, calculate_returns, calculate_volatility, 
    calculate_sharpe_ratio, calculate_max_drawdown
)


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        assert isinstance(loader.config, dict)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = DataLoader()
        symbols = ["AAPL", "GOOGL", "MSFT"]
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        data = loader.generate_synthetic_data(symbols, start_date, end_date)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(f"{symbol}_Close" in data.columns for symbol in symbols)
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        loader = DataLoader()
        data = loader.load_sample_data()
        
        assert "market_data" in data
        assert "fundamentals" in data
        assert "labels" in data
        
        assert isinstance(data["market_data"], pd.DataFrame)
        assert isinstance(data["fundamentals"], pd.DataFrame)
        assert isinstance(data["labels"], pd.DataFrame)


class TestPortfolioOptimizers:
    """Test cases for portfolio optimizers."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        set_seed(42)
        
        # Generate synthetic returns
        n_assets = 5
        n_periods = 252
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f"Asset_{i}" for i in range(n_assets)],
            index=pd.date_range("2020-01-01", periods=n_periods, freq="D")
        )
        
        return returns
    
    def test_genetic_algorithm_optimizer(self, sample_returns):
        """Test genetic algorithm optimizer."""
        config = {
            "genetic_algorithm": {
                "population_size": 50,
                "generations": 50,
                "mutation_rate": 0.1
            }
        }
        
        optimizer = GeneticAlgorithmOptimizer(config)
        weights = optimizer.optimize(sample_returns)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= 0)
    
    def test_mean_variance_optimizer(self, sample_returns):
        """Test mean-variance optimizer."""
        config = {
            "mean_variance": {
                "risk_aversion": 1.0,
                "constraints": {"max_weight": 0.4}
            }
        }
        
        optimizer = MeanVarianceOptimizer(config)
        weights = optimizer.optimize(sample_returns)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= 0)
        assert np.all(weights <= 0.4)
    
    def test_risk_parity_optimizer(self, sample_returns):
        """Test risk parity optimizer."""
        config = {
            "risk_parity": {
                "target_risk": 0.15,
                "max_iterations": 100
            }
        }
        
        optimizer = RiskParityOptimizer(config)
        weights = optimizer.optimize(sample_returns)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(sample_returns.columns)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= 0)
    
    def test_optimizer_evaluation(self, sample_returns):
        """Test optimizer evaluation method."""
        config = {"genetic_algorithm": {"population_size": 50, "generations": 50}}
        
        optimizer = GeneticAlgorithmOptimizer(config)
        weights = optimizer.optimize(sample_returns)
        metrics = optimizer.evaluate(sample_returns, weights)
        
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics


class TestBacktestEngine:
    """Test cases for BacktestEngine."""
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample data for backtesting."""
        set_seed(42)
        
        n_assets = 3
        n_periods = 100
        
        returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_periods, n_assets)),
            columns=[f"Asset_{i}" for i in range(n_assets)],
            index=pd.date_range("2020-01-01", periods=n_periods, freq="D")
        )
        
        weights = np.array([0.4, 0.3, 0.3])
        
        return returns, weights
    
    def test_backtest_engine_initialization(self):
        """Test BacktestEngine initialization."""
        config = {
            "backtesting": {
                "initial_capital": 1000000,
                "transaction_costs": {"commission": 0.001}
            }
        }
        
        engine = BacktestEngine(config)
        assert engine is not None
        assert engine.initial_capital == 1000000
        assert engine.commission == 0.001
    
    def test_run_backtest(self, sample_backtest_data):
        """Test running backtest."""
        returns, weights = sample_backtest_data
        
        config = {
            "backtesting": {
                "initial_capital": 1000000,
                "transaction_costs": {"commission": 0.001}
            }
        }
        
        engine = BacktestEngine(config)
        results = engine.run_backtest(returns, weights)
        
        assert isinstance(results, dict)
        assert "portfolio_returns" in results
        assert "performance_metrics" in results
        assert "trades" in results
        
        # Check portfolio returns
        portfolio_returns = results["portfolio_returns"]
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(returns)
        
        # Check performance metrics
        metrics = results["performance_metrics"]
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
    
    def test_performance_metrics_calculation(self, sample_backtest_data):
        """Test performance metrics calculation."""
        returns, weights = sample_backtest_data
        
        config = {"backtesting": {"initial_capital": 1000000}}
        engine = BacktestEngine(config)
        
        portfolio_returns = returns.dot(weights)
        metrics = engine._calculate_performance_metrics(portfolio_returns)
        
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        
        # Check that metrics are reasonable
        assert metrics["total_return"] > -1  # Can't lose more than 100%
        assert metrics["volatility"] >= 0
        assert metrics["max_drawdown"] <= 0


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        random1 = np.random.random()
        
        set_seed(42)
        random2 = np.random.random()
        
        assert random1 == random2
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = pd.Series([100, 105, 110, 108, 112])
        
        simple_returns = calculate_returns(prices, method="simple")
        log_returns = calculate_returns(prices, method="log")
        
        assert len(simple_returns) == len(prices) - 1
        assert len(log_returns) == len(prices) - 1
        
        # Check first simple return
        expected_simple = (105 - 100) / 100
        assert np.isclose(simple_returns.iloc[0], expected_simple)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        
        vol = calculate_volatility(returns, window=3)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(returns)
        assert vol.iloc[0] == np.nan  # First values should be NaN
        assert vol.iloc[-1] > 0  # Last value should be positive
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 50)  # More data points
        
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])
        
        max_dd, peak_date, trough_date = calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert isinstance(peak_date, pd.Timestamp)
        assert isinstance(trough_date, pd.Timestamp)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        set_seed(42)
        
        # Generate sample data
        loader = DataLoader()
        symbols = ["AAPL", "GOOGL", "MSFT"]
        data = loader.generate_synthetic_data(symbols, "2020-01-01", "2020-12-31")
        
        # Prepare returns
        close_prices = {}
        for col in data.columns:
            if col.endswith('_Close'):
                asset = col.split('_')[0]
                close_prices[asset] = data[col]
        
        returns = pd.DataFrame(close_prices).pct_change().dropna()
        
        # Run optimization
        config = {"genetic_algorithm": {"population_size": 50, "generations": 50}}
        optimizer = GeneticAlgorithmOptimizer(config)
        weights = optimizer.optimize(returns)
        
        # Run backtest
        backtest_config = {"backtesting": {"initial_capital": 1000000}}
        engine = BacktestEngine(backtest_config)
        results = engine.run_backtest(returns, weights)
        
        # Verify results
        assert isinstance(results, dict)
        assert "portfolio_returns" in results
        assert "performance_metrics" in results
        
        # Check that portfolio weights are valid
        assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= 0)
        
        # Check that performance metrics are reasonable
        metrics = results["performance_metrics"]
        assert metrics["total_return"] > -1
        assert metrics["volatility"] >= 0
        assert metrics["max_drawdown"] <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
