"""Utility functions for portfolio optimization."""

import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import yaml
from omegaconf import OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Get the best available device for computation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    # Fallback to CPU
    return "cpu"


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_omegaconf(config_path: Union[str, Path]) -> OmegaConf:
    """Load configuration using OmegaConf.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        
    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series.
        method: Return calculation method ('simple' or 'log').
        
    Returns:
        Returns series.
    """
    if method == "simple":
        return prices.pct_change().dropna()
    elif method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown return method: {method}")


def calculate_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility.
    
    Args:
        returns: Returns series.
        window: Rolling window size.
        annualize: Whether to annualize volatility.
        
    Returns:
        Volatility series.
    """
    volatility = returns.rolling(window=window).std()
    if annualize:
        volatility *= np.sqrt(252)
    return volatility


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series.
        risk_free_rate: Risk-free rate (annualized).
        
    Returns:
        Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown.
    
    Args:
        returns: Returns series.
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = cumulative.loc[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk.
    
    Args:
        returns: Returns series.
        confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR).
        
    Returns:
        VaR value.
    """
    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).
    
    Args:
        returns: Returns series.
        confidence_level: CVaR confidence level (e.g., 0.05 for 95% CVaR).
        
    Returns:
        CVaR value.
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Calculate correlation matrix.
    
    Args:
        returns: Returns DataFrame.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        
    Returns:
        Correlation matrix.
    """
    return returns.corr(method=method)


def calculate_covariance_matrix(returns: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """Calculate covariance matrix.
    
    Args:
        returns: Returns DataFrame.
        method: Covariance method ('sample', 'shrinkage').
        
    Returns:
        Covariance matrix.
    """
    if method == "sample":
        return returns.cov()
    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage estimator
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns.dropna()).covariance_
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    else:
        raise ValueError(f"Unknown covariance method: {method}")


def create_efficient_frontier(
    returns: pd.DataFrame, 
    risk_free_rate: float = 0.02,
    num_portfolios: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create efficient frontier.
    
    Args:
        returns: Returns DataFrame.
        risk_free_rate: Risk-free rate.
        num_portfolios: Number of portfolios to generate.
        
    Returns:
        Tuple of (returns, volatilities, sharpe_ratios).
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
        portfolio_sharpe_ratios.append((portfolio_return - risk_free_rate) / portfolio_volatility)
    
    return np.array(portfolio_returns), np.array(portfolio_volatilities), np.array(portfolio_sharpe_ratios)


def suppress_warnings() -> None:
    """Suppress common warnings."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_currency(value: float, currency: str = "USD") -> str:
    """Format value as currency.
    
    Args:
        value: Value to format.
        currency: Currency code.
        
    Returns:
        Formatted currency string.
    """
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage.
    
    Args:
        value: Value to format.
        decimals: Number of decimal places.
        
    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"
