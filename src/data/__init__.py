"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime, timedelta

from ..utils import set_seed, validate_dataframe, calculate_returns


class DataLoader:
    """Data loader for portfolio optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data loader.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        set_seed()
    
    def load_yfinance_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Data frequency ('1d', '1wk', '1mo').
            
        Returns:
            DataFrame with OHLCV data.
        """
        try:
            data = yf.download(symbols, start=start_date, end=end_date, interval=frequency)
            
            if len(symbols) == 1:
                # Single symbol returns Series, convert to DataFrame
                data = data.to_frame()
                data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])
            
            # Flatten MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [f"{symbol}_{col}" for symbol, col in data.columns]
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from Yahoo Finance: {e}")
    
    def generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "1d"
    ) -> pd.DataFrame:
        """Generate synthetic market data.
        
        Args:
            symbols: List of stock symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Data frequency.
            
        Returns:
            DataFrame with synthetic OHLCV data.
        """
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        data = {}
        
        for symbol in symbols:
            # Generate synthetic price series using geometric Brownian motion
            np.random.seed(hash(symbol) % 2**32)  # Deterministic seed per symbol
            
            # Parameters for GBM
            mu = np.random.uniform(0.05, 0.15)  # Annual drift
            sigma = np.random.uniform(0.15, 0.35)  # Annual volatility
            dt = 1/252  # Daily time step
            
            # Generate price path
            prices = [100]  # Starting price
            for _ in range(len(date_range) - 1):
                dW = np.random.normal(0, np.sqrt(dt))
                dS = mu * dt + sigma * dW
                prices.append(prices[-1] * (1 + dS))
            
            prices = np.array(prices)
            
            # Generate OHLC from close prices
            high_multiplier = np.random.uniform(1.001, 1.02, len(prices))
            low_multiplier = np.random.uniform(0.98, 0.999, len(prices))
            
            data[f"{symbol}_Open"] = prices * np.random.uniform(0.99, 1.01, len(prices))
            data[f"{symbol}_High"] = prices * high_multiplier
            data[f"{symbol}_Low"] = prices * low_multiplier
            data[f"{symbol}_Close"] = prices
            data[f"{symbol}_Adj Close"] = prices
            data[f"{symbol}_Volume"] = np.random.randint(1000000, 10000000, len(prices))
        
        df = pd.DataFrame(data, index=date_range)
        return df
    
    def load_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Load sample data for demonstration.
        
        Returns:
            Dictionary containing market data, fundamentals, and labels.
        """
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        start_date = "2015-01-01"
        end_date = "2024-01-01"
        
        # Load market data
        market_data = self.load_yfinance_data(symbols, start_date, end_date)
        
        # Generate synthetic fundamentals
        fundamentals = self._generate_synthetic_fundamentals(symbols, start_date, end_date)
        
        # Generate labels
        labels = self._generate_labels(market_data, symbols)
        
        return {
            "market_data": market_data,
            "fundamentals": fundamentals,
            "labels": labels
        }
    
    def _generate_synthetic_fundamentals(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic fundamental data.
        
        Args:
            symbols: List of stock symbols.
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with fundamental data.
        """
        # Generate quarterly dates
        quarters = pd.date_range(start=start_date, end=end_date, freq="Q")
        
        data = []
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            
            for quarter in quarters:
                data.append({
                    "symbol": symbol,
                    "date": quarter,
                    "market_cap": np.random.uniform(1e9, 1e12),
                    "pe_ratio": np.random.uniform(10, 50),
                    "pb_ratio": np.random.uniform(1, 10),
                    "debt_to_equity": np.random.uniform(0.1, 2.0),
                    "roe": np.random.uniform(0.05, 0.25),
                    "roa": np.random.uniform(0.02, 0.15),
                    "revenue_growth": np.random.uniform(-0.1, 0.3),
                    "earnings_growth": np.random.uniform(-0.2, 0.4)
                })
        
        return pd.DataFrame(data)
    
    def _generate_labels(
        self, 
        market_data: pd.DataFrame, 
        symbols: List[str]
    ) -> pd.DataFrame:
        """Generate labels for machine learning.
        
        Args:
            market_data: Market data DataFrame.
            symbols: List of stock symbols.
            
        Returns:
            DataFrame with labels.
        """
        labels_data = []
        
        for symbol in symbols:
            close_col = f"{symbol}_Close"
            if close_col not in market_data.columns:
                close_col = f"{symbol}_Adj Close"
            
            if close_col in market_data.columns:
                prices = market_data[close_col].dropna()
                returns = calculate_returns(prices)
                
                for i in range(len(returns)):
                    if i >= 21:  # Need at least 21 days for volatility calculation
                        date = returns.index[i]
                        
                        # Forward returns
                        forward_1d = returns.iloc[i:i+1].mean() if i < len(returns) - 1 else np.nan
                        forward_5d = returns.iloc[i:i+5].mean() if i < len(returns) - 5 else np.nan
                        forward_21d = returns.iloc[i:i+21].mean() if i < len(returns) - 21 else np.nan
                        
                        # Volatility
                        volatility_21d = returns.iloc[i-20:i+1].std() * np.sqrt(252)
                        
                        # Risk labels
                        high_volatility = 1 if volatility_21d > 0.3 else 0
                        
                        labels_data.append({
                            "datetime": date,
                            "symbol": symbol,
                            "forward_return_1d": forward_1d,
                            "forward_return_5d": forward_5d,
                            "forward_return_21d": forward_21d,
                            "volatility_21d": volatility_21d,
                            "high_volatility": high_volatility
                        })
        
        return pd.DataFrame(labels_data).dropna()
    
    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """Preprocess data to handle missing values.
        
        Args:
            data: Input DataFrame.
            method: Method to handle missing values.
            
        Returns:
            Preprocessed DataFrame.
        """
        if method == "forward_fill":
            return data.fillna(method="ffill").dropna()
        elif method == "backward_fill":
            return data.fillna(method="bfill").dropna()
        elif method == "interpolate":
            return data.interpolate().dropna()
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def detect_outliers(
        self, 
        data: pd.Series, 
        method: str = "iqr", 
        threshold: float = 3.0
    ) -> pd.Series:
        """Detect outliers in data.
        
        Args:
            data: Input series.
            method: Outlier detection method.
            threshold: Threshold for outlier detection.
            
        Returns:
            Boolean series indicating outliers.
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        
        elif method == "winsorize":
            # Simple winsorization
            lower_percentile = threshold * 100
            upper_percentile = (1 - threshold) * 100
            lower_bound = data.quantile(lower_percentile / 100)
            upper_bound = data.quantile(upper_percentile / 100)
            return (data < lower_bound) | (data > upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def save_data(self, data: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save data to file.
        
        Args:
            data: DataFrame to save.
            filepath: Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == ".csv":
            data.to_csv(filepath)
        elif filepath.suffix == ".parquet":
            data.to_parquet(filepath)
        elif filepath.suffix == ".h5":
            data.to_hdf(filepath, key="data", mode="w")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            filepath: Input file path.
            
        Returns:
            Loaded DataFrame.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == ".csv":
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        elif filepath.suffix == ".h5":
            return pd.read_hdf(filepath, key="data")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
