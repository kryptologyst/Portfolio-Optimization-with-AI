# Portfolio Optimization with AI

**DISCLAIMER: This is a research and educational demonstration only. NOT investment advice.**

A comprehensive portfolio optimization framework implementing multiple optimization techniques including genetic algorithms, mean-variance optimization, risk parity, and machine learning approaches.

## Features

- **Multiple Optimization Methods**: Genetic algorithms, mean-variance, Black-Litterman, risk parity
- **Risk Management**: VaR, CVaR, drawdown control, stress testing
- **Machine Learning**: XGBoost, LightGBM for return prediction and risk modeling
- **Backtesting**: Comprehensive backtesting with transaction costs and slippage
- **Visualization**: Interactive dashboards and comprehensive performance analytics
- **Explainability**: SHAP explanations for model decisions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Portfolio-Optimization-with-AI.git
cd Portfolio-Optimization-with-AI

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.optimizers import PortfolioOptimizer
from src.data.loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_sample_data()

# Initialize optimizer
optimizer = PortfolioOptimizer()

# Run optimization
weights = optimizer.optimize(data, method="genetic_algorithm")

# Evaluate performance
results = optimizer.evaluate(data, weights)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── features/       # Feature engineering
├── labels/         # Label generation
├── models/         # Optimization and ML models
├── backtest/       # Backtesting framework
├── risk/           # Risk management tools
└── utils/          # Utility functions

configs/            # Configuration files
scripts/            # Training and evaluation scripts
notebooks/          # Jupyter notebooks for analysis
tests/              # Unit tests
assets/             # Generated plots and results
demo/               # Interactive demos
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/data.yaml`: Data source and preprocessing settings
- `configs/optimization.yaml`: Optimization parameters
- `configs/risk.yaml`: Risk management settings
- `configs/backtest.yaml`: Backtesting parameters

## Data Schema

### Market Data (`market_data.csv`)
- `datetime`: Timestamp
- `symbol`: Asset symbol
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `adj_close`: Adjusted close price

### Fundamentals (`fundamentals.csv`)
- `symbol`: Asset symbol
- `date`: Reporting date
- `market_cap`, `pe_ratio`, `pb_ratio`: Financial ratios
- Additional fundamental metrics

### Labels (`labels.csv`)
- `datetime`: Timestamp
- `symbol`: Asset symbol
- `forward_return_1d`, `forward_return_5d`, `forward_return_21d`: Forward returns
- `volatility_21d`: Rolling volatility

## Optimization Methods

### 1. Genetic Algorithm
- Population-based optimization
- Customizable fitness functions
- Mutation and crossover operators

### 2. Mean-Variance Optimization
- Markowitz portfolio theory
- Efficient frontier calculation
- Constraint handling

### 3. Risk Parity
- Equal risk contribution
- Diversification-focused
- Lower concentration risk

### 4. Black-Litterman
- Bayesian approach
- Incorporates market views
- More stable than pure mean-variance

### 5. Machine Learning Approaches
- XGBoost for return prediction
- Risk model with LightGBM
- Ensemble methods

## Risk Management

- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo methods
- **Conditional VaR**: Expected shortfall calculation
- **Drawdown Control**: Maximum drawdown limits
- **Stress Testing**: Scenario analysis
- **Risk Budgeting**: Risk allocation across assets

## Evaluation Metrics

### Financial Metrics
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Volatility
- Information Ratio, Alpha, Beta
- Hit Rate, Average Trade P&L

### Machine Learning Metrics
- RMSE, MAE for return prediction
- AUC, Precision, Recall for classification
- R² for regression tasks

## Backtesting

Comprehensive backtesting framework with:
- Transaction costs and slippage modeling
- Realistic execution simulation
- Performance attribution
- Risk-adjusted returns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

**IMPORTANT: This software is for research and educational purposes only.**

This project is designed for academic research, educational purposes, and demonstration of portfolio optimization techniques. It is NOT intended as investment advice, financial guidance, or a recommendation to buy, sell, or hold any securities.

- All results are hypothetical and based on historical data
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- Consult qualified financial professionals before making investment decisions

Please read the full [DISCLAIMER.md](DISCLAIMER.md) for complete details.
# Portfolio-Optimization-with-AI
