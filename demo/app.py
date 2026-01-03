"""
Interactive Portfolio Optimization Demo using Streamlit

This demo provides an interactive interface for exploring portfolio optimization
methods, risk management, and backtesting results.

DISCLAIMER: This is for research and educational purposes only. NOT investment advice.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import our modules
from src.data import DataLoader
from src.models import (
    GeneticAlgorithmOptimizer, 
    MeanVarianceOptimizer, 
    RiskParityOptimizer,
    MachineLearningOptimizer
)
from src.backtest import BacktestEngine
from src.utils import (
    set_seed, load_config, ensure_directory, 
    format_currency, format_percentage
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization with AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Portfolio Optimization with AI</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ DISCLAIMER:</strong> This is a research and educational demonstration only. 
    <strong>NOT investment advice.</strong> All results are hypothetical and based on historical data. 
    Past performance does not guarantee future results. Consult qualified financial professionals 
    before making investment decisions.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Load configuration
try:
    config = load_config("configs/optimization.yaml")
except FileNotFoundError:
    config = {}

# Data parameters
st.sidebar.header("📊 Data Parameters")
symbols = st.sidebar.multiselect(
    "Select Assets",
    options=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    default=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Optimization parameters
st.sidebar.header("⚙️ Optimization Parameters")
optimization_method = st.sidebar.selectbox(
    "Optimization Method",
    options=["genetic_algorithm", "mean_variance", "risk_parity", "machine_learning"],
    format_func=lambda x: x.replace("_", " ").title()
)

# Method-specific parameters
if optimization_method == "genetic_algorithm":
    st.sidebar.subheader("Genetic Algorithm")
    population_size = st.sidebar.slider("Population Size", 50, 200, 100)
    generations = st.sidebar.slider("Generations", 50, 500, 200)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)
    
    # Update config
    config["genetic_algorithm"] = {
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate
    }

elif optimization_method == "mean_variance":
    st.sidebar.subheader("Mean-Variance")
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.1, 5.0, 1.0)
    max_weight = st.sidebar.slider("Max Weight per Asset", 0.1, 1.0, 0.4)
    
    config["mean_variance"] = {
        "risk_aversion": risk_aversion,
        "constraints": {"max_weight": max_weight}
    }

elif optimization_method == "risk_parity":
    st.sidebar.subheader("Risk Parity")
    target_risk = st.sidebar.slider("Target Risk", 0.05, 0.3, 0.15)
    
    config["risk_parity"] = {
        "target_risk": target_risk
    }

# Backtesting parameters
st.sidebar.header("📈 Backtesting Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 100000, 10000000, 1000000)
commission = st.sidebar.slider("Commission (bps)", 0, 50, 10) / 10000
rebalancing_freq = st.sidebar.selectbox("Rebalancing Frequency", ["daily", "weekly", "monthly", "quarterly"])

config["backtesting"] = {
    "initial_capital": initial_capital,
    "transaction_costs": {"commission": commission},
    "portfolio": {"rebalancing_frequency": rebalancing_freq}
}

# Main content
@st.cache_data
def load_and_prepare_data(symbols, start_date, end_date):
    """Load and prepare data for optimization."""
    set_seed(42)
    
    loader = DataLoader()
    
    # Load market data
    market_data = loader.load_yfinance_data(
        symbols, 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d")
    )
    
    # Prepare returns
    close_prices = {}
    for col in market_data.columns:
        if col.endswith('_Close') or col.endswith('_Adj Close'):
            asset = col.split('_')[0]
            close_prices[asset] = market_data[col]
    
    close_df = pd.DataFrame(close_prices)
    returns = close_df.pct_change().dropna()
    
    return market_data, returns

@st.cache_data
def run_optimization(returns, method, config):
    """Run portfolio optimization."""
    set_seed(42)
    
    if method == "genetic_algorithm":
        optimizer = GeneticAlgorithmOptimizer(config)
    elif method == "mean_variance":
        optimizer = MeanVarianceOptimizer(config)
    elif method == "risk_parity":
        optimizer = RiskParityOptimizer(config)
    elif method == "machine_learning":
        optimizer = MachineLearningOptimizer(config)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    weights = optimizer.optimize(returns)
    return weights

def create_performance_metrics(returns, weights):
    """Calculate performance metrics."""
    portfolio_returns = returns.dot(weights)
    
    metrics = {
        "Total Return": (1 + portfolio_returns).prod() - 1,
        "Annualized Return": portfolio_returns.mean() * 252,
        "Volatility": portfolio_returns.std() * np.sqrt(252),
        "Sharpe Ratio": portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
        "Max Drawdown": calculate_max_drawdown(portfolio_returns)[0]
    }
    
    return metrics, portfolio_returns

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def create_portfolio_weights_chart(weights, assets):
    """Create portfolio weights chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=assets,
            y=weights,
            text=[f"{w:.1%}" for w in weights],
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Portfolio Weights",
        xaxis_title="Assets",
        yaxis_title="Weight",
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    return fig

def create_equity_curve_chart(portfolio_returns, initial_capital):
    """Create equity curve chart."""
    portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis=dict(tickformat='$,.0f'),
        height=400
    )
    
    return fig

def create_drawdown_chart(portfolio_returns):
    """Create drawdown chart."""
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='red'),
        fillcolor='rgba(255,0,0,0.3)'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    return fig

def create_risk_return_scatter(returns, weights):
    """Create risk-return scatter plot."""
    portfolio_returns = returns.dot(weights)
    
    # Calculate individual asset metrics
    asset_returns = returns.mean() * 252
    asset_volatility = returns.std() * np.sqrt(252)
    
    # Portfolio metrics
    portfolio_return = portfolio_returns.mean() * 252
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    
    fig = go.Figure()
    
    # Individual assets
    fig.add_trace(go.Scatter(
        x=asset_volatility,
        y=asset_returns,
        mode='markers+text',
        text=returns.columns,
        textposition="top center",
        name='Individual Assets',
        marker=dict(size=10, color='lightblue')
    ))
    
    # Portfolio
    fig.add_trace(go.Scatter(
        x=[portfolio_volatility],
        y=[portfolio_return],
        mode='markers+text',
        text=['Portfolio'],
        textposition="top center",
        name='Optimized Portfolio',
        marker=dict(size=15, color='red')
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat='.1%'),
        yaxis=dict(tickformat='.1%'),
        height=400
    )
    
    return fig

# Run the application
if st.button("🚀 Run Optimization", type="primary"):
    if not symbols:
        st.error("Please select at least one asset.")
    else:
        with st.spinner("Loading data and running optimization..."):
            # Load data
            market_data, returns = load_and_prepare_data(symbols, start_date, end_date)
            
            # Run optimization
            weights = run_optimization(returns, optimization_method, config)
            
            # Calculate metrics
            metrics, portfolio_returns = create_performance_metrics(returns, weights)
            
            # Display results
            st.success("✅ Optimization completed successfully!")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance", "🎯 Risk Analysis", "📋 Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Weights")
                    weights_chart = create_portfolio_weights_chart(weights, returns.columns)
                    st.plotly_chart(weights_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Risk-Return Profile")
                    scatter_chart = create_risk_return_scatter(returns, weights)
                    st.plotly_chart(scatter_chart, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Equity Curve")
                    equity_chart = create_equity_curve_chart(portfolio_returns, initial_capital)
                    st.plotly_chart(equity_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Drawdown")
                    drawdown_chart = create_drawdown_chart(portfolio_returns)
                    st.plotly_chart(drawdown_chart, use_container_width=True)
            
            with tab3:
                st.subheader("Risk Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics['Sharpe Ratio']:.3f}",
                        help="Risk-adjusted return measure"
                    )
                
                with col2:
                    st.metric(
                        "Max Drawdown",
                        format_percentage(metrics['Max Drawdown']),
                        help="Maximum peak-to-trough decline"
                    )
                
                with col3:
                    st.metric(
                        "Volatility",
                        format_percentage(metrics['Volatility']),
                        help="Annualized standard deviation of returns"
                    )
                
                with col4:
                    st.metric(
                        "Total Return",
                        format_percentage(metrics['Total Return']),
                        help="Total return over the period"
                    )
            
            with tab4:
                st.subheader("Detailed Results")
                
                # Performance metrics table
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                st.dataframe(metrics_df, use_container_width=True)
                
                # Portfolio weights table
                weights_df = pd.DataFrame({
                    'Asset': returns.columns,
                    'Weight': weights,
                    'Weight (%)': [f"{w:.1%}" for w in weights]
                })
                st.dataframe(weights_df, use_container_width=True)
                
                # Backtesting results
                st.subheader("Backtesting Results")
                
                try:
                    backtest_engine = BacktestEngine(config)
                    backtest_results = backtest_engine.run_backtest(
                        returns, 
                        weights, 
                        rebalancing_frequency=rebalancing_freq
                    )
                    
                    backtest_metrics = backtest_results["performance_metrics"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Return Metrics:**")
                        st.write(f"- Total Return: {format_percentage(backtest_metrics['total_return'])}")
                        st.write(f"- Annualized Return: {format_percentage(backtest_metrics['annualized_return'])}")
                        st.write(f"- Cumulative Return: {format_percentage(backtest_metrics['cumulative_return'])}")
                    
                    with col2:
                        st.write("**Risk Metrics:**")
                        st.write(f"- Volatility: {format_percentage(backtest_metrics['volatility'])}")
                        st.write(f"- Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.3f}")
                        st.write(f"- Max Drawdown: {format_percentage(backtest_metrics['max_drawdown'])}")
                    
                    # Trade analysis
                    if backtest_results["trades"] is not None and len(backtest_results["trades"]) > 0:
                        st.subheader("Trade Analysis")
                        trades_df = backtest_results["trades"]
                        st.dataframe(trades_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Backtesting failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p><strong>Portfolio Optimization with AI</strong> - Research & Educational Demo</p>
    <p>⚠️ This tool is for educational purposes only. NOT investment advice.</p>
</div>
""", unsafe_allow_html=True)
