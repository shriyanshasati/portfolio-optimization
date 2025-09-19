"""
Portfolio Optimization with Monte Carlo Simulation
--------------------------------------------------
This project:
  - Downloads historical stock data automatically using yfinance.
  - Cleans and prepares the data for analysis.
  - Runs Monte Carlo simulations to identify the optimal portfolio (max Sharpe ratio).
  - Plots portfolio performance and simulation results.

Dependencies:
    pip install numpy pandas matplotlib seaborn plotly yfinance
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def ensure_price_dataframe(df):
    """Ensure dataframe only has Date + stock price columns (all numeric)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(how='all', inplace=True)

    for c in df.columns:
        if c != 'Date':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    cols = ['Date'] + [c for c in df.columns if c != 'Date']
    return df[cols].reset_index(drop=True)

# ---------------------------------------------------------
# Data Loader (yfinance)
# ---------------------------------------------------------

def load_or_download_prices(csv_path='stock_prices.csv', tickers=None):
    """Load data from CSV or download Adjusted Close from yfinance."""
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df

    if tickers is None:
        raise FileNotFoundError(f"{csv_path} not found and no tickers provided.")

    print("Downloading stock data using yfinance...")
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime.now()

    df = yf.download(tickers, start=start, end=end, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    else:
        df = df[['Adj Close']]

    df.reset_index(inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"Data downloaded and saved to {csv_path}")
    return df

# ---------------------------------------------------------
# Core Functions
# ---------------------------------------------------------

def price_scaling(raw_prices_df):
    """Scale all stock prices to start at 1 for comparison."""
    df = ensure_price_dataframe(raw_prices_df)
    scaled = pd.DataFrame({'Date': df['Date']})
    for c in df.columns[1:]:
        first_val = df[c].iloc[0]
        scaled[c] = df[c] / first_val if first_val != 0 else np.nan
    return scaled

def generate_portfolio_weights(n):
    """Generate random portfolio weights that sum to 1."""
    return np.random.default_rng().dirichlet(np.ones(n))

def asset_allocation(close_price_df, weights, initial_investment):
    """Calculate value of each stock position and portfolio over time."""
    df = ensure_price_dataframe(close_price_df)
    stocks = df.columns[1:]
    if len(stocks) != len(weights):
        raise ValueError("Number of weights does not match number of stocks")

    scaled = price_scaling(df)
    pos_df = pd.DataFrame({'Date': scaled['Date']})
    for i, s in enumerate(stocks):
        pos_df[s] = scaled[s] * weights[i] * initial_investment

    pos_df['Portfolio Value [$]'] = pos_df[stocks].sum(axis=1)
    pos_df['Portfolio Daily Return'] = pos_df['Portfolio Value [$]'].pct_change().fillna(0)
    return pos_df

def simulation_engine(weights, initial_investment, close_price_df, risk_free_rate=0.03):
    """Calculate expected return, volatility, Sharpe ratio, and final value."""
    df = ensure_price_dataframe(close_price_df)
    stocks = df.columns[1:]

    daily_returns = df[stocks].pct_change().dropna(how='all')
    if daily_returns.empty:
        raise ValueError("Not enough rows to compute returns.")

    mean_daily_returns = daily_returns.mean()
    covariance = daily_returns.cov()

    trading_days = 252
    expected_portfolio_return = float(np.dot(weights, mean_daily_returns) * trading_days)
    cov_annual = covariance * trading_days
    expected_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov_annual.values, weights))))

    scaled = price_scaling(df)
    final_positions = scaled.iloc[-1][stocks].values * weights * initial_investment
    final_value = float(np.nansum(final_positions))
    roi_pct = float((final_value - initial_investment) / initial_investment * 100.0)

    sharpe_ratio = (expected_portfolio_return - risk_free_rate) / expected_volatility if expected_volatility != 0 else np.nan

    return expected_portfolio_return, expected_volatility, sharpe_ratio, final_value, roi_pct

# ---------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------

def monte_carlo_simulation(close_price_df, sim_runs=5000, initial_investment=1_000_000, risk_free_rate=0.03, show_progress=False):
    """Run Monte Carlo simulation to find best portfolio (max Sharpe ratio)."""
    df = ensure_price_dataframe(close_price_df)
    stocks = df.columns[1:]
    n = len(stocks)

    weights_runs = np.zeros((sim_runs, n))
    sharpe_runs = np.zeros(sim_runs)
    returns_runs = np.zeros(sim_runs)
    vol_runs = np.zeros(sim_runs)
    final_values = np.zeros(sim_runs)
    roi_runs = np.zeros(sim_runs)

    for i in range(sim_runs):
        weights = generate_portfolio_weights(n)
        weights_runs[i, :] = weights
        exp_ret, vol, sharpe, final_val, roi = simulation_engine(weights, initial_investment, df, risk_free_rate)
        sharpe_runs[i], returns_runs[i], vol_runs[i], final_values[i], roi_runs[i] = sharpe, exp_ret, vol, final_val, roi
        if show_progress and ((i + 1) % max(1, sim_runs // 10) == 0):
            print(f"Completed {i + 1}/{sim_runs} simulations...")

    sim_out_df = pd.DataFrame({
        'Volatility': vol_runs,
        'Portfolio_Return': returns_runs,
        'Sharpe_Ratio': sharpe_runs,
        'Final_Value': final_values,
        'ROI_pct': roi_runs
    })

    best_idx = int(np.nanargmax(sharpe_runs))
    best = {
        'index': best_idx,
        'weights': weights_runs[best_idx, :],
        'Sharpe_Ratio': sharpe_runs[best_idx],
        'Portfolio_Return': returns_runs[best_idx],
        'Volatility': vol_runs[best_idx],
        'Final_Value': final_values[best_idx],
        'ROI_pct': roi_runs[best_idx],
        'stocks': list(stocks)
    }

    return {'sim_out_df': sim_out_df, 'weights_runs': weights_runs, 'best': best}

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def plot_simulation_results(sim_out_df, best):
    fig = px.scatter(
        sim_out_df, x='Volatility', y='Portfolio_Return', color='Sharpe_Ratio',
        title='Monte Carlo Portfolio Simulation (Volatility vs Return)',
        labels={'Portfolio_Return': 'Expected Annual Return', 'Volatility': 'Annual Volatility'}
    )
    fig.add_trace(go.Scatter(
        x=[best['Volatility']], y=[best['Portfolio_Return']],
        mode='markers+text', marker=dict(size=12, color='black'),
        text=['Best (max Sharpe)'], textposition='bottom center', name='Optimal'
    ))
    fig.show()

def plot_portfolio_value_over_time(portfolio_df):
    fig = px.line(portfolio_df, x='Date', y='Portfolio Value [$]', title='Portfolio Value Over Time')
    fig.show()

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------

if __name__ == "__main__":
    tickers = ['AMZN', 'JPM', 'KO', 'PEP', 'SPY', 'TSLA']
    close_price_df = load_or_download_prices('stock_prices.csv', tickers=tickers)
    close_price_df = ensure_price_dataframe(close_price_df)

    print("\nData preview:")
    print(close_price_df.head())   # only first 5 rows

    initial_investment = 1_000_000
    sim_runs = 2000
    print(f"\nRunning Monte Carlo simulation with {sim_runs} runs...")

    sim_res = monte_carlo_simulation(
        close_price_df, sim_runs=sim_runs,
        initial_investment=initial_investment,
        risk_free_rate=0.03,
        show_progress=True
    )

    best = sim_res['best']

    # Display concise results
    print("\nBest Portfolio (Max Sharpe Ratio):")
    for stock, w in zip(best['stocks'], best['weights']):
        print(f"{stock}: {w:.4f}")
    print(f"Sharpe Ratio = {best['Sharpe_Ratio']:.4f}")
    print(f"Expected Annual Return = {best['Portfolio_Return']:.4f}")
    print(f"Volatility = {best['Volatility']:.4f}")
    print(f"Final Value = ${best['Final_Value']:,.2f}")
    print(f"ROI = {best['ROI_pct']:.2f}%")

    # Plot
    plot_simulation_results(sim_res['sim_out_df'], best)
    best_portfolio_df = asset_allocation(close_price_df, best['weights'], initial_investment)
    plot_portfolio_value_over_time(best_portfolio_df)
