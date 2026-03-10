"""
Data acquisition module: real ETF data via yfinance + simulated data.
Ref: Section 4.1, 5 of Uysal et al. (2021).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from config import (
    ETF_TICKERS, DATA_START, DATA_END,
    IN_SAMPLE_START, OUT_SAMPLE_END
)


def download_etf_data(
    tickers: list[str] = ETF_TICKERS,
    start: str = DATA_START,
    end: str = DATA_END,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for ETFs, compute daily returns.

    Returns:
        pd.DataFrame with columns = tickers, index = dates, values = daily returns.
    """
    if cache_path is not None:
        try:
            returns = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"[data_loader] Loaded cached data from {cache_path}")
            return returns
        except FileNotFoundError:
            pass

    print(f"[data_loader] Downloading {tickers} from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # yfinance returns MultiIndex columns; extract 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Compute daily simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change().dropna()

    # Ensure column order matches tickers
    returns = returns[tickers]

    if cache_path is not None:
        returns.to_csv(cache_path)
        print(f"[data_loader] Cached data to {cache_path}")

    return returns


def generate_simulated_data(
    n_days: int = 175 + 150,  # need lookback + test days
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate simulated multi-asset returns from a multivariate normal.
    Parameters calibrated from 7 ETFs (2011-2021) as described in Section 4.1.

    Ref: "The distribution parameters are determined by the mean and covariance
         matrix of daily returns of seven ETFs from 2011 to 2021"

    Expected daily returns (from paper Section 4.1):
        [0.059%, 0.013%, -0.011%, 0.022%, 0.056%, 0.017%, 0.017%]
    """
    np.random.seed(seed)

    # Mean daily returns from Section 4.1 (in decimal form)
    mu = np.array([0.00059, 0.00013, -0.00011, 0.00022, 0.00056, 0.00017, 0.00017])

    # Approximate covariance matrix (from typical ETF correlations)
    # Using representative values; exact values not in paper
    daily_vols = np.array([0.0111, 0.0138, 0.0025, 0.0046, 0.0032, 0.0102, 0.0100])

    # Correlation matrix (approximate)
    corr = np.array([
        [1.00, 0.90, 0.05, 0.20, 0.10, 0.30, 0.05],  # VTI
        [0.90, 1.00, 0.03, 0.18, 0.08, 0.28, 0.03],  # IWM
        [0.05, 0.03, 1.00, 0.70, 0.65, -0.05, 0.15], # AGG
        [0.20, 0.18, 0.70, 1.00, 0.55, 0.05, 0.10],  # LQD
        [0.10, 0.08, 0.65, 0.55, 1.00, -0.05, 0.10], # MUB
        [0.30, 0.28, -0.05, 0.05, -0.05, 1.00, 0.30],# DBC
        [0.05, 0.03, 0.15, 0.10, 0.10, 0.30, 1.00],  # GLD
    ])

    # Σ = diag(σ) @ corr @ diag(σ)
    D = np.diag(daily_vols)
    cov = D @ corr @ D

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < 0):
        cov += (-eigvals.min() + 1e-8) * np.eye(len(mu))

    # Generate returns
    returns_np = np.random.multivariate_normal(mu, cov, size=n_days)

    # Create DataFrame with fake dates
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    returns = pd.DataFrame(returns_np, index=dates, columns=ETF_TICKERS)

    return returns


if __name__ == "__main__":
    # Quick test with simulated data
    sim_returns = generate_simulated_data(n_days=325, seed=42)
    print("=== Simulated Data ===")
    print(f"Shape: {sim_returns.shape}")
    print(f"Mean daily returns (%):\n{sim_returns.mean() * 100}")
    print(f"Annualized vol:\n{sim_returns.std() * np.sqrt(252)}")
    print()

    # Download real data (will be slow first time)
    try:
        real_returns = download_etf_data(cache_path="etf_returns.csv")
        print("=== Real ETF Data ===")
        print(f"Shape: {real_returns.shape}")
        print(f"Date range: {real_returns.index[0]} to {real_returns.index[-1]}")
        print(f"Mean daily returns (%):\n{real_returns.mean() * 100}")
    except Exception as e:
        print(f"Could not download real data: {e}")
