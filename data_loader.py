"""
Data acquisition: real ETF data via yfinance + simulated data generation.
Ref: Section 4.1, 5 of Uysal et al. (2021).

Key design decisions:
  - μ is HARDCODED from paper Section 4.1 (not computed dynamically)
  - Σ is computed from real 2011-2021 ETF data
  - Section 5.4.2 augmented universe adds a synthetic low-vol asset
"""
import numpy as np
import pandas as pd
import os

from config import (
    ETF_TICKERS, DATA_START, DATA_END,
    SIM_MU_PAPER, RANDOM_ASSET_MU, RANDOM_ASSET_SIGMA,
)


# ======================================================================
# Real ETF Data
# ======================================================================

def download_etf_data(
    tickers: list[str] | None = None,
    start: str = DATA_START,
    end: str = DATA_END,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for ETFs, compute daily returns.

    Returns:
        pd.DataFrame: columns=tickers, index=dates, values=daily simple returns.
    """
    if tickers is None:
        tickers = ETF_TICKERS

    if cache_path is not None:
        try:
            returns = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"[data_loader] Loaded cached data from {cache_path}")
            return returns
        except FileNotFoundError:
            pass

    import yfinance as yf
    print(f"[data_loader] Downloading {tickers} from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Daily simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change().dropna()
    returns = returns[tickers]  # ensure column order

    if cache_path is not None:
        returns.to_csv(cache_path)
        print(f"[data_loader] Cached data to {cache_path}")

    return returns


# ======================================================================
# Distribution Parameters for Simulation
# ======================================================================

def compute_distribution_params(
    tickers: list[str] | None = None,
    start: str = "2011-01-01",
    end: str = "2021-06-30",
    cache_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute μ and Σ for simulated data generation (Section 4.1).

    μ is HARDCODED from the paper: [0.059, 0.013, -0.011, 0.022, 0.056, 0.017, 0.017] %
    Σ is computed from real ETF daily returns (2011-2021).

    Returns:
        mu: (n_assets,) mean daily returns
        cov: (n_assets, n_assets) covariance matrix
    """
    if tickers is None:
        tickers = ETF_TICKERS

    # μ: hardcoded from paper Section 4.1 (percentage → decimal)
    mu = np.array(SIM_MU_PAPER) / 100.0

    # Σ: from real ETF data
    returns = download_etf_data(tickers, start, end, cache_path=cache_path)
    cov = returns.cov().values

    # Ensure PSD
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < 0):
        cov += (-eigvals.min() + 1e-10) * np.eye(len(mu))

    return mu, cov


# ======================================================================
# Simulated Data Generation
# ======================================================================

def generate_simulated_data(
    n_days: int,
    seed: int = 42,
    mu: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate simulated multi-asset returns from a multivariate normal.

    Section 4.1: "The distribution parameters are determined by the mean
    and covariance matrix of daily returns of seven ETFs from 2011 to 2021"

    Args:
        n_days: total number of days to generate (warmup + train + test)
        seed: random seed for reproducibility
        mu: mean vector (if None, computed from real data)
        cov: covariance matrix (if None, computed from real data)

    Returns:
        pd.DataFrame with shape (n_days, 7)
    """
    if mu is None or cov is None:
        mu, cov = compute_distribution_params(cache_path=cache_path)

    rng = np.random.RandomState(seed)
    returns_np = rng.multivariate_normal(mu, cov, size=n_days)

    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    returns = pd.DataFrame(returns_np, index=dates, columns=ETF_TICKERS)

    return returns


# ======================================================================
# Augmented Universe (Section 5.4.2)
# ======================================================================

def generate_augmented_universe(
    returns_7etf: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Section 5.4.2: Add a synthetic low-volatility, low-return random asset
    to the 7-ETF universe for testing Stochastic Gates filtering.

    "a random asset with a mean of −0.05% and standard deviation of 0.05%"

    Args:
        returns_7etf: original 7-ETF returns DataFrame
        seed: random seed

    Returns:
        pd.DataFrame with 8 columns (7 ETF + 1 RANDOM)
    """
    rng = np.random.RandomState(seed)
    n_days = len(returns_7etf)

    random_returns = rng.normal(RANDOM_ASSET_MU, RANDOM_ASSET_SIGMA, n_days)
    random_series = pd.Series(
        random_returns, index=returns_7etf.index, name="RANDOM"
    )

    returns_8 = pd.concat([returns_7etf, random_series], axis=1)
    return returns_8


# ======================================================================
# Main (smoke test)
# ======================================================================

if __name__ == "__main__":
    # 1. Test simulated data
    print("=== Simulated Data ===")
    cache = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    mu, cov = compute_distribution_params(cache_path=cache)
    print(f"μ (hardcoded, ×10000): {(mu * 10000).round(1)}")
    print(f"Σ shape: {cov.shape}")
    print(f"Σ is PSD: {np.all(np.linalg.eigvalsh(cov) > 0)}")

    from config import SIM_TOTAL_DAYS
    sim_data = generate_simulated_data(SIM_TOTAL_DAYS, seed=0, mu=mu, cov=cov)
    print(f"Simulated returns shape: {sim_data.shape}")  # (355, 7)
    print(f"Mean daily returns (%):\n{sim_data.mean() * 100}")
    print()

    # 2. Test augmented universe
    print("=== Augmented Universe ===")
    aug_data = generate_augmented_universe(sim_data, seed=0)
    print(f"Augmented shape: {aug_data.shape}")  # (355, 8)
    print(f"RANDOM asset mean: {aug_data['RANDOM'].mean():.6f}")
    print(f"RANDOM asset std:  {aug_data['RANDOM'].std():.6f}")
    print()

    # 3. Test real data download
    try:
        real_data = download_etf_data(cache_path=cache)
        print("=== Real ETF Data ===")
        print(f"Shape: {real_data.shape}")
        print(f"Date range: {real_data.index[0]} to {real_data.index[-1]}")
    except Exception as e:
        print(f"Could not load real data: {e}")
