"""
Feature engineering module following Algorithm 1.
Ref: Section 3.4.2 and Algorithm 1 of Uysal et al. (2021).

Input features per asset:
  - Past 5 daily returns (5 features)
  - Past 10, 20, 30 day average returns (3 features)
  - Past 10, 20, 30 day volatilities (3 features)
  Total per asset = 11, total for 7 assets = 77
"""
import numpy as np
import pandas as pd
import torch
from config import PAST_RETURNS_DAYS, AVG_WINDOWS, COV_WINDOW, N_ASSETS


def build_features(
    returns: pd.DataFrame,
    t: int,
    past_ret_days: int = PAST_RETURNS_DAYS,
    avg_windows: list[int] = AVG_WINDOWS,
) -> np.ndarray:
    """
    Build feature vector for day t (0-indexed into returns DataFrame).

    "To comply with non-anticipativity constraint, the input feature does not
     include asset returns on the day of interest." — Section 3.4.2

    Args:
        returns: DataFrame of daily returns, shape (T, n_assets)
        t: current day index (we use data up to t-1)
        past_ret_days: number of past daily returns (default 5)
        avg_windows: list of windows for avg return/vol (default [10, 20, 30])

    Returns:
        np.ndarray of shape (n_features,) where n_features = n_assets * 11
    """
    n_assets = returns.shape[1]
    features = []

    # All data up to but NOT including day t
    # Past 5 daily returns: r_{t-1}, r_{t-2}, ..., r_{t-5}
    for lag in range(1, past_ret_days + 1):
        if t - lag >= 0:
            features.append(returns.iloc[t - lag].values)
        else:
            features.append(np.zeros(n_assets))

    # Average returns and volatilities for each window
    for w in avg_windows:
        start_idx = max(0, t - w)
        window_data = returns.iloc[start_idx:t]
        if len(window_data) > 0:
            features.append(window_data.mean().values)      # avg return
            features.append(window_data.std().values)        # volatility
        else:
            features.append(np.zeros(n_assets))
            features.append(np.zeros(n_assets))

    # Stack: shape = (11, n_assets) → flatten to (11 * n_assets,)
    return np.concatenate(features)


def estimate_covariance(
    returns: pd.DataFrame,
    t: int,
    window: int = COV_WINDOW,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Estimate sample covariance matrix using past `window` days.
    "We use a sample estimator of the covariance matrix ... obtained from
     the historical returns of the past 30 days." — Section 5.1

    Args:
        returns: DataFrame of daily returns
        t: current day index (use data up to t-1)
        window: look-back window for covariance estimation
        epsilon: regularization term for PSD guarantee

    Returns:
        np.ndarray of shape (n_assets, n_assets), guaranteed PSD
    """
    start_idx = max(0, t - window)
    window_data = returns.iloc[start_idx:t].values

    if len(window_data) < 2:
        n = returns.shape[1]
        return np.eye(n) * epsilon

    cov = np.cov(window_data, rowvar=False)

    # Ensure PSD by adding εI
    cov += epsilon * np.eye(cov.shape[0])

    return cov


def prepare_rolling_data(
    returns: pd.DataFrame,
    lookback: int,
    rebalance: int,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> list[dict]:
    """
    Prepare data batches for rolling-window training per Algorithm 1.

    For each rebalancing day t, prepare:
      - Training features and returns for days [t-K, ..., t-1]
      - Covariance matrices for each training day
      - Test day features and actual returns

    Args:
        returns: full returns DataFrame
        lookback: training window K (e.g., 150)
        rebalance: rebalance frequency (e.g., 25 for market, 5 for sim)
        start_idx: first rebalancing day (default = lookback)
        end_idx: last rebalancing day (default = len(returns))

    Returns:
        List of dicts, each containing training and test data for one batch.
    """
    T = len(returns)
    if start_idx is None:
        start_idx = lookback
    if end_idx is None:
        end_idx = T

    batches = []
    # Rebalancing days
    rebal_days = list(range(start_idx, end_idx, rebalance))

    for t in rebal_days:
        # Training period: [t-K, t-1]
        train_start = max(0, t - lookback)

        # Build features and collect returns for training period
        train_features = []
        train_returns_list = []
        train_covs = []

        for s in range(train_start, t):
            # Need enough history to build features
            min_history = max(PAST_RETURNS_DAYS, max(AVG_WINDOWS))
            if s < min_history:
                continue

            feat = build_features(returns, s)
            train_features.append(feat)
            # Realized return on day s (used for computing portfolio return)
            train_returns_list.append(returns.iloc[s].values)
            # Covariance estimate for day s
            cov_s = estimate_covariance(returns, s)
            train_covs.append(cov_s)

        if len(train_features) == 0:
            continue

        train_features = np.array(train_features)      # (K', n_features)
        train_returns_arr = np.array(train_returns_list)  # (K', n_assets)
        train_covs = np.array(train_covs)                # (K', n, n)

        # Test period: days [t, t+rebalance-1]
        test_end = min(t + rebalance, T)
        test_features = []
        test_returns_list = []
        test_covs = []

        for s in range(t, test_end):
            min_history = max(PAST_RETURNS_DAYS, max(AVG_WINDOWS))
            if s < min_history:
                continue
            feat = build_features(returns, s)
            test_features.append(feat)
            test_returns_list.append(returns.iloc[s].values)
            cov_s = estimate_covariance(returns, s)
            test_covs.append(cov_s)

        if len(test_features) == 0:
            continue

        test_features = np.array(test_features)
        test_returns_arr = np.array(test_returns_list)
        test_covs = np.array(test_covs)

        batches.append({
            "train_features": torch.tensor(train_features, dtype=torch.float32),
            "train_returns": torch.tensor(train_returns_arr, dtype=torch.float32),
            "train_covs": torch.tensor(train_covs, dtype=torch.float64),
            "test_features": torch.tensor(test_features, dtype=torch.float32),
            "test_returns": torch.tensor(test_returns_arr, dtype=torch.float32),
            "test_covs": torch.tensor(test_covs, dtype=torch.float64),
            "rebal_day": t,
        })

    return batches


if __name__ == "__main__":
    from data_loader import generate_simulated_data

    returns = generate_simulated_data(n_days=325, seed=42)
    print(f"Returns shape: {returns.shape}")

    # Test feature building
    feat = build_features(returns, t=50)
    print(f"Feature vector shape: {feat.shape}")  # should be 77

    # Test covariance estimation
    cov = estimate_covariance(returns, t=50)
    print(f"Covariance matrix shape: {cov.shape}")
    print(f"Is PSD: {np.all(np.linalg.eigvalsh(cov) > 0)}")

    # Test rolling data preparation
    batches = prepare_rolling_data(returns, lookback=150, rebalance=5)
    print(f"Number of batches: {len(batches)}")
    if batches:
        b = batches[0]
        print(f"Train features: {b['train_features'].shape}")
        print(f"Train returns: {b['train_returns'].shape}")
        print(f"Train covs: {b['train_covs'].shape}")
