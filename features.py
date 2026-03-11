"""
Feature engineering module following Algorithm 1.
Ref: Section 3.4.2 and Algorithm 1 of Uysal et al. (2021).

Input features per asset:
  - Past 5 daily returns:  r_{t-1}, ..., r_{t-5}     (5 features)
  - Past 10, 20, 30 day average returns               (3 features)
  - Past 10, 20, 30 day volatilities (std dev)         (3 features)
  Total per asset = 11, total for 7 assets = 77

"To comply with non-anticipativity constraint, the input feature does not
 include asset returns on the day of interest." — Section 3.4.2
"""
import numpy as np
import pandas as pd
from config import PAST_RETURNS_DAYS, AVG_WINDOWS, COV_WINDOW


def build_features(
    returns: pd.DataFrame,
    t: int,
    past_ret_days: int = PAST_RETURNS_DAYS,
    avg_windows: list[int] | None = None,
) -> np.ndarray:
    """
    Build feature vector for day t (0-indexed into returns DataFrame).

    Non-anticipativity: uses data up to but NOT including day t.

    Args:
        returns: DataFrame of daily returns, shape (T, n_assets)
        t: current day index
        past_ret_days: number of past daily returns (default 5)
        avg_windows: windows for avg return/volatility (default [10, 20, 30])

    Returns:
        np.ndarray of shape (n_assets * 11,) = (77,) for 7 assets
    """
    if avg_windows is None:
        avg_windows = AVG_WINDOWS

    n_assets = returns.shape[1]
    features = []

    # Past daily returns: r_{t-1}, r_{t-2}, ..., r_{t-5}
    for lag in range(1, past_ret_days + 1):
        idx = t - lag
        if idx >= 0:
            features.append(returns.iloc[idx].values)
        else:
            features.append(np.zeros(n_assets))

    # Average returns and volatilities for each window
    for w in avg_windows:
        start_idx = max(0, t - w)
        window_data = returns.iloc[start_idx:t]
        if len(window_data) > 0:
            features.append(window_data.mean().values)   # avg return
            features.append(window_data.std().values)     # volatility
        else:
            features.append(np.zeros(n_assets))
            features.append(np.zeros(n_assets))

    # Stack: (11, n_assets) → flatten to (11 * n_assets,)
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

    Uses data from [t-window, t) — non-anticipative.

    Args:
        returns: DataFrame of daily returns
        t: current day index (uses data up to t-1)
        window: look-back window for covariance estimation
        epsilon: regularization for PSD guarantee

    Returns:
        np.ndarray of shape (n_assets, n_assets), guaranteed PSD
    """
    start_idx = max(0, t - window)
    window_data = returns.iloc[start_idx:t].values

    n = returns.shape[1]

    if len(window_data) < 2:
        return np.eye(n) * epsilon

    cov = np.cov(window_data, rowvar=False)

    # Ensure PSD: Σ + εI
    cov += epsilon * np.eye(n)

    return cov


def get_n_features(n_assets: int = 7) -> int:
    """
    Compute the total number of features.
    = n_assets × (PAST_RETURNS_DAYS + 2 × len(AVG_WINDOWS))
    = 7 × (5 + 2×3) = 7 × 11 = 77
    """
    return n_assets * (PAST_RETURNS_DAYS + 2 * len(AVG_WINDOWS))


# ======================================================================
# Main (smoke test)
# ======================================================================

if __name__ == "__main__":
    from data_loader import generate_simulated_data
    from config import SIM_TOTAL_DAYS

    returns = generate_simulated_data(SIM_TOTAL_DAYS, seed=42)
    print(f"Returns shape: {returns.shape}")

    # Test feature building at day 50 (well within warmup)
    feat = build_features(returns, t=50)
    print(f"Feature vector shape: {feat.shape}")  # should be 77
    print(f"Expected n_features: {get_n_features()}")
    assert feat.shape[0] == get_n_features()

    # Test covariance estimation
    cov = estimate_covariance(returns, t=50)
    print(f"Covariance matrix shape: {cov.shape}")
    eigvals = np.linalg.eigvalsh(cov)
    print(f"Min eigenvalue: {eigvals.min():.2e} (should be > 0)")
    assert np.all(eigvals > 0), "Covariance not PSD!"

    # Test at boundary: day 30 (minimum for full features)
    feat_boundary = build_features(returns, t=30)
    print(f"Feature at t=30 shape: {feat_boundary.shape}")
    assert feat_boundary.shape[0] == get_n_features()

    # Test at early day: should still work (zero-padded)
    feat_early = build_features(returns, t=2)
    print(f"Feature at t=2 shape: {feat_early.shape}")
    assert feat_early.shape[0] == get_n_features()

    print("✅ All feature tests passed!")
