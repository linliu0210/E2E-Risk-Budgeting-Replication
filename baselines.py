"""
Baseline strategies: Nominal Risk Parity, Equal Weight (1/N).
Ref: Section 5.2, 6.1 of Uysal et al. (2021).
"""
import numpy as np
import torch
from risk_budget_layer import build_risk_budget_layer, solve_risk_parity
from features import estimate_covariance
from config import N_ASSETS, COV_WINDOW


def equal_weight_allocation(n_assets: int = N_ASSETS) -> np.ndarray:
    """
    Equal weight (1/N) fix-mix strategy.
    "Fix-mix with equal weights (1/N) is showed by many researchers
     to be a robust and relatively strong benchmark." — Section 5.2
    """
    return np.ones(n_assets) / n_assets


def nominal_risk_parity(
    returns_df,
    start_idx: int,
    end_idx: int,
    cov_window: int = COV_WINDOW,
    rebalance: int = 25,
    n_assets: int = N_ASSETS,
) -> np.ndarray:
    """
    Run nominal risk parity strategy on historical data.
    Equal risk budgets b = [1/n, ..., 1/n] with sample covariance.

    Args:
        returns_df: DataFrame of daily returns
        start_idx: first day to compute allocation
        end_idx: last day
        cov_window: look-back for covariance estimation
        rebalance: rebalance frequency in days

    Returns:
        portfolio_daily_returns: array of daily portfolio returns
    """
    layer = build_risk_budget_layer(n_assets=n_assets)
    portfolio_returns = []

    current_weights = np.ones(n_assets) / n_assets  # initial equal weight

    for t in range(start_idx, end_idx):
        # Rebalance at specified frequency
        if (t - start_idx) % rebalance == 0:
            cov = estimate_covariance(returns_df, t, window=cov_window)
            Sigma_t = torch.tensor(cov, dtype=torch.float64)

            try:
                with torch.no_grad():
                    z = solve_risk_parity(Sigma_t, layer, n_assets=n_assets)
                current_weights = z.numpy()
            except Exception as e:
                print(f"[baselines] RP solver failed at t={t}: {e}, keeping previous weights")

        # Realized return
        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(current_weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


def equal_weight_portfolio(
    returns_df,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """
    Run equal weight (1/N) strategy — buy and hold with equal allocation.

    Returns:
        portfolio_daily_returns: array of daily portfolio returns
    """
    n_assets = returns_df.shape[1]
    weights = np.ones(n_assets) / n_assets
    portfolio_returns = []

    for t in range(start_idx, end_idx):
        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


def nominal_rp_positive(
    returns_df,
    start_idx: int,
    end_idx: int,
    cov_window: int = COV_WINDOW,
    rebalance: int = 25,
) -> np.ndarray:
    """
    Nominal RP invested only in assets with positive returns in past 30 days.
    Ref: "Nominal RP-positive" in Section 6.1.
    """
    n_assets = returns_df.shape[1]
    portfolio_returns = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_idx, end_idx):
        if (t - start_idx) % rebalance == 0:
            # Check which assets had positive avg returns in past 30 days
            lookback_start = max(0, t - 30)
            avg_rets = returns_df.iloc[lookback_start:t].mean().values
            positive_mask = avg_rets > 0

            if positive_mask.sum() == 0:
                # If no asset is positive, equal weight all
                current_weights = np.ones(n_assets) / n_assets
            else:
                # Risk parity on positive assets only
                n_pos = positive_mask.sum()
                cov = estimate_covariance(returns_df, t, window=cov_window)

                # Subset covariance
                pos_indices = np.where(positive_mask)[0]
                cov_sub = cov[np.ix_(pos_indices, pos_indices)]

                try:
                    layer_sub = build_risk_budget_layer(n_assets=n_pos)
                    Sigma_sub = torch.tensor(cov_sub, dtype=torch.float64)
                    with torch.no_grad():
                        z_sub = solve_risk_parity(Sigma_sub, layer_sub, n_assets=n_pos)
                    w_sub = z_sub.numpy()
                except Exception:
                    w_sub = np.ones(n_pos) / n_pos

                current_weights = np.zeros(n_assets)
                current_weights[pos_indices] = w_sub

        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(current_weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


if __name__ == "__main__":
    from data_loader import generate_simulated_data

    returns = generate_simulated_data(n_days=325, seed=42)
    print(f"Returns shape: {returns.shape}")

    # Test equal weight
    ew_rets = equal_weight_portfolio(returns, start_idx=150, end_idx=325)
    print(f"Equal weight returns: {len(ew_rets)} days, avg={ew_rets.mean():.6f}")

    # Test nominal risk parity
    rp_rets = nominal_risk_parity(returns, start_idx=150, end_idx=175, rebalance=5)
    print(f"Risk parity returns: {len(rp_rets)} days, avg={rp_rets.mean():.6f}")
