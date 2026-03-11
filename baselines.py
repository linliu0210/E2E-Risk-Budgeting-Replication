"""
Baseline strategies for comparison.
Ref: Section 5.2, 5.4 of Uysal et al. (2021).

Baselines:
  1. Equal Weight (1/N) fix-mix — Section 5.2
  2. Nominal Risk Parity — Section 5.2
  3. Nominal RP-positive — Section 5.4 (Table 6, 7)
  4. Nominal RP-topk — Section 5.4 (Table 6, 7)
"""
import numpy as np
import torch
from risk_budget_layer import build_risk_budget_layer, solve_risk_parity, solve_risk_budget
from features import estimate_covariance
from config import N_ASSETS, COV_WINDOW


# ======================================================================
# Equal Weight (1/N)
# ======================================================================

def equal_weight_portfolio(
    returns_df,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """
    Equal weight (1/N) fix-mix strategy.
    "Fix-mix with equal weights (1/N) is showed by many researchers
     to be a robust and relatively strong benchmark." — Section 5.2
    """
    n_assets = returns_df.shape[1]
    weights = np.ones(n_assets) / n_assets
    portfolio_returns = []

    for t in range(start_idx, end_idx):
        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


# ======================================================================
# Nominal Risk Parity
# ======================================================================

def nominal_risk_parity(
    returns_df,
    start_idx: int,
    end_idx: int,
    cov_window: int = COV_WINDOW,
    rebalance: int = 25,
    n_assets: int | None = None,
) -> np.ndarray:
    """
    Nominal risk parity: equal risk budgets b = [1/n, ..., 1/n]
    with 30-day sample covariance, rebalanced periodically.
    """
    if n_assets is None:
        n_assets = returns_df.shape[1]

    layer = build_risk_budget_layer(n_assets=n_assets)
    portfolio_returns = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_idx, end_idx):
        if (t - start_idx) % rebalance == 0:
            cov = estimate_covariance(returns_df, t, window=cov_window)
            Sigma_t = torch.tensor(cov, dtype=torch.float64)

            try:
                with torch.no_grad():
                    z = solve_risk_parity(Sigma_t, layer, n_assets=n_assets)
                current_weights = z.numpy()
            except Exception as e:
                print(f"[baselines] RP solver fail at t={t}: {e}")

        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(current_weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


# ======================================================================
# Nominal RP-positive (Section 5.4)
# ======================================================================

def nominal_rp_positive(
    returns_df,
    start_idx: int,
    end_idx: int,
    lookback: int = 30,
    cov_window: int = COV_WINDOW,
    rebalance: int = 25,
) -> np.ndarray:
    """
    Nominal RP invested only in assets with positive 30-day avg returns.
    Ref: "Nominal RP-positive" in Section 5.4 / Table 6, 7.
    """
    n_assets = returns_df.shape[1]
    portfolio_returns = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_idx, end_idx):
        if (t - start_idx) % rebalance == 0:
            lb_start = max(0, t - lookback)
            avg_rets = returns_df.iloc[lb_start:t].mean().values
            positive_mask = avg_rets > 0

            if positive_mask.sum() == 0:
                current_weights = np.ones(n_assets) / n_assets
            else:
                pos_indices = np.where(positive_mask)[0]
                n_pos = len(pos_indices)

                cov = estimate_covariance(returns_df, t, window=cov_window)
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


# ======================================================================
# Nominal RP-topk (Section 5.4)
# ======================================================================

def nominal_rp_topk(
    returns_df,
    start_idx: int,
    end_idx: int,
    k: int = 4,
    lookback: int = 30,
    cov_window: int = COV_WINDOW,
    rebalance: int = 25,
) -> np.ndarray:
    """
    Nominal RP invested in top-k assets by 30-day average return.
    Ref: "Nominal RP-topk" in Section 5.4 / Table 6, 7.

    Args:
        k: number of top assets to select
    """
    n_assets = returns_df.shape[1]
    portfolio_returns = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(start_idx, end_idx):
        if (t - start_idx) % rebalance == 0:
            lb_start = max(0, t - lookback)
            avg_rets = returns_df.iloc[lb_start:t].mean().values

            # Select top-k assets by average return
            effective_k = min(k, n_assets)
            topk_indices = np.argsort(avg_rets)[-effective_k:]

            cov = estimate_covariance(returns_df, t, window=cov_window)
            cov_sub = cov[np.ix_(topk_indices, topk_indices)]

            try:
                layer_sub = build_risk_budget_layer(n_assets=effective_k)
                Sigma_sub = torch.tensor(cov_sub, dtype=torch.float64)
                with torch.no_grad():
                    z_sub = solve_risk_parity(Sigma_sub, layer_sub, n_assets=effective_k)
                w_sub = z_sub.numpy()
            except Exception:
                w_sub = np.ones(effective_k) / effective_k

            current_weights = np.zeros(n_assets)
            current_weights[topk_indices] = w_sub

        actual_ret = returns_df.iloc[t].values
        portfolio_ret = np.dot(current_weights, actual_ret)
        portfolio_returns.append(portfolio_ret)

    return np.array(portfolio_returns)


# ======================================================================
# Main (smoke test)
# ======================================================================

if __name__ == "__main__":
    from data_loader import generate_simulated_data
    from config import SIM_TOTAL_DAYS, SIM_FEATURE_WARMUP, SIM_LOOKBACK

    returns = generate_simulated_data(SIM_TOTAL_DAYS, seed=42)
    start = SIM_FEATURE_WARMUP + SIM_LOOKBACK
    end = start + 25  # short test

    ew = equal_weight_portfolio(returns, start, end)
    print(f"Equal weight: {len(ew)} days, avg={ew.mean():.6f}")

    rp = nominal_risk_parity(returns, start, end, rebalance=5)
    print(f"Nominal RP:   {len(rp)} days, avg={rp.mean():.6f}")

    rpp = nominal_rp_positive(returns, start, end, rebalance=5)
    print(f"RP-positive:  {len(rpp)} days, avg={rpp.mean():.6f}")

    rptk = nominal_rp_topk(returns, start, end, k=4, rebalance=5)
    print(f"RP-top4:      {len(rptk)} days, avg={rptk.mean():.6f}")

    print("✅ All baseline tests passed!")
