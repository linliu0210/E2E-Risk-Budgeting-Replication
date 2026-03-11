"""
Algorithm 1: Rolling-window E2E Training Pipeline.
Ref: Algorithm 1 (p.12) and Sections 4-5 of Uysal et al. (2021).

This is an implementation consistent with Algorithm 1's description and the
experiment settings in Sections 4/5. The outer loop iterates by rebalance
frequency; the inner loop trains for n_steps gradient steps.

Key design choices:
  - SGD optimizer (consistent with paper's θ ← θ − lr·∇R)
  - StepLR decay (×0.9 every 3 steps, per Section 5.1)
  - NO gradient clipping (not mentioned in paper)
  - Fresh θ initialization per rebalance batch (Algorithm 1 Line 2)
  - StochasticGateNet: separate parameter group for gate_mu learning rate
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import ModelFreeNet, ModelBasedNet, StochasticGateNet
from features import build_features, estimate_covariance, get_n_features
from losses import get_loss_fn
from config import (
    N_ASSETS, FEATURE_WARMUP,
    LR_DECAY_FACTOR, LR_DECAY_EVERY,
)


def create_model(
    model_type: str,
    n_features: int,
    n_assets: int = N_ASSETS,
    with_filter: bool = True,
) -> nn.Module:
    """Create model by type string."""
    if model_type == "model_free":
        return ModelFreeNet(n_features, n_assets=n_assets)
    elif model_type == "model_based":
        return ModelBasedNet(n_features, n_assets=n_assets)
    elif model_type == "gate_filter":
        return StochasticGateNet(n_features, n_assets=n_assets, with_filter=True)
    elif model_type == "gate_no_filter":
        return StochasticGateNet(n_features, n_assets=n_assets, with_filter=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_one_batch(
    model: nn.Module,
    returns_df: pd.DataFrame,
    train_days: list[int],
    lr: float,
    n_steps: int,
    loss_type: str,
    gate_lr: float | None = None,
    verbose: bool = False,
) -> float:
    """
    Inner loop of Algorithm 1: train model on one rolling window.

    For each gradient step:
      1. Forward pass through all train_days: features → allocation → return
      2. Compute loss (Sharpe or cumulative return) over all returns
      3. Backward + SGD update

    Args:
        model: neural network model
        returns_df: full returns DataFrame
        train_days: list of day indices in the training window
        lr: learning rate η
        n_steps: number of training steps
        loss_type: "sharpe" or "cumret"
        gate_lr: independent learning rate for gate μ (StochasticGateNet only)
        verbose: print training progress

    Returns:
        final loss value
    """
    model.train()
    loss_fn = get_loss_fn(loss_type)
    n_assets = returns_df.shape[1]

    # Build optimizer with separate parameter groups for gate μ
    if isinstance(model, StochasticGateNet) and gate_lr is not None:
        nn_params = [p for name, p in model.named_parameters()
                     if "gate_mu" not in name]
        optimizer = torch.optim.SGD([
            {"params": nn_params, "lr": lr},
            {"params": [model.gate_mu], "lr": gate_lr},
        ])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_EVERY, gamma=LR_DECAY_FACTOR
    )

    final_loss = 0.0

    for step in range(n_steps):
        optimizer.zero_grad()
        portfolio_returns = []

        for s in train_days:
            # 1. Build features (non-anticipative: uses data up to s-1)
            feat = build_features(returns_df, s)
            x_s = torch.tensor(feat, dtype=torch.float32)

            # 2. Estimate covariance (past 30 days)
            cov_s = estimate_covariance(returns_df, s)
            Sigma_s = torch.tensor(cov_s, dtype=torch.float64)

            # 3. Forward pass
            try:
                z_s, b_s = model(x_s, Sigma_s)
            except Exception as e:
                if verbose:
                    print(f"    [train] Solver fail at s={s}, step={step}: {e}")
                # Fallback: equal weight produces no gradient, but keeps running
                z_s = torch.ones(n_assets, dtype=torch.float32) / n_assets
                z_s = z_s.detach()

            # 4. Realized return
            actual_ret = torch.tensor(
                returns_df.iloc[s].values, dtype=torch.float32
            )
            port_ret = torch.dot(z_s.squeeze(), actual_ret)
            portfolio_returns.append(port_ret)

        # 5. Compute loss
        loss = loss_fn(portfolio_returns)

        # 6. Backward + update (NO gradient clipping)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss = loss.item()

        if verbose and (step + 1) % max(1, n_steps // 5) == 0:
            print(f"    Step {step+1}/{n_steps}: loss={final_loss:.6f}")

    return final_loss


def run_backtest(
    returns_df: pd.DataFrame,
    model_type: str = "model_based",
    loss_type: str = "sharpe",
    lr: float = 150.0,
    n_steps: int = 10,
    lookback: int = 150,
    rebalance: int = 25,
    start_idx: int | None = None,
    end_idx: int | None = None,
    seed: int = 42,
    gate_lr: float | None = None,
    verbose: bool = False,
) -> dict:
    """
    Outer loop of Algorithm 1: rolling-window backtest.

    For each rebalance day t:
      1. Initialize fresh model θ
      2. Train on [t-K, t-1]
      3. Apply trained model on [t, t+rebalance-1]
      4. Record portfolio returns

    Args:
        returns_df: full returns DataFrame
        model_type: "model_free", "model_based", "gate_filter", "gate_no_filter"
        loss_type: "sharpe" or "cumret"
        lr: learning rate η
        n_steps: training steps per batch
        lookback: training window K
        rebalance: rebalance frequency
        start_idx: first rebalance day (default: warmup + lookback)
        end_idx: last day (default: len(returns_df))
        seed: random seed for model initialization
        gate_lr: gate μ learning rate (for StochasticGateNet)
        verbose: print progress

    Returns:
        dict with portfolio_returns, dates, weights history
    """
    T = len(returns_df)
    n_assets = returns_df.shape[1]
    n_features = get_n_features(n_assets)

    # Determine valid range
    min_history = FEATURE_WARMUP  # need 30 days for features
    if start_idx is None:
        start_idx = max(lookback + min_history, min_history + lookback)
    if end_idx is None:
        end_idx = T

    # Generate rebalancing schedule
    rebal_days = list(range(start_idx, end_idx, rebalance))

    # Collect results
    all_returns = []
    all_dates = []
    all_weights = []

    for batch_idx, t in enumerate(rebal_days):
        if verbose:
            print(f"\n  Batch {batch_idx+1}/{len(rebal_days)} — "
                  f"rebal_day={t}, train=[{max(min_history, t-lookback)}, {t-1}]")

        # 1. Fresh model initialization (Algorithm 1 Line 2)
        torch.manual_seed(seed)
        model = create_model(model_type, n_features, n_assets=n_assets)

        # 2. Training window: [max(min_history, t-K), t-1]
        train_start = max(min_history, t - lookback)
        train_days = list(range(train_start, t))

        if len(train_days) == 0:
            continue

        # 3. Train
        final_loss = train_one_batch(
            model, returns_df, train_days,
            lr=lr, n_steps=n_steps, loss_type=loss_type,
            gate_lr=gate_lr, verbose=verbose,
        )

        if verbose:
            print(f"  Training done. Final loss: {final_loss:.6f}")

        # 4. Test period: [t, min(t+rebalance, end_idx))
        model.eval()
        test_end = min(t + rebalance, end_idx)

        with torch.no_grad():
            for s in range(t, test_end):
                if s >= T:
                    break

                feat = build_features(returns_df, s)
                x_s = torch.tensor(feat, dtype=torch.float32)
                cov_s = estimate_covariance(returns_df, s)
                Sigma_s = torch.tensor(cov_s, dtype=torch.float64)

                try:
                    z_s, b_s = model(x_s, Sigma_s)
                    weights_s = z_s.squeeze().numpy()
                except Exception as e:
                    if verbose:
                        print(f"    [test] Solver fail at s={s}: {e}")
                    weights_s = np.ones(n_assets) / n_assets

                # Clamp weights to valid range
                weights_s = np.maximum(weights_s, 0)
                w_sum = weights_s.sum()
                if w_sum > 1e-6:
                    weights_s = weights_s / w_sum
                else:
                    weights_s = np.ones(n_assets) / n_assets

                actual_ret = returns_df.iloc[s].values
                port_ret = np.dot(weights_s, actual_ret)

                all_returns.append(port_ret)
                all_dates.append(returns_df.index[s])
                all_weights.append(weights_s.copy())

    result = {
        "portfolio_returns": np.array(all_returns),
        "dates": all_dates,
        "weights": np.array(all_weights) if all_weights else np.array([]),
    }

    # Add gate info for StochasticGateNet
    if model_type in ("gate_filter", "gate_no_filter"):
        try:
            result["gate_status"] = model.get_gate_status()
        except Exception:
            pass

    return result


# ======================================================================
# Smoke Test
# ======================================================================

if __name__ == "__main__":
    from data_loader import generate_simulated_data
    from evaluate import compute_all_metrics, print_metrics_table
    from config import SIM_TOTAL_DAYS, SIM_FEATURE_WARMUP, SIM_LOOKBACK

    print("=== Smoke Test: train.py ===\n")

    # Generate small simulated dataset
    returns = generate_simulated_data(SIM_TOTAL_DAYS, seed=42)
    start = SIM_FEATURE_WARMUP + SIM_LOOKBACK
    end = start + 15  # very short test

    # Test model-based
    print("--- Model-based E2E ---")
    res_mb = run_backtest(
        returns, model_type="model_based", loss_type="sharpe",
        lr=10.0, n_steps=5, lookback=SIM_LOOKBACK, rebalance=5,
        start_idx=start, end_idx=end, verbose=True,
    )
    if len(res_mb["portfolio_returns"]) > 0:
        m = compute_all_metrics(res_mb["portfolio_returns"], "model-based")
        print_metrics_table([m])

    # Test model-free
    print("\n--- Model-free E2E ---")
    res_mf = run_backtest(
        returns, model_type="model_free", loss_type="sharpe",
        lr=10.0, n_steps=5, lookback=SIM_LOOKBACK, rebalance=5,
        start_idx=start, end_idx=end, verbose=True,
    )
    if len(res_mf["portfolio_returns"]) > 0:
        m = compute_all_metrics(res_mf["portfolio_returns"], "model-free")
        print_metrics_table([m])

    print("\n✅ Smoke test passed!")
