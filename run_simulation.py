"""
Run simulation study (Section 4 of Uysal et al. 2021).

"We simulate a seven-asset universe where the returns of the assets follow
 a multi-variate normal distribution, and is independently and identically
 distributed for each trading day." — Section 4.1

Hyperparameters for simulation (Section 4.2):
  - Hidden neurons: 32
  - Learning rate: 10
  - Training steps: 50
  - Rolling window: 150 days
  - Test window: 5 days
  - Total simulation: 175 days
  - Seeds: 100
"""
import sys
import os
import numpy as np
import argparse

from data_loader import generate_simulated_data
from train import run_backtest
from baselines import nominal_risk_parity, equal_weight_portfolio
from evaluate import (
    compute_all_metrics, print_metrics_table,
    plot_cumulative_returns, return_over_avg_dd, sharpe_ratio,
)
from config import SIM_LOOKBACK, SIM_REBALANCE, SIM_LR, SIM_N_STEPS, SIM_DAYS


def run_simulation(
    n_seeds: int = 5,
    n_days: int = SIM_DAYS + SIM_LOOKBACK,
    loss_type: str = "sharpe",
    verbose: bool = False,
):
    """
    Run full simulation study with multiple seeds.

    For each seed:
      1. Generate fresh simulated data
      2. Run model-based E2E
      3. Run model-free E2E
      4. Run nominal risk parity baseline
    """
    print(f"="*70)
    print(f"Simulation Study: {n_seeds} seeds, loss={loss_type}")
    print(f"Lookback={SIM_LOOKBACK}, Rebalance={SIM_REBALANCE}, "
          f"LR={SIM_LR}, Steps={SIM_N_STEPS}")
    print(f"="*70)

    # Collect performance metrics across seeds
    model_based_metrics = []
    model_free_metrics = []
    rp_metrics = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")

        # Generate simulated data
        returns = generate_simulated_data(n_days=n_days, seed=seed)
        start_idx = SIM_LOOKBACK
        end_idx = SIM_LOOKBACK + SIM_DAYS

        # 1. Model-based E2E
        print("  Running model-based E2E...")
        mb_results = run_backtest(
            returns,
            model_type="model_based",
            loss_type=loss_type,
            lr=SIM_LR,
            n_steps=SIM_N_STEPS,
            lookback=SIM_LOOKBACK,
            rebalance=SIM_REBALANCE,
            start_idx=start_idx,
            end_idx=end_idx,
            seed=seed,
            verbose=verbose,
        )
        mb_m = compute_all_metrics(mb_results["portfolio_returns"], "model-based")
        model_based_metrics.append(mb_m)
        print(f"  Model-based: Sharpe={mb_m['Sharpe']:.4f}, "
              f"Ret/AvgDD={mb_m['Return/Ave.DD']:.4f}")

        # 2. Model-free E2E
        print("  Running model-free E2E...")
        mf_results = run_backtest(
            returns,
            model_type="model_free",
            loss_type=loss_type,
            lr=SIM_LR,
            n_steps=SIM_N_STEPS,
            lookback=SIM_LOOKBACK,
            rebalance=SIM_REBALANCE,
            start_idx=start_idx,
            end_idx=end_idx,
            seed=seed,
            verbose=verbose,
        )
        mf_m = compute_all_metrics(mf_results["portfolio_returns"], "model-free")
        model_free_metrics.append(mf_m)
        print(f"  Model-free:  Sharpe={mf_m['Sharpe']:.4f}, "
              f"Ret/AvgDD={mf_m['Return/Ave.DD']:.4f}")

        # 3. Nominal Risk Parity (same data, deterministic)
        rp_rets = nominal_risk_parity(
            returns, start_idx=start_idx, end_idx=end_idx,
            rebalance=SIM_REBALANCE,
        )
        rp_m = compute_all_metrics(rp_rets, "nominal-RP")
        rp_metrics.append(rp_m)
        print(f"  Nominal RP:  Sharpe={rp_m['Sharpe']:.4f}, "
              f"Ret/AvgDD={rp_m['Return/Ave.DD']:.4f}")

    # Aggregate results
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({n_seeds} seeds, loss={loss_type})")
    print(f"{'='*70}")

    # Hypothesis testing (Section 4.2)
    mb_ratios = [m["Return/Ave.DD"] for m in model_based_metrics]
    mf_ratios = [m["Return/Ave.DD"] for m in model_free_metrics]
    rp_ratio = np.mean([m["Return/Ave.DD"] for m in rp_metrics])

    mb_sharpes = [m["Sharpe"] for m in model_based_metrics]
    mf_sharpes = [m["Sharpe"] for m in model_free_metrics]

    print(f"\nModel-based Return/AvgDD: mean={np.mean(mb_ratios):.3f}, "
          f"std={np.std(mb_ratios):.3f}")
    print(f"Model-free  Return/AvgDD: mean={np.mean(mf_ratios):.3f}, "
          f"std={np.std(mf_ratios):.3f}")
    print(f"Nominal RP  Return/AvgDD: {rp_ratio:.3f}")

    print(f"\nModel-based Sharpe: mean={np.mean(mb_sharpes):.3f}, "
          f"std={np.std(mb_sharpes):.3f}")
    print(f"Model-free  Sharpe: mean={np.mean(mf_sharpes):.3f}, "
          f"std={np.std(mf_sharpes):.3f}")

    # Hypothesis 1: model-free < model-based?
    if n_seeds >= 2:
        z1_num = np.mean(mf_ratios) - np.mean(mb_ratios)
        z1_den = np.sqrt(np.std(mf_ratios)**2/n_seeds + np.std(mb_ratios)**2/n_seeds)
        z1 = z1_num / (z1_den + 1e-10)
        print(f"\nHypothesis 1 (free < based): Z = {z1:.2f}")
        print(f"  {'REJECT H0 (model-based wins)' if z1 < -1.645 else 'Cannot reject H0'} at 5%")

        # Hypothesis 2: model-based > RP?
        z2 = (np.mean(mb_ratios) - rp_ratio) / (np.std(mb_ratios)/np.sqrt(n_seeds) + 1e-10)
        print(f"Hypothesis 2 (based > RP):   Z = {z2:.2f}")
        print(f"  {'REJECT H0 (model-based beats RP)' if z2 > 1.645 else 'Cannot reject H0'} at 5%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation study (Section 4)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--loss", type=str, default="sharpe", choices=["sharpe", "cumret"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_simulation(
        n_seeds=args.seeds,
        loss_type=args.loss,
        verbose=args.verbose,
    )
