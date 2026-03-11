"""
run_gates.py — Section 5.4: Stochastic Gates Experiments.
Ref: Tables 6, 7 of Uysal et al. (2021).

Reproduces:
  - Table 6 (Section 5.4.1): 7 ETF, e2e ± gate (with/no filter)
  - Table 7 (Section 5.4.2): 8 assets (7 ETF + 1 random low-vol asset)

NOTE: Table 6 and Table 7 have DIFFERENT hyperparameters!
"""
import numpy as np
import os
import argparse
import pandas as pd

from config import (
    N_ASSETS, MKT_LOOKBACK, MKT_REBALANCE,
    MKT_LR_SHARPE, MKT_STEPS_SHARPE,
    GATE_T6_LR, GATE_T6_MU_LR, GATE_T6_STEPS,  # Table 6
    GATE_T7_LR, GATE_T7_MU_LR, GATE_T7_STEPS,  # Table 7
    GATE_T7_E2E_LR, GATE_T7_E2E_STEPS,          # Table 7 baseline
    OUT_SAMPLE_START, FEATURE_WARMUP,
)
from data_loader import download_etf_data, generate_augmented_universe
from baselines import (
    nominal_risk_parity, nominal_rp_positive, nominal_rp_topk,
    equal_weight_portfolio,
)
from train import run_backtest
from evaluate import compute_all_metrics, print_metrics_table, plot_cumulative_returns


def find_date_index(returns_df: pd.DataFrame, date_str: str) -> int:
    target = pd.Timestamp(date_str)
    idx = returns_df.index.searchsorted(target)
    return min(idx, len(returns_df) - 1)


def run_table6_experiment(
    returns_df: pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """
    Table 6 (Section 5.4.1): 7 ETF with Stochastic Gates.

    Strategies:
      1. E2E-sharpe (no gate):               η=150, n=10  (same as Table 4)
      2. E2E-sharpe + gate (with_filter):     η=150, μ_lr=10, n=10
      3. E2E-sharpe + gate (no_filter):       η=150, μ_lr=10, n=10
      4. Nominal RP
      5. RP-positive
      6. RP-topk (k=4)
    """
    start_idx = max(
        find_date_index(returns_df, OUT_SAMPLE_START),
        FEATURE_WARMUP + MKT_LOOKBACK,
    )
    end_idx = len(returns_df)

    print(f"\n=== Table 6: 7 ETF ===")
    print(f"OOS: {returns_df.index[start_idx].date()} to {returns_df.index[end_idx-1].date()}")

    results = {}

    # 1. E2E no gate
    print("\n  E2E-sharpe (no gate)...")
    res = run_backtest(
        returns_df, model_type="model_based", loss_type="sharpe",
        lr=MKT_LR_SHARPE, n_steps=MKT_STEPS_SHARPE,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx, verbose=verbose,
    )
    results["E2E-sharpe"] = res["portfolio_returns"]

    # 2. E2E + gate (with_filter)
    print("\n  E2E-sharpe + gate (with_filter)...")
    res = run_backtest(
        returns_df, model_type="gate_filter", loss_type="sharpe",
        lr=GATE_T6_LR, n_steps=GATE_T6_STEPS,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        gate_lr=GATE_T6_MU_LR, verbose=verbose,
    )
    results["E2E + gate (with filter)"] = res["portfolio_returns"]
    if "gate_status" in res:
        print(f"    Gate status: {res['gate_status']}")

    # 3. E2E + gate (no_filter)
    print("\n  E2E-sharpe + gate (no_filter)...")
    res = run_backtest(
        returns_df, model_type="gate_no_filter", loss_type="sharpe",
        lr=GATE_T6_LR, n_steps=GATE_T6_STEPS,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        gate_lr=GATE_T6_MU_LR, verbose=verbose,
    )
    results["E2E + gate (no filter)"] = res["portfolio_returns"]
    if "gate_status" in res:
        print(f"    Gate status: {res['gate_status']}")

    # 4. Nominal RP
    print("\n  Nominal RP...")
    results["Nominal RP"] = nominal_risk_parity(
        returns_df, start_idx, end_idx, rebalance=MKT_REBALANCE,
    )

    # 5. RP-positive
    print("\n  RP-positive...")
    results["RP-positive"] = nominal_rp_positive(
        returns_df, start_idx, end_idx, rebalance=MKT_REBALANCE,
    )

    # 6. RP-topk
    print("\n  RP-topk (k=4)...")
    results["RP-topk"] = nominal_rp_topk(
        returns_df, start_idx, end_idx, k=4, rebalance=MKT_REBALANCE,
    )

    return results


def run_table7_experiment(
    returns_df_7: pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """
    Table 7 (Section 5.4.2): 8 assets = 7 ETF + 1 random low-vol asset.

    "We add a random asset with a mean of −0.05% and standard deviation
     of 0.05% to our existing set of seven ETFs." — Section 5.4.2

    NOTE: Different hyperparameters from Table 6!
      - E2E no gate: η=500, n=5
      - E2E + gate:  η=750, μ_lr=750, n=10
    """
    # Create augmented 8-asset universe
    returns_8 = generate_augmented_universe(returns_df_7, seed=42)
    n_assets_aug = 8

    start_idx = max(
        find_date_index(returns_8, OUT_SAMPLE_START),
        FEATURE_WARMUP + MKT_LOOKBACK,
    )
    end_idx = len(returns_8)

    print(f"\n=== Table 7: 8 Assets (7 ETF + 1 random) ===")
    print(f"OOS: {returns_8.index[start_idx].date()} to {returns_8.index[end_idx-1].date()}")
    print(f"RANDOM asset: μ={returns_8['RANDOM'].mean():.6f}, "
          f"σ={returns_8['RANDOM'].std():.6f}")

    results = {}

    # 1. E2E no gate (different HP from Table 6!)
    print("\n  E2E-sharpe (no gate, η=500, n=5)...")
    res = run_backtest(
        returns_8, model_type="model_based", loss_type="sharpe",
        lr=GATE_T7_E2E_LR, n_steps=GATE_T7_E2E_STEPS,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx, verbose=verbose,
    )
    results["E2E-sharpe (8 assets)"] = res["portfolio_returns"]

    # 2. E2E + gate (with_filter)
    print("\n  E2E + gate (with_filter, η=750, μ_lr=750, n=10)...")
    res = run_backtest(
        returns_8, model_type="gate_filter", loss_type="sharpe",
        lr=GATE_T7_LR, n_steps=GATE_T7_STEPS,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        gate_lr=GATE_T7_MU_LR, verbose=verbose,
    )
    results["E2E + gate (with filter, 8)"] = res["portfolio_returns"]
    if "gate_status" in res:
        print(f"    Gate status: {res['gate_status']}")

    # 3. E2E + gate (no_filter)
    print("\n  E2E + gate (no_filter, η=750, μ_lr=750, n=10)...")
    res = run_backtest(
        returns_8, model_type="gate_no_filter", loss_type="sharpe",
        lr=GATE_T7_LR, n_steps=GATE_T7_STEPS,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        gate_lr=GATE_T7_MU_LR, verbose=verbose,
    )
    results["E2E + gate (no filter, 8)"] = res["portfolio_returns"]
    if "gate_status" in res:
        print(f"    Gate status: {res['gate_status']}")

    # 4-6. Baselines on 8 assets
    print("\n  Nominal RP (8 assets)...")
    results["Nominal RP (8)"] = nominal_risk_parity(
        returns_8, start_idx, end_idx,
        rebalance=MKT_REBALANCE, n_assets=n_assets_aug,
    )

    print("\n  RP-positive (8 assets)...")
    results["RP-positive (8)"] = nominal_rp_positive(
        returns_8, start_idx, end_idx, rebalance=MKT_REBALANCE,
    )

    print("\n  RP-topk k=4 (8 assets)...")
    results["RP-topk (8)"] = nominal_rp_topk(
        returns_8, start_idx, end_idx, k=4, rebalance=MKT_REBALANCE,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Section 5.4: Stochastic Gates")
    parser.add_argument("--table", type=str, default="both",
                        choices=["table6", "table7", "both"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/gates")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    cache_path = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    returns_df = download_etf_data(cache_path=cache_path)
    print(f"ETF data: {returns_df.shape}")

    # Table 6
    if args.table in ("table6", "both"):
        results_t6 = run_table6_experiment(returns_df, args.verbose)

        metrics_t6 = []
        for name, rets in results_t6.items():
            if len(rets) > 0:
                metrics_t6.append(compute_all_metrics(rets, name))

        print(f"\n{'='*70}")
        print("TABLE 6: 7 ETF with Stochastic Gates")
        print(f"{'='*70}")
        df_t6 = print_metrics_table(metrics_t6)
        df_t6.to_csv(os.path.join(args.save_dir, "table6.csv"))

        plot_cumulative_returns(
            results_t6,
            title="Table 6: 7 ETF with Stochastic Gates (OOS)",
            save_path=os.path.join(args.save_dir, "table6_cumulative.png"),
        )

    # Table 7
    if args.table in ("table7", "both"):
        results_t7 = run_table7_experiment(returns_df, args.verbose)

        metrics_t7 = []
        for name, rets in results_t7.items():
            if len(rets) > 0:
                metrics_t7.append(compute_all_metrics(rets, name))

        print(f"\n{'='*70}")
        print("TABLE 7: 8 Assets (7 ETF + 1 random) with Stochastic Gates")
        print(f"{'='*70}")
        df_t7 = print_metrics_table(metrics_t7)
        df_t7.to_csv(os.path.join(args.save_dir, "table7.csv"))

        plot_cumulative_returns(
            results_t7,
            title="Table 7: 8 Assets with Stochastic Gates (OOS)",
            save_path=os.path.join(args.save_dir, "table7_cumulative.png"),
        )

    print(f"\n✅ Stochastic Gates experiments complete! Results in {args.save_dir}")


if __name__ == "__main__":
    main()
