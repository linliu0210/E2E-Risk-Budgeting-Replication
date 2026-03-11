"""
run_market.py — Section 5.2-5.3: Market Data Experiments.
Ref: Tables 4, 5 of Uysal et al. (2021).

Reproduces:
  - Table 4: model-based E2E vs Nominal RP vs Fix-mix (Sharpe & Cum. Return)
  - Table 5: model-based vs model-free comparison

Setup:
  - 7 ETFs, 2011-2021 data
  - In-sample: 2011-2016, Out-of-sample: 2017-2021
  - K=150, rebalance=25, LR/steps from Table 2/3 grid search
"""
import numpy as np
import os
import argparse
import pandas as pd

from config import (
    ETF_TICKERS, N_ASSETS,
    MKT_LOOKBACK, MKT_REBALANCE,
    MKT_LR_SHARPE, MKT_STEPS_SHARPE,
    MKT_LR_RETURN, MKT_STEPS_RETURN,
    IN_SAMPLE_START, OUT_SAMPLE_START, OUT_SAMPLE_END,
    FEATURE_WARMUP,
)
from data_loader import download_etf_data
from baselines import equal_weight_portfolio, nominal_risk_parity
from train import run_backtest
from evaluate import (
    compute_all_metrics, print_metrics_table,
    plot_cumulative_returns,
)


def find_date_index(returns_df: pd.DataFrame, date_str: str) -> int:
    """Find the nearest index for a given date string."""
    target = pd.Timestamp(date_str)
    idx = returns_df.index.searchsorted(target)
    return min(idx, len(returns_df) - 1)


def run_market_experiment(
    returns_df: pd.DataFrame,
    period: str = "out_sample",
    loss_type: str = "sharpe",
    verbose: bool = False,
) -> dict:
    """
    Run market data experiment for Tables 4/5.

    Args:
        returns_df: full ETF returns DataFrame
        period: "in_sample", "out_sample", or "full"
        loss_type: "sharpe" or "cumret"
        verbose: print progress

    Returns:
        dict mapping strategy name → daily returns array
    """
    # Determine period boundaries
    if period == "out_sample":
        start_idx = find_date_index(returns_df, OUT_SAMPLE_START)
        end_idx = len(returns_df)
    elif period == "in_sample":
        start_idx = find_date_index(returns_df, IN_SAMPLE_START)
        end_idx = find_date_index(returns_df, OUT_SAMPLE_START)
    elif period == "full":
        start_idx = max(FEATURE_WARMUP + MKT_LOOKBACK,
                       find_date_index(returns_df, IN_SAMPLE_START) + MKT_LOOKBACK)
        end_idx = len(returns_df)
    else:
        raise ValueError(f"Unknown period: {period}")

    # Ensure enough history
    start_idx = max(start_idx, FEATURE_WARMUP + MKT_LOOKBACK)

    print(f"\nPeriod: {period}")
    print(f"  Dates: {returns_df.index[start_idx]} to {returns_df.index[end_idx-1]}")
    print(f"  Days: {end_idx - start_idx}")

    # Select learning rate and steps based on loss type
    if loss_type == "sharpe":
        lr, n_steps = MKT_LR_SHARPE, MKT_STEPS_SHARPE
    else:
        lr, n_steps = MKT_LR_RETURN, MKT_STEPS_RETURN

    results = {}

    # --- E2E Model-Based ---
    print(f"\n  Running E2E Model-Based ({loss_type}, η={lr}, n={n_steps})...")
    res = run_backtest(
        returns_df, model_type="model_based", loss_type=loss_type,
        lr=lr, n_steps=n_steps,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        verbose=verbose,
    )
    results[f"E2E-{loss_type} (model-based)"] = res["portfolio_returns"]

    # --- E2E Model-Free ---
    print(f"\n  Running E2E Model-Free ({loss_type}, η={lr}, n={n_steps})...")
    res = run_backtest(
        returns_df, model_type="model_free", loss_type=loss_type,
        lr=lr, n_steps=n_steps,
        lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
        start_idx=start_idx, end_idx=end_idx,
        verbose=verbose,
    )
    results[f"E2E-{loss_type} (model-free)"] = res["portfolio_returns"]

    # --- Nominal RP ---
    print("\n  Running Nominal Risk Parity...")
    rp_rets = nominal_risk_parity(
        returns_df, start_idx, end_idx, rebalance=MKT_REBALANCE,
    )
    results["Nominal RP"] = rp_rets

    # --- Fix-mix (1/N) ---
    print("\n  Running Fix-mix (1/N)...")
    ew_rets = equal_weight_portfolio(returns_df, start_idx, end_idx)
    results["Fix-mix (1/N)"] = ew_rets

    return results


def main():
    parser = argparse.ArgumentParser(description="Section 5.2-5.3: Market Experiments")
    parser.add_argument("--loss", type=str, default="sharpe",
                        choices=["sharpe", "cumret"])
    parser.add_argument("--period", type=str, default="out_sample",
                        choices=["in_sample", "out_sample", "full"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/market")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load real ETF data
    cache_path = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    returns_df = download_etf_data(cache_path=cache_path)
    print(f"ETF data: {returns_df.shape}, {returns_df.index[0]} to {returns_df.index[-1]}")

    # Run experiment
    results = run_market_experiment(returns_df, args.period, args.loss, args.verbose)

    # Compute and print metrics
    metrics_list = []
    for name, rets in results.items():
        if len(rets) > 0:
            m = compute_all_metrics(rets, name)
            metrics_list.append(m)

    print(f"\n{'='*70}")
    print(f"RESULTS — Period: {args.period}, Loss: {args.loss}")
    print(f"{'='*70}")
    df = print_metrics_table(metrics_list)

    # Save
    df.to_csv(os.path.join(args.save_dir, f"table_{args.period}_{args.loss}.csv"))

    # Plot
    plot_cumulative_returns(
        results,
        title=f"Market: {args.period}, loss={args.loss}",
        save_path=os.path.join(args.save_dir, f"cumulative_{args.period}_{args.loss}.png"),
    )

    print(f"\n✅ Market experiment complete! Results in {args.save_dir}")


if __name__ == "__main__":
    main()
