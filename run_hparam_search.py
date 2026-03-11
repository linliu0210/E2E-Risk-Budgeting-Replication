"""
run_hparam_search.py — Section 5.1: Hyperparameter Grid Search.
Ref: Tables 2, 3 of Uysal et al. (2021).

Reproduces:
  - Table 2: Grid search for Sharpe loss (η × n) → Train/Val Sharpe
  - Table 3: Grid search for Cumulative Return loss (η × n) → Train/Val Return

Setup:
  - Train: 2011-2014
  - Validation: 2015-2016
  - Grid: η ∈ {50, 100, 150, 200, 300, 500}, n ∈ {5, 10, 15, 20, 25, 50}
"""
import numpy as np
import os
import time
import argparse
import pandas as pd

from config import (
    MKT_LOOKBACK, MKT_REBALANCE,
    HP_LR_CANDIDATES, HP_STEPS_CANDIDATES,
    IN_SAMPLE_START, TRAIN_END, VALIDATION_END,
    FEATURE_WARMUP,
)
from data_loader import download_etf_data
from train import run_backtest
from evaluate import sharpe_ratio, annualized_return


def find_date_index(returns_df: pd.DataFrame, date_str: str) -> int:
    target = pd.Timestamp(date_str)
    idx = returns_df.index.searchsorted(target)
    return min(idx, len(returns_df) - 1)


def grid_search(
    returns_df: pd.DataFrame,
    model_type: str = "model_based",
    loss_type: str = "sharpe",
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Grid search over (η, n) pairs.

    Train: IN_SAMPLE_START to TRAIN_END
    Validation: TRAIN_END to VALIDATION_END

    Returns:
        best_params: {"lr": ..., "n_steps": ...}
        results_df: full grid results
    """
    # Period boundaries
    train_start = max(
        find_date_index(returns_df, IN_SAMPLE_START),
        FEATURE_WARMUP + MKT_LOOKBACK,
    )
    train_end = find_date_index(returns_df, TRAIN_END)
    val_start = train_end
    val_end = find_date_index(returns_df, VALIDATION_END)

    print(f"  Train:  idx {train_start} to {train_end} "
          f"({returns_df.index[train_start].date()} to {returns_df.index[train_end].date()})")
    print(f"  Val:    idx {val_start} to {val_end} "
          f"({returns_df.index[val_start].date()} to {returns_df.index[val_end].date()})")

    results = []

    for lr in HP_LR_CANDIDATES:
        for n_steps in HP_STEPS_CANDIDATES:
            print(f"\n  η={lr}, n={n_steps} ...", end=" ", flush=True)
            t0 = time.time()

            # Train period evaluation
            res_train = run_backtest(
                returns_df, model_type=model_type, loss_type=loss_type,
                lr=lr, n_steps=n_steps,
                lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
                start_idx=train_start, end_idx=train_end,
                verbose=False,
            )

            # Validation period evaluation
            res_val = run_backtest(
                returns_df, model_type=model_type, loss_type=loss_type,
                lr=lr, n_steps=n_steps,
                lookback=MKT_LOOKBACK, rebalance=MKT_REBALANCE,
                start_idx=val_start, end_idx=val_end,
                verbose=False,
            )

            elapsed = time.time() - t0

            train_rets = res_train["portfolio_returns"]
            val_rets = res_val["portfolio_returns"]

            if loss_type == "sharpe":
                train_metric = sharpe_ratio(train_rets) if len(train_rets) > 1 else 0
                val_metric = sharpe_ratio(val_rets) if len(val_rets) > 1 else 0
                metric_name = "Sharpe"
            else:
                train_metric = annualized_return(train_rets) if len(train_rets) > 0 else 0
                val_metric = annualized_return(val_rets) if len(val_rets) > 0 else 0
                metric_name = "Return"

            row = {
                "η": lr,
                "n": n_steps,
                f"Train {metric_name}": train_metric,
                f"Val {metric_name}": val_metric,
                "Time (s)": elapsed,
            }
            results.append(row)
            print(f"Train={train_metric:.4f}, Val={val_metric:.4f}, "
                  f"Time={elapsed:.1f}s")

    results_df = pd.DataFrame(results)

    # Find best validation metric
    val_col = [c for c in results_df.columns if c.startswith("Val")][0]
    best_idx = results_df[val_col].idxmax()
    best_row = results_df.iloc[best_idx]
    best_params = {"lr": best_row["η"], "n_steps": int(best_row["n"])}

    print(f"\n  ★ Best: η={best_params['lr']}, n={best_params['n_steps']}, "
          f"Val {val_col}={best_row[val_col]:.4f}")

    return best_params, results_df


def main():
    parser = argparse.ArgumentParser(description="Section 5.1: Hyperparameter Search")
    parser.add_argument("--loss", type=str, default="sharpe",
                        choices=["sharpe", "cumret"])
    parser.add_argument("--model", type=str, default="model_based",
                        choices=["model_based", "model_free"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/hparam_search")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    cache_path = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    returns_df = download_etf_data(cache_path=cache_path)
    print(f"ETF data: {returns_df.shape}")

    # Run grid search
    print(f"\n{'='*60}")
    print(f"Grid Search: {args.model}, loss={args.loss}")
    print(f"  η candidates: {HP_LR_CANDIDATES}")
    print(f"  n candidates: {HP_STEPS_CANDIDATES}")
    print(f"  Total configs: {len(HP_LR_CANDIDATES) * len(HP_STEPS_CANDIDATES)}")
    print(f"{'='*60}")

    best, df = grid_search(returns_df, args.model, args.loss, args.verbose)

    # Save results as Table 2/3 format
    table_path = os.path.join(args.save_dir, f"table_{args.loss}_{args.model}.csv")
    df.to_csv(table_path, index=False)
    print(f"\nResults saved to {table_path}")

    # Pivot table (η × n matrix, matching paper format)
    val_col = [c for c in df.columns if c.startswith("Val")][0]
    pivot = df.pivot(index="η", columns="n", values=val_col)
    print(f"\nPivot Table ({val_col}):")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    pivot_path = os.path.join(args.save_dir, f"pivot_{args.loss}_{args.model}.csv")
    pivot.to_csv(pivot_path)

    print(f"\n✅ Grid search complete! Best: η={best['lr']}, n={best['n_steps']}")


if __name__ == "__main__":
    main()
