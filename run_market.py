"""
Run real market data experiment (Section 5 of Uysal et al. 2021).

"We present the computational results over the last ten years of daily data,
 where we keep the first six years (2011-2016/12) for hyperparameter search
 and the remaining years (2017-2021) for out-of-sample testing." — Section 5.1

ETFs: VTI, IWM, AGG, LQD, MUB, DBC, GLD
"""
import os
import numpy as np
import pandas as pd
import argparse

from data_loader import download_etf_data
from train import run_backtest
from baselines import nominal_risk_parity, equal_weight_portfolio
from evaluate import (
    compute_all_metrics, print_metrics_table,
    plot_cumulative_returns,
)
from config import (
    ETF_TICKERS, N_ASSETS,
    IN_SAMPLE_START, IN_SAMPLE_END,
    OUT_SAMPLE_START, OUT_SAMPLE_END,
    MKT_LOOKBACK, MKT_REBALANCE,
    MKT_LR_SHARPE, MKT_STEPS_SHARPE,
    MKT_LR_RETURN, MKT_STEPS_RETURN,
)


def find_date_index(returns_df, date_str: str) -> int:
    """Find the index of the nearest date >= date_str."""
    target = pd.Timestamp(date_str)
    mask = returns_df.index >= target
    if mask.any():
        return mask.values.argmax()
    return len(returns_df) - 1


def run_market_experiment(
    loss_type: str = "sharpe",
    period: str = "out_sample",
    verbose: bool = True,
):
    """
    Run the market data experiment.

    Args:
        loss_type: "sharpe" or "cumret"
        period: "in_sample", "out_sample", or "full"
        verbose: print progress
    """
    # 1. Download / load data
    cache_path = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    returns_df = download_etf_data(cache_path=cache_path)

    print(f"Data loaded: {returns_df.shape}")
    print(f"Date range: {returns_df.index[0].strftime('%Y-%m-%d')} to "
          f"{returns_df.index[-1].strftime('%Y-%m-%d')}")

    # 2. Determine period
    if period == "in_sample":
        start_date, end_date = IN_SAMPLE_START, IN_SAMPLE_END
    elif period == "out_sample":
        start_date, end_date = OUT_SAMPLE_START, OUT_SAMPLE_END
    else:
        start_date, end_date = IN_SAMPLE_START, OUT_SAMPLE_END

    start_idx = find_date_index(returns_df, start_date)
    end_idx = find_date_index(returns_df, end_date)

    print(f"\nRunning {period} period: {returns_df.index[start_idx].strftime('%Y-%m-%d')} "
          f"to {returns_df.index[end_idx].strftime('%Y-%m-%d')}")

    # 3. Select hyperparameters based on loss type
    if loss_type == "sharpe":
        lr = MKT_LR_SHARPE
        n_steps = MKT_STEPS_SHARPE
    else:
        lr = MKT_LR_RETURN
        n_steps = MKT_STEPS_RETURN

    # 4. Run model-based E2E
    print(f"\n{'='*60}")
    print(f"Model-based E2E (loss={loss_type}, lr={lr}, steps={n_steps})")
    print(f"{'='*60}")
    e2e_mb = run_backtest(
        returns_df,
        model_type="model_based",
        loss_type=loss_type,
        lr=lr,
        n_steps=n_steps,
        lookback=MKT_LOOKBACK,
        rebalance=MKT_REBALANCE,
        start_idx=start_idx,
        end_idx=end_idx,
        verbose=verbose,
    )

    # 5. Run model-free E2E (same hyperparameters for fair comparison)
    print(f"\n{'='*60}")
    print(f"Model-free E2E (loss={loss_type}, lr={lr}, steps={n_steps})")
    print(f"{'='*60}")
    e2e_mf = run_backtest(
        returns_df,
        model_type="model_free",
        loss_type=loss_type,
        lr=lr,
        n_steps=n_steps,
        lookback=MKT_LOOKBACK,
        rebalance=MKT_REBALANCE,
        start_idx=start_idx,
        end_idx=end_idx,
        verbose=verbose,
    )

    # 6. Run baselines
    print(f"\n{'='*60}")
    print("Baselines")
    print(f"{'='*60}")

    rp_rets = nominal_risk_parity(
        returns_df, start_idx=start_idx, end_idx=end_idx,
        rebalance=MKT_REBALANCE,
    )
    ew_rets = equal_weight_portfolio(
        returns_df, start_idx=start_idx, end_idx=end_idx,
    )

    # 7. Compile and display results (matching Table 4/5)
    print(f"\n{'='*60}")
    print(f"RESULTS — {period} ({start_date} to {end_date})")
    print(f"{'='*60}")

    metrics = [
        compute_all_metrics(e2e_mb["portfolio_returns"], f"e2e-{loss_type} (model-based)"),
        compute_all_metrics(e2e_mf["portfolio_returns"], f"e2e-{loss_type} (model-free)"),
        compute_all_metrics(rp_rets, "Nominal RP"),
        compute_all_metrics(ew_rets, "Fix-mix (1/N)"),
    ]
    df = print_metrics_table(metrics)

    # 8. Plot cumulative returns
    plot_data = {
        f"e2e-{loss_type} (model-based)": e2e_mb["portfolio_returns"],
        f"e2e-{loss_type} (model-free)": e2e_mf["portfolio_returns"],
        "Nominal RP": rp_rets,
        "Fix-mix (1/N)": ew_rets,
    }

    plot_path = os.path.join(
        os.path.dirname(__file__),
        f"cumulative_returns_{period}_{loss_type}.png"
    )
    plot_cumulative_returns(
        plot_data,
        title=f"Portfolio Performance — {period} ({loss_type})",
        dates=np.array(e2e_mb["dates"]) if len(e2e_mb["dates"]) == len(rp_rets) else None,
        save_path=plot_path,
    )

    return metrics, e2e_mb, e2e_mf, rp_rets, ew_rets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run market data experiment (Section 5)")
    parser.add_argument("--loss", type=str, default="sharpe", choices=["sharpe", "cumret"])
    parser.add_argument("--period", type=str, default="out_sample",
                        choices=["in_sample", "out_sample", "full"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_market_experiment(
        loss_type=args.loss,
        period=args.period,
        verbose=args.verbose,
    )
