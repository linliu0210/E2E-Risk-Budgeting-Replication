"""
run_simulation.py — Section 4: Simulation Study.
Ref: Section 4.1-4.2 of Uysal et al. (2021).

Reproduces:
  - Table 1: Performance metrics across 100 seeds
  - Figures 5, 8: Cumulative wealth curves
  - Section 4.2: Statistical hypothesis testing

Setup:
  - 355 days simulated data (30 warmup + 150 train + 175 test)
  - 100 seeds, LR=10, 50 steps, rebalance every 5 days
  - Z-test with top-5 outlier removal
  - D'Agostino-Pearson normality test + Q-Q plots
"""
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import scipy.stats as stats

from config import (
    SIM_TOTAL_DAYS, SIM_FEATURE_WARMUP, SIM_LOOKBACK, SIM_REBALANCE,
    SIM_LR, SIM_N_STEPS, SIM_N_SEEDS, SIM_TEST_DAYS, N_ASSETS,
)
from data_loader import compute_distribution_params, generate_simulated_data
from baselines import equal_weight_portfolio, nominal_risk_parity
from train import run_backtest
from evaluate import (
    compute_all_metrics, print_metrics_table,
    plot_cumulative_returns, return_over_avg_dd, sharpe_ratio,
)


def run_single_seed(
    seed: int,
    mu: np.ndarray,
    cov: np.ndarray,
    loss_type: str = "sharpe",
    verbose: bool = False,
) -> dict:
    """
    Run one simulation seed: model-based, model-free, and baselines.

    Test period starts at SIM_FEATURE_WARMUP + SIM_LOOKBACK, runs for SIM_TEST_DAYS.
    """
    # Generate IID multivariate normal data
    returns = generate_simulated_data(SIM_TOTAL_DAYS, seed=seed, mu=mu, cov=cov)
    test_start = SIM_FEATURE_WARMUP + SIM_LOOKBACK
    test_end = test_start + SIM_TEST_DAYS

    results = {}

    # E2E Model-Based
    res_mb = run_backtest(
        returns, model_type="model_based", loss_type=loss_type,
        lr=SIM_LR, n_steps=SIM_N_STEPS,
        lookback=SIM_LOOKBACK, rebalance=SIM_REBALANCE,
        start_idx=test_start, end_idx=test_end,
        seed=seed, verbose=verbose,
    )
    results["model_based"] = res_mb["portfolio_returns"]

    # E2E Model-Free
    res_mf = run_backtest(
        returns, model_type="model_free", loss_type=loss_type,
        lr=SIM_LR, n_steps=SIM_N_STEPS,
        lookback=SIM_LOOKBACK, rebalance=SIM_REBALANCE,
        start_idx=test_start, end_idx=test_end,
        seed=seed, verbose=verbose,
    )
    results["model_free"] = res_mf["portfolio_returns"]

    # Nominal RP
    rp_rets = nominal_risk_parity(
        returns, test_start, test_end, rebalance=SIM_REBALANCE,
    )
    results["nominal_rp"] = rp_rets

    # Fix-mix (1/N)
    ew_rets = equal_weight_portfolio(returns, test_start, test_end)
    results["fix_mix"] = ew_rets

    return results


def run_statistical_tests(
    scores_mb: np.ndarray,
    scores_mf: np.ndarray,
    score_rp: float,
    alpha: float = 0.01,
    save_dir: str | None = None,
) -> dict:
    """
    Section 4.2: Statistical hypothesis testing.

    Steps:
      1. Remove top-5 outlier seeds (highest Return/Avg.DD scores)
      2. D'Agostino-Pearson normality test
      3. Q-Q plots
      4. Z-test:
         H1: model-free < model-based  (R_free - R_based)
         H2: model-based > nominal RP  (R_based - R_parity)
    """
    n_total = len(scores_mb)
    assert n_total == len(scores_mf) == SIM_N_SEEDS

    # 1. Identify and remove top-5 outliers (by model-based score)
    idx_sorted = np.argsort(scores_mb)
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[idx_sorted[-5:]] = False  # remove top-5
    n_keep = keep_mask.sum()

    mb_trimmed = scores_mb[keep_mask]
    mf_trimmed = scores_mf[keep_mask]

    # 2. D'Agostino-Pearson normality test
    stat_mb, p_mb = normaltest(mb_trimmed)
    stat_mf, p_mf = normaltest(mf_trimmed)

    print(f"\n=== D'Agostino-Pearson Normality Test (n={n_keep}) ===")
    print(f"  Model-based: stat={stat_mb:.4f}, p={p_mb:.4f} "
          f"{'✓ Normal' if p_mb > alpha else '✗ Non-normal'}")
    print(f"  Model-free:  stat={stat_mf:.4f}, p={p_mf:.4f} "
          f"{'✓ Normal' if p_mf > alpha else '✗ Non-normal'}")

    # 3. Q-Q plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        stats.probplot(mb_trimmed, plot=axes[0])
        axes[0].set_title(f"Q-Q: Model-based (n={n_keep})")
        stats.probplot(mf_trimmed, plot=axes[1])
        axes[1].set_title(f"Q-Q: Model-free (n={n_keep})")
        plt.tight_layout()
        qq_path = os.path.join(save_dir, "qq_plots.png")
        plt.savefig(qq_path, dpi=150)
        plt.close()
        print(f"  Q-Q plots saved to {qq_path}")

    # 4. Z-test
    mean_mb = mb_trimmed.mean()
    mean_mf = mf_trimmed.mean()
    std_mb = mb_trimmed.std(ddof=1)
    std_mf = mf_trimmed.std(ddof=1)

    # H1: model-free < model-based → Z1 = (mean_free - mean_based) / SE
    Z1 = (mean_mf - mean_mb) / np.sqrt(std_mf**2 / n_keep + std_mb**2 / n_keep)
    # Z1 << 0 means model-based is significantly better

    # H2: model-based > nominal RP → Z2 = (mean_based - R_parity) / SE
    Z2 = (mean_mb - score_rp) / (std_mb / np.sqrt(n_keep))
    # Z2 >> 0 means model-based is significantly better

    print(f"\n=== Z-test Results (α={alpha}) ===")
    print(f"  H1 (free < based): Z = {Z1:.2f}")
    print(f"    mean_free={mean_mf:.4f}, mean_based={mean_mb:.4f}")
    print(f"  H2 (based > RP):   Z = {Z2:.2f}")
    print(f"    mean_based={mean_mb:.4f}, RP_score={score_rp:.4f}")
    print(f"  Critical Z(0.01, one-sided) = -2.326 / +2.326")

    return {
        "n_kept": n_keep,
        "normality_mb_p": p_mb,
        "normality_mf_p": p_mf,
        "Z1_free_vs_based": Z1,
        "Z2_based_vs_rp": Z2,
        "mean_mb": mean_mb,
        "mean_mf": mean_mf,
        "std_mb": std_mb,
        "std_mf": std_mf,
    }


def main():
    parser = argparse.ArgumentParser(description="Section 4: Simulation Study")
    parser.add_argument("--seeds", type=int, default=SIM_N_SEEDS)
    parser.add_argument("--loss", type=str, default="sharpe",
                        choices=["sharpe", "cumret"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/simulation")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Compute distribution parameters (μ hardcoded, Σ from real data)
    cache_path = os.path.join(os.path.dirname(__file__), "etf_returns.csv")
    mu, cov = compute_distribution_params(cache_path=cache_path)
    print(f"μ (×10000): {(mu * 10000).round(1)}")
    print(f"Σ diag (×10000): {(np.diag(cov) * 10000).round(2)}")

    # Run all seeds
    all_metrics = {"model_based": [], "model_free": [], "nominal_rp": [], "fix_mix": []}
    scores_retdd = {"model_based": [], "model_free": []}

    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{args.seeds}")
        print(f"{'='*60}")

        results = run_single_seed(seed, mu, cov, args.loss, args.verbose)

        for method, rets in results.items():
            if len(rets) > 0:
                metrics = compute_all_metrics(rets, method)
                all_metrics[method].append(metrics)

                if method in scores_retdd:
                    scores_retdd[method].append(metrics["Return/Avg.DD"])

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS — {args.seeds} seeds, loss={args.loss}")
    print(f"{'='*60}")

    summary_rows = []
    for method in ["model_based", "model_free", "nominal_rp", "fix_mix"]:
        if all_metrics[method]:
            metrics_arr = {
                k: [m[k] for m in all_metrics[method]]
                for k in ["Return", "Volatility", "Sharpe", "MDD", "Return/Avg.DD"]
            }
            row = {
                "Portfolio": method,
                "Return (mean)": np.mean(metrics_arr["Return"]),
                "Sharpe (mean)": np.mean(metrics_arr["Sharpe"]),
                "Ret/AvgDD (mean)": np.mean(metrics_arr["Return/Avg.DD"]),
                "Ret/AvgDD (std)": np.std(metrics_arr["Return/Avg.DD"]),
            }
            summary_rows.append(row)
            print(f"\n  {method}: Sharpe={row['Sharpe (mean)']:.4f}, "
                  f"Ret/AvgDD={row['Ret/AvgDD (mean)']:.4f} ± {row['Ret/AvgDD (std)']:.4f}")

    # Statistical testing
    if args.seeds >= 20 and scores_retdd["model_based"]:
        scores_mb = np.array(scores_retdd["model_based"])
        scores_mf = np.array(scores_retdd["model_free"])

        # RP score: use first seed's RP metric (deterministic)
        rp_scores = [m["Return/Avg.DD"] for m in all_metrics["nominal_rp"]]
        rp_score = np.mean(rp_scores)

        test_results = run_statistical_tests(
            scores_mb, scores_mf, rp_score,
            save_dir=args.save_dir,
        )

    # Save example cumulative returns plot (first seed)
    if all_metrics["model_based"]:
        first_seed_results = run_single_seed(0, mu, cov, args.loss)
        plot_cumulative_returns(
            {k: v for k, v in first_seed_results.items() if len(v) > 0},
            title=f"Simulation: Cumulative Returns (seed=0, loss={args.loss})",
            save_path=os.path.join(args.save_dir, f"cumulative_{args.loss}.png"),
        )

    print(f"\n✅ Simulation study complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
