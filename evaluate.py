"""
Evaluation module: portfolio performance metrics and visualization.
Ref: Tables 1, 4, 5, 6, 7 of Uysal et al. (2021).

Metrics:
  - Annualized Return (geometric)
  - Annualized Volatility
  - Sharpe Ratio (annualized)
  - Maximum Drawdown (MDD)
  - Calmar Ratio (Return / MDD)
  - Return / Average Drawdown
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def annualized_return(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized geometric return from daily simple returns."""
    cumulative = np.prod(1 + daily_returns)
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    return cumulative ** (trading_days / n_days) - 1


def annualized_volatility(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized volatility from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return np.std(daily_returns, ddof=1) * np.sqrt(trading_days)


def sharpe_ratio(daily_returns: np.ndarray, trading_days: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio = (ann_return - rf) / ann_vol."""
    ann_ret = annualized_return(daily_returns, trading_days)
    ann_vol = annualized_volatility(daily_returns, trading_days)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - rf) / ann_vol


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Maximum drawdown from daily returns."""
    if len(daily_returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)


def average_drawdown(daily_returns: np.ndarray) -> float:
    """Average drawdown from daily returns."""
    if len(daily_returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.mean(drawdown)


def calmar_ratio(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    ann_ret = annualized_return(daily_returns, trading_days)
    mdd = max_drawdown(daily_returns)
    if mdd < 1e-10:
        return 0.0
    return ann_ret / mdd


def return_over_avg_dd(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Return / Average Drawdown ratio.
    Main metric for simulation hypothesis testing (Section 4.2).
    """
    ann_ret = annualized_return(daily_returns, trading_days)
    avg_dd = average_drawdown(daily_returns)
    if avg_dd < 1e-10:
        return 0.0
    return ann_ret / avg_dd


def compute_all_metrics(daily_returns: np.ndarray, name: str = "") -> dict:
    """Compute all portfolio performance metrics."""
    return {
        "Portfolio": name,
        "Return": annualized_return(daily_returns),
        "Volatility": annualized_volatility(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns),
        "MDD": max_drawdown(daily_returns),
        "Calmar": calmar_ratio(daily_returns),
        "Return/Avg.DD": return_over_avg_dd(daily_returns),
    }


def print_metrics_table(metrics_list: list[dict]) -> pd.DataFrame:
    """Print metrics as a formatted table (matching paper Table 4/5 format)."""
    df = pd.DataFrame(metrics_list)
    df = df.set_index("Portfolio")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def plot_cumulative_returns(
    returns_dict: dict[str, np.ndarray],
    title: str = "Cumulative Portfolio Performance",
    dates: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Plot cumulative returns for multiple strategies.

    Args:
        returns_dict: {strategy_name: daily_returns_array}
        title: plot title
        dates: optional date index
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for name, rets in returns_dict.items():
        cum = np.cumprod(1 + rets)
        if dates is not None and len(dates) == len(cum):
            ax.plot(dates, cum, label=name)
        else:
            ax.plot(cum, label=name)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cumulative Wealth", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved plot to {save_path}")

    plt.close()
    return fig


# ======================================================================
# Main (smoke test)
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.01, 252)

    metrics = compute_all_metrics(daily_rets, name="Random Portfolio")
    print_metrics_table([metrics])
    print("✅ Evaluate tests passed!")
