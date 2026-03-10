"""
Evaluation module: portfolio performance metrics and visualization.
Ref: Table 1, 4, 5, 6 of Uysal et al. (2021).

Metrics:
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio
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
    ann_ret = cumulative ** (trading_days / n_days) - 1
    return ann_ret


def annualized_volatility(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized volatility from daily returns."""
    return np.std(daily_returns, ddof=1) * np.sqrt(trading_days)


def sharpe_ratio(daily_returns: np.ndarray, trading_days: int = 252, rf: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    Sharpe = (annualized_return - rf) / annualized_vol
    """
    ann_ret = annualized_return(daily_returns, trading_days)
    ann_vol = annualized_volatility(daily_returns, trading_days)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - rf) / ann_vol


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Maximum drawdown from daily returns."""
    cumulative = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)


def average_drawdown(daily_returns: np.ndarray) -> float:
    """Average drawdown from daily returns."""
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
    """Return / Average Drawdown ratio (main metric in paper simulations)."""
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
        "Calmar Ratio": calmar_ratio(daily_returns),
        "Return/Ave.DD": return_over_avg_dd(daily_returns),
    }


def print_metrics_table(metrics_list: list[dict]):
    """Print metrics as a formatted table (matching paper Table 4 format)."""
    df = pd.DataFrame(metrics_list)
    df = df.set_index("Portfolio")
    # Format to 4 decimal places
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


def sharpe_loss(portfolio_returns: 'torch.Tensor', eps: float = 1e-8) -> 'torch.Tensor':
    """
    Differentiable negative Sharpe ratio loss for training.
    Loss = -mean(R_p) / (std(R_p) + eps)

    "Two risk-reward functions are chosen to train neural networks:
     Sharpe ratio and cumulative return" — Section 3.5.2
    """
    import torch
    mean_ret = portfolio_returns.mean()
    std_ret = portfolio_returns.std() + eps
    return -mean_ret / std_ret


def cumulative_return_loss(portfolio_returns: 'torch.Tensor') -> 'torch.Tensor':
    """
    Differentiable negative cumulative return loss for training.
    Loss = -prod(1 + R_p)
    """
    import torch
    return -torch.prod(1 + portfolio_returns)


if __name__ == "__main__":
    # Quick test with random returns
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.01, 252)

    metrics = compute_all_metrics(daily_rets, name="Random Portfolio")
    print_metrics_table([metrics])
