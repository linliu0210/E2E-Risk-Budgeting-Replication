"""
Rolling-window training loop implementing Algorithm 1.
Ref: Algorithm 1, Section 3.5.2 of Uysal et al. (2021).

"In the backtesting framework, the neural network models are re-trained
 every 25 days with a look-back window of 150 days." — Section 5.1

"We use linear update rules for learning rates and decrease it by the
 factor of 0.9 in every three steps." — Section 5.1
"""
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from model_based_net import ModelBasedNet
from model_free_net import ModelFreeNet
from features import build_features, estimate_covariance
from evaluate import sharpe_loss, cumulative_return_loss
from config import (
    N_ASSETS, HIDDEN_DIM, LEAKY_RELU_ALPHA,
    LR_DECAY_FACTOR, LR_DECAY_EVERY, RB_EPSILON,
)


def train_one_batch(
    model: nn.Module,
    returns_df,
    train_days: list[int],
    lr: float,
    n_steps: int,
    loss_type: str = "sharpe",
    verbose: bool = False,
) -> nn.Module:
    """
    Train the model on one batch (one rolling window).
    Implements the inner loop of Algorithm 1 (lines 3-11).

    Args:
        model: ModelBasedNet or ModelFreeNet
        returns_df: full returns DataFrame
        train_days: list of day indices in training window
        lr: initial learning rate
        n_steps: number of gradient steps
        loss_type: "sharpe" or "cumret"
        verbose: print training progress

    Returns:
        Trained model (modified in-place)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Learning rate scheduler: decay by 0.9 every 3 steps
    # "We use linear update rules for learning rates and decrease it
    #  by the factor of 0.9 in every three steps." — Section 5.1
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_EVERY, gamma=LR_DECAY_FACTOR
    )

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward pass through all training days
        portfolio_returns = []

        for s in train_days:
            # Build features for day s (non-anticipativity: uses data up to s-1)
            feat = build_features(returns_df, s)
            x_s = torch.tensor(feat, dtype=torch.float32)

            # Estimate covariance for day s
            cov_s = estimate_covariance(returns_df, s)
            Sigma_s = torch.tensor(cov_s, dtype=torch.float64)

            # Forward: get allocation
            try:
                z_s, b_s = model(x_s, Sigma_s)
            except Exception as e:
                if verbose:
                    print(f"  [train] Solver failed at step={step}, day={s}: {e}")
                continue

            # Realized return on day s
            # Eq: r_s = z_s^T * actual_returns_s
            actual_ret = torch.tensor(
                returns_df.iloc[s].values, dtype=torch.float32
            )
            port_ret = torch.dot(z_s.squeeze(), actual_ret)
            portfolio_returns.append(port_ret)

        if len(portfolio_returns) < 2:
            if verbose:
                print(f"  [train] Step {step}: too few valid days, skipping")
            continue

        portfolio_returns = torch.stack(portfolio_returns)

        # Compute loss (lines 10-11 of Algorithm 1)
        if loss_type == "sharpe":
            loss = sharpe_loss(portfolio_returns)
        elif loss_type == "cumret":
            loss = cumulative_return_loss(portfolio_returns)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Backward + update
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if verbose and (step % max(1, n_steps // 5) == 0):
            print(f"  Step {step}/{n_steps}, loss={loss.item():.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.4f}")

    return model


def run_backtest(
    returns_df,
    model_type: str = "model_based",
    loss_type: str = "sharpe",
    lr: float = 150.0,
    n_steps: int = 10,
    lookback: int = 150,
    rebalance: int = 25,
    start_idx: int | None = None,
    end_idx: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run full rolling-window backtest (Algorithm 1 outer loop).

    "For every [rebalance]-day period, we train the neural network with
     the data of [lookback] days immediately previous to the period of
     interest. We keep the same weights for the [rebalance]-day period,
     and repeat the same process for the next period." — Section 4.2

    Args:
        returns_df: full returns DataFrame
        model_type: "model_based" or "model_free"
        loss_type: "sharpe" or "cumret"
        lr: initial learning rate
        n_steps: number of training steps per batch
        lookback: rolling window size K
        rebalance: rebalance frequency
        start_idx: first test day (default: lookback)
        end_idx: last test day (default: len(returns))
        seed: random seed for NN initialization
        verbose: print progress

    Returns:
        dict with portfolio_returns, all_weights, all_budgets, dates
    """
    T = len(returns_df)
    min_history = 30  # minimum history needed for features

    if start_idx is None:
        start_idx = max(lookback, min_history)
    if end_idx is None:
        end_idx = T

    # Feature dimension
    n_features = N_ASSETS * (5 + 3 * 2)  # 77 for 7 assets

    # Rebalancing days
    rebal_days = list(range(start_idx, end_idx, rebalance))

    all_portfolio_returns = []
    all_weights = []
    all_budgets = []
    all_dates = []

    for batch_idx, t in enumerate(rebal_days):
        if verbose:
            print(f"\n[Batch {batch_idx+1}/{len(rebal_days)}] "
                  f"Rebalancing day {t} ({returns_df.index[t].strftime('%Y-%m-%d')})")

        # 1. Initialize fresh model (line 2 of Algorithm 1)
        # "Initialization: weights θ" — each batch starts from scratch
        torch.manual_seed(seed)
        if model_type == "model_based":
            model = ModelBasedNet(n_features=n_features)
        else:
            model = ModelFreeNet(n_features=n_features)

        # 2. Training days: [t-K, ..., t-1]
        train_start = max(min_history, t - lookback)
        train_days = list(range(train_start, t))

        # 3. Train on rolling window (inner loop, lines 3-11)
        model = train_one_batch(
            model, returns_df, train_days,
            lr=lr, n_steps=n_steps, loss_type=loss_type,
            verbose=verbose,
        )

        # 4. Apply trained model to test period [t, t+rebalance)
        # "With trained θ*, calculate risk contribution on day t and
        #  allocate accordingly" — line 12
        model.eval()
        test_end = min(t + rebalance, end_idx)

        with torch.no_grad():
            for s in range(t, test_end):
                feat = build_features(returns_df, s)
                x_s = torch.tensor(feat, dtype=torch.float32)
                cov_s = estimate_covariance(returns_df, s)
                Sigma_s = torch.tensor(cov_s, dtype=torch.float64)

                try:
                    z_s, b_s = model(x_s, Sigma_s)
                    weights_s = z_s.squeeze().numpy()
                    budgets_s = b_s.squeeze().numpy()
                except Exception as e:
                    if verbose:
                        print(f"  [test] Solver failed at day {s}: {e}, using equal weight")
                    weights_s = np.ones(N_ASSETS) / N_ASSETS
                    budgets_s = np.ones(N_ASSETS) / N_ASSETS

                # Realized return (line 13)
                actual_ret = returns_df.iloc[s].values
                port_ret = np.dot(weights_s, actual_ret)

                all_portfolio_returns.append(port_ret)
                all_weights.append(weights_s)
                all_budgets.append(budgets_s)
                all_dates.append(returns_df.index[s])

    results = {
        "portfolio_returns": np.array(all_portfolio_returns),
        "weights": np.array(all_weights),
        "budgets": np.array(all_budgets),
        "dates": all_dates,
        "model_type": model_type,
        "loss_type": loss_type,
        "lr": lr,
        "n_steps": n_steps,
    }

    if verbose:
        from evaluate import compute_all_metrics, print_metrics_table
        name = f"e2e-{loss_type} ({model_type})"
        metrics = compute_all_metrics(results["portfolio_returns"], name=name)
        print(f"\n{'='*60}")
        print_metrics_table([metrics])

    return results


if __name__ == "__main__":
    from data_loader import generate_simulated_data

    print("=== Quick smoke test on small simulated data ===")
    returns = generate_simulated_data(n_days=220, seed=42)

    # Very short test: 50-day lookback, 10-day rebalance, 5 steps
    results = run_backtest(
        returns,
        model_type="model_based",
        loss_type="sharpe",
        lr=50.0,
        n_steps=3,
        lookback=50,
        rebalance=10,
        start_idx=60,
        end_idx=80,
        verbose=True,
    )
    print(f"\nPortfolio returns length: {len(results['portfolio_returns'])}")
    print(f"Sample weights: {results['weights'][0].round(4)}")
