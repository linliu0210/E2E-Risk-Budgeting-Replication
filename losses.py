"""
Differentiable loss functions for E2E training.
Ref: Section 3.5.2 of Uysal et al. (2021).

"Two risk-reward functions are chosen to train neural networks:
 Sharpe ratio and cumulative return" — Section 3.5.2
"""
import torch


def sharpe_loss(
    portfolio_returns: torch.Tensor | list[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable negative Sharpe ratio loss.

    Loss = -mean(R_p) / (std(R_p) + ε)

    This computes the DAILY Sharpe ratio (not annualized).
    Annualization is applied at evaluation time, not during training.

    Args:
        portfolio_returns: tensor or list of daily portfolio returns
        eps: small constant for numerical stability

    Returns:
        scalar loss (negative Sharpe, to be minimized)
    """
    if isinstance(portfolio_returns, list):
        rets = torch.stack(portfolio_returns)
    else:
        rets = portfolio_returns

    mean_ret = rets.mean()
    std_ret = rets.std() + eps
    return -mean_ret / std_ret


def cumulative_return_loss(
    portfolio_returns: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    """
    Differentiable negative cumulative return loss using log-sum trick.

    Mathematically equivalent to -prod(1 + r_s), but uses
    -sum(log(1 + r_s)) which is numerically stable and avoids
    underflow/overflow in the backward pass.

    Clamps returns to avoid log(0) when r_s ≈ -1.

    Args:
        portfolio_returns: tensor or list of daily portfolio returns

    Returns:
        scalar loss (negative log cumulative return, to be minimized)
    """
    if isinstance(portfolio_returns, list):
        rets = torch.stack(portfolio_returns)
    else:
        rets = portfolio_returns

    # log1p(x) = log(1+x), stable for small x
    # clamp to avoid log(0) or log(negative)
    return -torch.sum(torch.log1p(torch.clamp(rets, min=-0.999999)))


def get_loss_fn(loss_type: str):
    """Return the appropriate loss function by name."""
    if loss_type == "sharpe":
        return sharpe_loss
    elif loss_type in ("cumret", "cumulative_return"):
        return cumulative_return_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'sharpe' or 'cumret'.")


# ======================================================================
# Tests
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Test Sharpe loss
    rets = torch.randn(30) * 0.01 + 0.001
    rets.requires_grad_(True)
    loss_s = sharpe_loss(rets)
    loss_s.backward()
    print(f"Sharpe loss: {loss_s.item():.6f}")
    print(f"Gradient norm: {rets.grad.norm().item():.6f}")
    assert not torch.isnan(loss_s)

    # Test cumulative return loss
    rets2 = torch.randn(30) * 0.01 + 0.001
    rets2.requires_grad_(True)
    loss_c = cumulative_return_loss(rets2)
    loss_c.backward()
    print(f"Cumulative return loss: {loss_c.item():.6f}")
    print(f"Gradient norm: {rets2.grad.norm().item():.6f}")
    assert not torch.isnan(loss_c)

    # Test with list input
    rets_list = [torch.tensor(0.01), torch.tensor(-0.005), torch.tensor(0.008)]
    loss_list = sharpe_loss(rets_list)
    print(f"List Sharpe loss: {loss_list.item():.6f}")

    # Test extreme case (all same return → std ≈ 0)
    rets_flat = torch.ones(10) * 0.001
    loss_flat = sharpe_loss(rets_flat)
    print(f"Flat Sharpe loss: {loss_flat.item():.6f} (should be finite)")
    assert not torch.isnan(loss_flat)

    # Test get_loss_fn
    fn = get_loss_fn("sharpe")
    assert fn is sharpe_loss
    fn2 = get_loss_fn("cumret")
    assert fn2 is cumulative_return_loss

    print("✅ All loss tests passed!")
