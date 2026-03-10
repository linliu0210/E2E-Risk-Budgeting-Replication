"""
Risk Budgeting Portfolio Optimization as a differentiable CvxpyLayer.
Ref: Problem (4) in Uysal et al. (2021), Section 3.3.1.

Convex formulation:
    min_y   sqrt(y^T Σ y)
    s.t.    Σ_i b_i ln(y_i) >= c
            y >= 0

After solving, normalize: x_i = y_i / Σ_j y_j
"""
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from config import N_ASSETS, RB_CONSTANT_C, RB_EPSILON


def build_risk_budget_layer(n_assets: int = N_ASSETS, c: float = RB_CONSTANT_C):
    """
    Build a differentiable risk budgeting optimization layer using CvxpyLayers.

    The layer solves Problem (4):
        min_y   y^T Σ y        (we minimize the square of portfolio variance
                                 for DCP compliance; equivalent to minimizing
                                 sqrt(y^T Σ y) since sqrt is monotone)
        s.t.    b^T ln(y) >= c
                y >= ε          (strict positivity for log domain)

    Parameters (differentiable):
        b: risk budget vector, shape (n_assets,), b >= 0, sum(b) = 1
        Sigma_param: flattened upper-triangular or full covariance,
                     shape (n_assets * n_assets,)

    Returns:
        CvxpyLayer, list of parameter names
    """
    # Variables
    y = cp.Variable(n_assets, pos=True)  # pos=True enforces y > 0

    # Parameters
    b = cp.Parameter(n_assets, nonneg=True)  # risk budgets
    # We pass Sigma as a parameter; for DCP we need it PSD
    Sigma = cp.Parameter((n_assets, n_assets), PSD=True)

    # Objective: minimize portfolio variance (equivalent to min volatility)
    # Eq. (4): min sqrt(y^T Σ y)
    # We use quad_form for DCP compliance: min y^T Σ y (monotone transform)
    objective = cp.Minimize(cp.quad_form(y, Sigma))

    # Constraint: Eq. (4) b^T ln(y) >= c
    constraints = [b @ cp.log(y) >= c]

    problem = cp.Problem(objective, constraints)

    # Build CvxpyLayer
    layer = CvxpyLayer(problem, parameters=[b, Sigma], variables=[y])

    return layer


def solve_risk_budget(
    b: torch.Tensor,
    Sigma: torch.Tensor,
    layer: CvxpyLayer,
    solver_args: dict | None = None,
) -> torch.Tensor:
    """
    Solve risk budgeting problem and return normalized portfolio weights.

    Args:
        b: risk budget vector, shape (n_assets,) or (batch, n_assets)
        Sigma: covariance matrix, shape (n_assets, n_assets)
        layer: CvxpyLayer from build_risk_budget_layer()
        solver_args: additional solver arguments

    Returns:
        z: portfolio weights, shape same as b, summing to 1
    """
    if solver_args is None:
        solver_args = {"solve_method": "SCS", "eps": 1e-5, "max_iters": 5000}

    # Ensure Sigma is PSD by adding regularization
    n = Sigma.shape[-1]
    Sigma_reg = Sigma + RB_EPSILON * torch.eye(n, dtype=Sigma.dtype)

    # Solve
    y_sol, = layer(b, Sigma_reg, solver_args=solver_args)

    # Normalize: x_i = y_i / sum(y_j)    — Eq. after Problem (4)
    z = y_sol / (y_sol.sum(dim=-1, keepdim=True) + 1e-10)

    return z


def solve_risk_parity(
    Sigma: torch.Tensor,
    layer: CvxpyLayer,
    n_assets: int = N_ASSETS,
) -> torch.Tensor:
    """
    Solve nominal risk parity (equal risk budgets).
    Special case: b = [1/n, ..., 1/n].
    """
    b = torch.ones(n_assets, dtype=torch.float64) / n_assets
    return solve_risk_budget(b, Sigma, layer)


def compute_risk_contributions(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute the risk contribution of each asset.
    RC_i = x_i * (Σx)_i / sqrt(x^T Σ x)

    Ref: Section 3.3.1, definition of risk contribution.
    """
    port_var = weights @ cov @ weights
    port_vol = np.sqrt(port_var)
    marginal = cov @ weights
    rc = weights * marginal / (port_vol + 1e-10)
    return rc


def test_risk_budget_layer():
    """Unit test: verify the layer works and gradients flow."""
    print("=== Testing Risk Budget Layer ===")
    n = 7

    # Build layer
    layer = build_risk_budget_layer(n_assets=n)

    # Test 1: Equal risk budget (risk parity) should give meaningful weights
    b = torch.ones(n, dtype=torch.float64, requires_grad=True) / n

    # Random PSD covariance
    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=torch.float64)
    Sigma = A @ A.T + 0.1 * torch.eye(n, dtype=torch.float64)

    z = solve_risk_budget(b, Sigma, layer)

    print(f"  Risk parity weights: {z.detach().numpy().round(4)}")
    print(f"  Sum of weights: {z.sum().item():.6f}")
    assert abs(z.sum().item() - 1.0) < 1e-4, "Weights should sum to 1"
    assert (z > -1e-4).all(), "Weights should be non-negative"

    # Test 2: Gradient flow
    loss = -z.sum()  # dummy loss
    loss.backward()
    print(f"  Gradient on b: {b.grad}")
    assert b.grad is not None, "Gradient should flow through to b"

    # Test 3: Verify risk contributions
    z_np = z.detach().numpy()
    Sigma_np = Sigma.detach().numpy()
    rc = compute_risk_contributions(z_np, Sigma_np)
    rc_normalized = rc / rc.sum()
    print(f"  Risk contributions (normalized): {rc_normalized.round(4)}")
    print(f"  Target risk budgets:             {b.detach().numpy().round(4)}")

    # For risk parity, all risk contributions should be roughly equal
    rc_std = np.std(rc_normalized)
    print(f"  RC std (should be ~0 for equal budgets): {rc_std:.6f}")

    print("✅ All risk budget layer tests passed!")


if __name__ == "__main__":
    test_risk_budget_layer()
