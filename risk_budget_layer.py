"""
Risk Budgeting Portfolio Optimization as a differentiable CvxpyLayer.
Ref: Problem (4) in Uysal et al. (2021), Section 3.3.1.

Convex formulation (p.10):
    min_y   y^T Σ y                (≡ min sqrt(y^T Σ y), monotone)
    s.t.    Σ_i b_i ln(y_i) >= c
            y >= ε

DPP-compliant implementation:
    We pass L (Cholesky factor of Σ) as parameter instead of Σ itself,
    and use cp.sum_squares(L @ y) = ||Ly||^2 = y^T L^T L y = y^T Σ y.
    This avoids the DPP violation from cp.quad_form(y, Sigma_param).

After solving, normalize: z_i = y_i / Σ_j y_j
"""
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from config import N_ASSETS, RB_CONSTANT_C, RB_EPSILON


def build_risk_budget_layer(
    n_assets: int = N_ASSETS,
    c: float = RB_CONSTANT_C,
) -> CvxpyLayer:
    """
    Build a DPP-compliant differentiable risk budgeting CvxpyLayer.

    Parameters (differentiable):
        b: risk budget vector, shape (n_assets,)
        L: lower Cholesky factor of Σ, shape (n_assets, n_assets)
           so that Σ = L L^T

    Returns:
        CvxpyLayer instance
    """
    # Variable
    y = cp.Variable(n_assets, pos=True)

    # Parameters
    b = cp.Parameter(n_assets, nonneg=True)
    L = cp.Parameter((n_assets, n_assets))  # Cholesky factor

    # Objective: min ||L y||^2 = y^T Σ y (DPP compliant!)
    objective = cp.Minimize(cp.sum_squares(L @ y))

    # Constraint: b^T ln(y) >= c
    constraints = [b @ cp.log(y) >= c]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem should be DPP!"

    layer = CvxpyLayer(problem, parameters=[b, L], variables=[y])
    return layer


def sigma_to_cholesky(Sigma: torch.Tensor, epsilon: float = RB_EPSILON) -> torch.Tensor:
    """
    Compute Cholesky factor L from Σ, with regularization.

    Σ_reg = Σ + εI
    L L^T = Σ_reg
    """
    n = Sigma.shape[-1]
    Sigma_reg = Sigma + epsilon * torch.eye(n, dtype=Sigma.dtype)
    try:
        L = torch.linalg.cholesky(Sigma_reg)
    except torch.linalg.LinAlgError:
        # Fallback: add more regularization
        Sigma_reg = Sigma_reg + 1e-4 * torch.eye(n, dtype=Sigma.dtype)
        L = torch.linalg.cholesky(Sigma_reg)
    return L


def solve_risk_budget(
    b: torch.Tensor,
    Sigma: torch.Tensor,
    layer: CvxpyLayer,
    solver_args: dict | None = None,
) -> torch.Tensor:
    """
    Solve risk budgeting problem and return normalized portfolio weights.

    Args:
        b: risk budget vector, shape (n_assets,), float64
        Sigma: covariance matrix, shape (n_assets, n_assets), float64
        layer: CvxpyLayer from build_risk_budget_layer()

    Returns:
        z: normalized weights, shape (n_assets,), summing to 1
    """
    if solver_args is None:
        solver_args = {"solve_method": "SCS", "eps": 1e-5, "max_iters": 5000}

    # Convert Σ to Cholesky factor L
    L = sigma_to_cholesky(Sigma)

    # Solve
    (y_sol,) = layer(b, L, solver_args=solver_args)

    # Normalize: z_i = y_i / sum(y_j)
    z = y_sol / (y_sol.sum(dim=-1, keepdim=True) + 1e-10)

    return z


def solve_risk_parity(
    Sigma: torch.Tensor,
    layer: CvxpyLayer,
    n_assets: int = N_ASSETS,
) -> torch.Tensor:
    """
    Solve nominal risk parity: equal risk budgets b = [1/n, ..., 1/n].
    """
    b = torch.ones(n_assets, dtype=torch.float64) / n_assets
    return solve_risk_budget(b, Sigma, layer)


def compute_risk_contributions(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """
    Compute the risk contribution of each asset.
    RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
    """
    port_var = weights @ cov @ weights
    port_vol = np.sqrt(port_var)
    marginal = cov @ weights
    rc = weights * marginal / (port_vol + 1e-10)
    return rc


# ======================================================================
# Unit Tests
# ======================================================================

def test_risk_budget_layer():
    """Verify the layer works, produces valid weights, and gradients flow."""
    print("=== Testing Risk Budget Layer ===")
    n = 7

    layer = build_risk_budget_layer(n_assets=n)

    # Test 1: Equal risk budget (risk parity)
    # Create b as a leaf tensor (do NOT use operations like /n which make non-leaf)
    b_raw = torch.full((n,), 1.0 / n, dtype=torch.float64, requires_grad=True)

    torch.manual_seed(42)
    A = torch.randn(n, n, dtype=torch.float64)
    Sigma = A @ A.T + 0.1 * torch.eye(n, dtype=torch.float64)

    z = solve_risk_budget(b_raw, Sigma, layer)

    print(f"  Weights: {z.detach().numpy().round(4)}")
    print(f"  Sum: {z.sum().item():.6f}")
    assert abs(z.sum().item() - 1.0) < 1e-4, "Weights should sum to 1"
    assert (z > -1e-4).all(), "Weights should be non-negative"

    # Test 2: Gradient flow (b_raw is a leaf tensor, so .grad will be populated)
    loss = -z[0]  # use single weight, not z.sum() which is ~constant
    loss.backward()
    print(f"  Gradient on b: {b_raw.grad}")
    assert b_raw.grad is not None, "Gradient should flow through to b"

    # Test 3: Risk contributions for equal budgets
    z_np = z.detach().numpy()
    Sigma_np = Sigma.detach().numpy()
    rc = compute_risk_contributions(z_np, Sigma_np)
    rc_normalized = rc / rc.sum()
    print(f"  Normalized RC: {rc_normalized.round(4)}")
    rc_std = np.std(rc_normalized)
    print(f"  RC std (should be ~0): {rc_std:.6f}")

    # Test 4: Different budget vectors
    b2 = torch.tensor([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    z2 = solve_risk_budget(b2, Sigma, layer)
    print(f"  Asymmetric budgets test — weights sum: {z2.sum().item():.6f}")
    assert abs(z2.sum().item() - 1.0) < 1e-4

    print("✅ Risk budget layer tests passed!")


if __name__ == "__main__":
    test_risk_budget_layer()
