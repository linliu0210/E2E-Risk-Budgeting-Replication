"""
Model-based E2E Neural Network for Risk Budgeting Portfolio.
Ref: Section 3.4.2, Figure 2 of Uysal et al. (2021).

Architecture:
    Input(n_features) → Linear(hidden_dim) → LeakyReLU(α=0.1)
    → Linear(n_assets) → Softmax → risk_budget b
    → CvxpyLayer(Problem 4) → allocation z
"""
import torch
import torch.nn as nn
from risk_budget_layer import build_risk_budget_layer, solve_risk_budget
from config import N_ASSETS, HIDDEN_DIM, LEAKY_RELU_ALPHA


class ModelBasedNet(nn.Module):
    """
    Model-based end-to-end network.
    NN outputs risk budgets, CvxpyLayer solves for allocation.

    "The second hidden layer has the number of neurons equalling the number
     of assets. We apply a softmax function on the second hidden layer to
     normalize the values and interpret them as the risk budget allocated
     to each of the assets." — Section 3.4.2
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int = N_ASSETS,
        hidden_dim: int = HIDDEN_DIM,
        alpha: float = LEAKY_RELU_ALPHA,
    ):
        super().__init__()
        self.n_assets = n_assets

        # Neural network layers
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        self.softmax = nn.Softmax(dim=-1)

        # Risk budgeting optimization layer (not trainable itself)
        self.rb_layer = build_risk_budget_layer(n_assets=n_assets)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        Sigma: torch.Tensor,
        solver_args: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: features → risk budgets → portfolio weights.

        Args:
            x: input features, shape (n_features,) or (batch, n_features)
            Sigma: covariance matrix, shape (n_assets, n_assets)
            solver_args: CvxpyLayer solver arguments

        Returns:
            z: portfolio weights, shape (n_assets,) or (batch, n_assets)
            b: risk budgets (for inspection/logging)
        """
        # NN forward: features → risk budgets
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        b = self.softmax(h)  # risk budgets ∈ simplex

        # Clamp to avoid numerical issues (very small budgets → log problem)
        b_clamped = torch.clamp(b, min=1e-4)
        b_clamped = b_clamped / b_clamped.sum(dim=-1, keepdim=True)

        # Convert to float64 for CVXPY solver precision
        b_64 = b_clamped.double()
        Sigma_64 = Sigma.double()

        # Solve risk budgeting optimization (Problem 4)
        # Process each sample individually (CvxpyLayer doesn't natively batch)
        if b_64.dim() == 1:
            z = solve_risk_budget(b_64, Sigma_64, self.rb_layer, solver_args)
        else:
            # Batch processing: solve for each sample
            z_list = []
            for i in range(b_64.shape[0]):
                z_i = solve_risk_budget(b_64[i], Sigma_64, self.rb_layer, solver_args)
                z_list.append(z_i)
            z = torch.stack(z_list)

        return z.float(), b

    def get_risk_budgets(self, x: torch.Tensor) -> torch.Tensor:
        """Get risk budgets without solving the optimization (for inspection)."""
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        return self.softmax(h)


def test_forward():
    """Quick test of model-based forward pass."""
    print("=== Testing ModelBasedNet ===")
    n_features = 77  # 7 assets × 11 features

    model = ModelBasedNet(n_features=n_features)

    # Random input
    torch.manual_seed(42)
    x = torch.randn(n_features)

    # Random PSD covariance
    A = torch.randn(N_ASSETS, N_ASSETS)
    Sigma = A @ A.T + 0.1 * torch.eye(N_ASSETS)

    z, b = model(x, Sigma)

    print(f"  Input shape: {x.shape}")
    print(f"  Risk budgets: {b.detach().numpy().round(4)}")
    print(f"  Portfolio weights: {z.detach().numpy().round(4)}")
    print(f"  Weights sum: {z.sum().item():.6f}")
    assert abs(z.sum().item() - 1.0) < 1e-3
    assert (z > -1e-3).all()
    print("✅ ModelBasedNet test passed!")


if __name__ == "__main__":
    test_forward()
