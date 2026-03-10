"""
Model-free E2E Neural Network for Portfolio Allocation.
Ref: Section 3.4.1, Figure 1 of Uysal et al. (2021).

Architecture:
    Input(n_features) → Linear(hidden_dim) → LeakyReLU(α=0.1)
    → Linear(n_assets) → Softmax → allocation z

"For the model-free approach, we employ a fully feed-forward neural network
 with one hidden layer. ... the output layer consists of n neurons,
 representing the allocation in each asset." — Section 3.4.1
"""
import torch
import torch.nn as nn
from config import N_ASSETS, HIDDEN_DIM, LEAKY_RELU_ALPHA


class ModelFreeNet(nn.Module):
    """
    Model-free end-to-end network.
    NN directly outputs portfolio weights via softmax.
    No optimization layer — purely explicit layers.
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

        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        Sigma: torch.Tensor = None,  # unused, for API compatibility
        solver_args: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: features → portfolio weights directly.

        Args:
            x: input features, shape (n_features,) or (batch, n_features)
            Sigma: unused (kept for API compatibility with model-based)

        Returns:
            z: portfolio weights, shape (n_assets,) or (batch, n_assets)
            b: same as z (no separate risk budget in model-free)
        """
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        z = self.softmax(h)  # directly outputs weights

        return z, z  # second return is dummy for API compatibility


def test_forward():
    """Quick test of model-free forward pass."""
    print("=== Testing ModelFreeNet ===")
    n_features = 77

    model = ModelFreeNet(n_features=n_features)

    torch.manual_seed(42)
    x = torch.randn(n_features)

    z, _ = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Portfolio weights: {z.detach().numpy().round(4)}")
    print(f"  Weights sum: {z.sum().item():.6f}")
    assert abs(z.sum().item() - 1.0) < 1e-5
    assert (z > -1e-5).all()
    print("✅ ModelFreeNet test passed!")


if __name__ == "__main__":
    test_forward()
