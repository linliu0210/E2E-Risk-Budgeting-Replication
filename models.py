"""
Neural network models for E2E Risk Budgeting.
Ref: Section 3.4.1 (model-free), 3.4.2 (model-based), 3.5 (stochastic gates).

Three model classes:
  1. ModelFreeNet:       features → z (direct allocation via softmax)
  2. ModelBasedNet:      features → b (risk budget) → CvxpyLayer → z
  3. StochasticGateNet:  features → b → Stochastic Gate → CvxpyLayer → z
"""
import torch
import torch.nn as nn
from risk_budget_layer import build_risk_budget_layer, solve_risk_budget
from config import N_ASSETS, HIDDEN_DIM, LEAKY_RELU_ALPHA, GATE_SIGMA, GATE_MU_INIT, GATE_THRESHOLD


# ======================================================================
# ModelFreeNet (Section 3.4.1)
# ======================================================================

class ModelFreeNet(nn.Module):
    """
    Model-free end-to-end network.
    NN directly outputs portfolio weights via softmax.

    Architecture (p.11):
        Input(n_features) → Linear(hidden_dim) → LeakyReLU(α)
        → Linear(n_assets) → Softmax → z
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
        Sigma: torch.Tensor = None,
        solver_args: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: features → portfolio weights directly.

        Returns:
            z: portfolio weights
            b: same as z (no separate risk budget in model-free)
        """
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        z = self.softmax(h)
        return z, z  # second is dummy for API compatibility


# ======================================================================
# ModelBasedNet (Section 3.4.2)
# ======================================================================

class ModelBasedNet(nn.Module):
    """
    Model-based end-to-end network.
    NN outputs risk budgets b, CvxpyLayer solves for allocation z.

    Architecture (p.12):
        Input(n_features) → Hidden Layer 1 (hidden_dim, LeakyReLU)
        → Hidden Layer 2 (n_assets, Softmax) → risk budget b
        → CvxpyLayer(Problem 4) → allocation z

    "The second hidden layer has the number of neurons equalling the number
     of assets. We apply a softmax function on the second hidden layer to
     normalize the values and interpret them as the risk budget." — p.12
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
        self.rb_layer = build_risk_budget_layer(n_assets=n_assets)
        self._init_weights()

    def _init_weights(self):
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
        Forward pass: features → risk budgets → CvxpyLayer → weights.

        Args:
            x: features, shape (n_features,)
            Sigma: covariance, shape (n_assets, n_assets), float64
            solver_args: CvxpyLayer solver arguments

        Returns:
            z: portfolio weights (float32)
            b: risk budgets (for logging)
        """
        # NN forward
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        b = self.softmax(h)

        # Clamp for numerical stability in log constraint
        b_clamped = torch.clamp(b, min=1e-4)
        b_clamped = b_clamped / b_clamped.sum(dim=-1, keepdim=True)

        # Convert to float64 for solver
        b_64 = b_clamped.double()
        Sigma_64 = Sigma.double()

        # Solve Problem (4)
        z = solve_risk_budget(b_64, Sigma_64, self.rb_layer, solver_args)

        return z.float(), b


# ======================================================================
# StochasticGateNet (Section 3.5 / 5.4)
# ======================================================================

class StochasticGateNet(nn.Module):
    """
    Model-based network with Stochastic Gates for asset selection.

    Paper-specified (p.13-14):
      - μ initialized to 0.5
      - σ = 0.1 (Gaussian noise)
      - Test threshold = 0.5
      - No penalty term (unlike Yamada et al.)

    [IMPL] Implementation assumptions:
      - Training: soft gate via clamp(μ+ε, 0, 1), always use full Σ
      - Testing: hard gate (μ >= 0.5), optional Σ subsetting
      - μ has independent learning rate η_μ

    CRITICAL: During training, we NEVER subset Σ. This ensures
    gradient flow to ALL gate parameters μ_d, preventing "dead neurons"
    where a gate drops below 0.5 due to noise and can never recover.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int = N_ASSETS,
        hidden_dim: int = HIDDEN_DIM,
        alpha: float = LEAKY_RELU_ALPHA,
        gate_sigma: float = GATE_SIGMA,
        gate_mu_init: float = GATE_MU_INIT,
        gate_threshold: float = GATE_THRESHOLD,
        with_filter: bool = True,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.gate_sigma = gate_sigma
        self.gate_threshold = gate_threshold
        self.with_filter = with_filter

        # NN layers (same as ModelBasedNet)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        self.softmax = nn.Softmax(dim=-1)

        # Gate parameter μ — independent from NN weights
        self.gate_mu = nn.Parameter(
            torch.full((n_assets,), gate_mu_init)
        )

        # CvxpyLayer for full-size problem (used in training always)
        self.rb_layer_full = build_risk_budget_layer(n_assets=n_assets)

        # Cache for subset layers (test-time use only)
        self._rb_layer_cache: dict[int, object] = {}

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _get_subset_layer(self, n_selected: int):
        """Get or create a CvxpyLayer for n_selected assets (test only)."""
        if n_selected not in self._rb_layer_cache:
            self._rb_layer_cache[n_selected] = build_risk_budget_layer(
                n_assets=n_selected
            )
        return self._rb_layer_cache[n_selected]

    def forward(
        self,
        x: torch.Tensor,
        Sigma: torch.Tensor,
        solver_args: dict | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with stochastic gates.

        CRITICAL DESIGN:
        - Training: soft gate, ALWAYS use full Σ → ensures gradient to all μ_d
        - Testing: hard gate, optionally subset Σ (with_filter mode)

        Args:
            x: features, shape (n_features,)
            Sigma: covariance, shape (n_assets, n_assets), float64
            training: override self.training if provided

        Returns:
            z: portfolio weights (float32)
            b: risk budgets (pre-gate, for logging)
        """
        is_training = training if training is not None else self.training

        # NN forward → risk budgets
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        b = self.softmax(h)

        if is_training:
            # ── TRAINING: soft gate, full Σ ──────────────────────────
            eps = torch.randn_like(self.gate_mu) * self.gate_sigma
            gate = torch.clamp(self.gate_mu + eps, 0.0, 1.0)

            b_gated = b * gate
            b_gated_sum = b_gated.sum()
            if b_gated_sum < 1e-6:
                # Fallback: if all gates nearly zero, use raw b
                b_gated = b
                b_gated_sum = b_gated.sum()
            b_gated = b_gated / b_gated_sum

            # Clamp for log stability
            b_gated = torch.clamp(b_gated, min=1e-4)
            b_gated = b_gated / b_gated.sum()

            b_64 = b_gated.double()
            Sigma_64 = Sigma.double()

            # ALWAYS use full Σ during training — no subsetting!
            z = solve_risk_budget(b_64, Sigma_64, self.rb_layer_full, solver_args)
            return z.float(), b

        else:
            # ── TESTING: hard gate ───────────────────────────────────
            gate = (self.gate_mu >= self.gate_threshold).float()
            selected = (self.gate_mu >= self.gate_threshold)

            # Safety: if all filtered out, return equal weight
            if not selected.any():
                return torch.ones(self.n_assets) / self.n_assets, b

            if self.with_filter:
                # Subset Σ and solve on selected assets only
                Sigma_sub = Sigma[selected][:, selected].double()
                b_sub = (b * gate)[selected]
                b_sub = torch.clamp(b_sub, min=1e-4)
                b_sub = b_sub / b_sub.sum()
                b_sub_64 = b_sub.double()

                n_sel = selected.sum().item()
                layer_sub = self._get_subset_layer(n_sel)

                y_sub = solve_risk_budget(b_sub_64, Sigma_sub, layer_sub, solver_args)

                # Map back to full allocation
                z = torch.zeros(self.n_assets)
                z[selected] = y_sub.float() / y_sub.sum().float()
                return z, b

            else:
                # No filter: use full Σ with gated b
                b_gated = b * gate
                b_gated = torch.clamp(b_gated, min=1e-4)
                b_gated = b_gated / b_gated.sum()
                b_64 = b_gated.double()
                Sigma_64 = Sigma.double()

                z = solve_risk_budget(b_64, Sigma_64, self.rb_layer_full, solver_args)
                return z.float(), b

    def get_gate_status(self) -> dict:
        """Return gate parameters and which assets are selected."""
        with torch.no_grad():
            mu = self.gate_mu.numpy()
            selected = mu >= self.gate_threshold
        return {
            "mu": mu,
            "selected": selected,
            "n_selected": selected.sum(),
        }


# ======================================================================
# Tests
# ======================================================================

def test_all_models():
    """Test all three model types."""
    from features import get_n_features

    n_features = get_n_features()
    n = N_ASSETS
    torch.manual_seed(42)

    # Random PSD covariance
    A = torch.randn(n, n, dtype=torch.float64)
    Sigma = A @ A.T + 0.1 * torch.eye(n, dtype=torch.float64)

    x = torch.randn(n_features)

    # --- Test ModelFreeNet ---
    print("=== Testing ModelFreeNet ===")
    mf = ModelFreeNet(n_features)
    z_mf, _ = mf(x)
    print(f"  Weights: {z_mf.detach().numpy().round(4)}")
    print(f"  Sum: {z_mf.sum().item():.6f}")
    assert abs(z_mf.sum().item() - 1.0) < 1e-5
    print("  ✅ ModelFreeNet OK")

    # --- Test ModelBasedNet ---
    print("\n=== Testing ModelBasedNet ===")
    mb = ModelBasedNet(n_features)
    z_mb, b_mb = mb(x, Sigma)
    print(f"  Budgets: {b_mb.detach().numpy().round(4)}")
    print(f"  Weights: {z_mb.detach().numpy().round(4)}")
    print(f"  Sum: {z_mb.sum().item():.6f}")
    assert abs(z_mb.sum().item() - 1.0) < 1e-3
    print("  ✅ ModelBasedNet OK")

    # --- Test StochasticGateNet (training mode) ---
    print("\n=== Testing StochasticGateNet (training) ===")
    sg = StochasticGateNet(n_features, with_filter=True)
    sg.train()
    z_sg, b_sg = sg(x, Sigma, training=True)
    print(f"  Weights: {z_sg.detach().numpy().round(4)}")
    print(f"  Sum: {z_sg.sum().item():.6f}")
    assert abs(z_sg.sum().item() - 1.0) < 1e-3

    # Gradient flow test: use -z[0] (not -z.sum() which is ~constant)
    loss = -z_sg[0]
    loss.backward()
    print(f"  gate_mu grad: {sg.gate_mu.grad}")
    assert sg.gate_mu.grad is not None, "Gate mu gradient should flow!"
    print("  ✅ StochasticGateNet (training) OK")

    # --- Test StochasticGateNet (eval mode, with_filter) ---
    print("\n=== Testing StochasticGateNet (eval, with_filter) ===")
    sg.eval()
    z_eval, _ = sg(x, Sigma, training=False)
    gate_info = sg.get_gate_status()
    print(f"  Gate μ: {gate_info['mu'].round(3)}")
    print(f"  Selected: {gate_info['selected']} ({gate_info['n_selected']} assets)")
    print(f"  Weights: {z_eval.detach().numpy().round(4)}")
    print("  ✅ StochasticGateNet (eval) OK")

    # --- Test StochasticGateNet (no_filter mode) ---
    print("\n=== Testing StochasticGateNet (no_filter) ===")
    sg_nf = StochasticGateNet(n_features, with_filter=False)
    sg_nf.eval()
    z_nf, _ = sg_nf(x, Sigma, training=False)
    print(f"  Weights: {z_nf.detach().numpy().round(4)}")
    print(f"  Sum: {z_nf.sum().item():.6f}")
    print("  ✅ StochasticGateNet (no_filter) OK")

    print("\n✅ All model tests passed!")


if __name__ == "__main__":
    test_all_models()
