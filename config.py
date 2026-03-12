"""
Hyperparameter configuration for E2E Risk Budgeting replication.
Ref: Uysal et al. (2021) — Sections 3, 4, 5.

All parameters sourced from original paper; implementation inferences
marked with [IMPL].
"""

# ──────────────────────────────────────────────────────────────────────
# Assets (Section 5, p.17)
# ──────────────────────────────────────────────────────────────────────
ETF_TICKERS = ["VTI", "IWM", "AGG", "LQD", "MUB", "DBC", "GLD"]
N_ASSETS = 7

# ──────────────────────────────────────────────────────────────────────
# Date Ranges (Section 5.1, p.18)
# ──────────────────────────────────────────────────────────────────────
DATA_START = "2010-06-01"          # buffer before in-sample start
DATA_END = "2021-06-30"
IN_SAMPLE_START = "2011-01-03"
TRAIN_END = "2014-12-31"           # grid-search: train period end
VALIDATION_END = "2016-12-31"      # grid-search: validation period end
OUT_SAMPLE_START = "2017-01-03"
OUT_SAMPLE_END = "2021-06-30"

# ──────────────────────────────────────────────────────────────────────
# Feature Engineering (Section 3.4.2 / Algorithm 1, p.12)
# ──────────────────────────────────────────────────────────────────────
PAST_RETURNS_DAYS = 5              # r_{t-1} ... r_{t-5}
AVG_WINDOWS = [10, 20, 30]         # avg return & vol windows
COV_WINDOW = 30                    # sample covariance window (p.18)
FEATURE_WARMUP = max(PAST_RETURNS_DAYS, max(AVG_WINDOWS))  # = 30

# ──────────────────────────────────────────────────────────────────────
# Neural Network Architecture (Section 3.4.1-3.4.2, p.11-12)
# ──────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 32                    # "first hidden layer: 32 neurons"
LEAKY_RELU_ALPHA = 0.1             # LeakyReLU negative slope

# ──────────────────────────────────────────────────────────────────────
# Section 4: Simulation Experiment (p.15-16)
# ──────────────────────────────────────────────────────────────────────
SIM_LOOKBACK = 150                 # training window K
SIM_REBALANCE = 5                  # test window / rebalance freq
SIM_LR = 10.0                     # learning rate η
SIM_N_STEPS = 50                   # training steps per batch
SIM_TEST_DAYS = 175                # pure test period
SIM_FEATURE_WARMUP = FEATURE_WARMUP  # 30 days for feature build
SIM_TOTAL_DAYS = SIM_FEATURE_WARMUP + SIM_LOOKBACK + SIM_TEST_DAYS  # = 355
SIM_N_SEEDS = 100                  # number of simulation seeds

# Paper Section 4.1 — hardcoded mean daily returns (percentage → decimal)
# "The expected daily returns of the assets are ... 0.059, 0.013, −0.011,
#  0.022, 0.056, 0.017, 0.017"
SIM_MU_PAPER = [0.059, 0.013, -0.011, 0.022, 0.056, 0.017, 0.017]

# ──────────────────────────────────────────────────────────────────────
# Section 5: Market Experiment (p.18-22)
# ──────────────────────────────────────────────────────────────────────
MKT_LOOKBACK = 150                 # training window K
MKT_REBALANCE = 25                 # rebalance freq
MKT_LR_SHARPE = 150.0              # best η for Sharpe (Table 2)
MKT_STEPS_SHARPE = 10              # best n for Sharpe
MKT_LR_RETURN = 300.0              # best η for cumulative return (Table 3)
MKT_STEPS_RETURN = 25              # best n for cumulative return
LR_DECAY_FACTOR = 0.9              # "decrease by factor 0.9" (p.18)
LR_DECAY_EVERY = 3                 # "every three steps" (p.18)

# Grid search candidate sets (Table 2/3, p.18)
HP_LR_CANDIDATES = [50, 100, 150, 250, 300, 500]
HP_STEPS_CANDIDATES = [5, 10, 25, 50]

# ──────────────────────────────────────────────────────────────────────
# Risk Budgeting Layer — Problem (4) (Section 3.3.1, p.10)
# ──────────────────────────────────────────────────────────────────────
RB_CONSTANT_C = 1.0                # "c is an arbitrary positive constant"
RB_EPSILON = 1e-6                  # PSD regularization

# ──────────────────────────────────────────────────────────────────────
# Stochastic Gates (Section 3.5 / 5.4, p.13-14, Table 6-7)
# ──────────────────────────────────────────────────────────────────────
GATE_SIGMA = 0.1                   # σ for Gaussian noise (p.14)
GATE_MU_INIT = 0.5                 # initial μ_d (p.14)
GATE_THRESHOLD = 0.5               # test-time threshold (p.14)

# --- Table 6: 7 ETF (Section 5.4.1) ---
GATE_T6_LR = 150.0                # η for NN weights
GATE_T6_MU_LR = 10.0              # η_μ for gate parameters
GATE_T6_STEPS = 10                 # training steps

# --- Table 7: 8 assets (Section 5.4.2) ---
GATE_T7_LR = 750.0                # η for NN weights
GATE_T7_MU_LR = 750.0             # η_μ for gate parameters
GATE_T7_STEPS = 10                 # training steps
GATE_T7_E2E_LR = 500.0            # η for no-gate e2e baseline
GATE_T7_E2E_STEPS = 5             # steps for no-gate baseline

# --- Random Asset for Section 5.4.2 ---
RANDOM_ASSET_MU = -0.0005         # "mean of −0.05%" (p.27)
RANDOM_ASSET_SIGMA = 0.0005       # "standard deviation of 0.05%"
