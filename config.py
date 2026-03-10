"""
Hyperparameter configuration for Uysal 2021 E2E Risk Budgeting replication.
Ref: Section 3.5, 4.2, 5.1 of the paper.
"""

# ============================================================
# ETF Universe (Section 5)
# ============================================================
ETF_TICKERS = ["VTI", "IWM", "AGG", "LQD", "MUB", "DBC", "GLD"]
N_ASSETS = len(ETF_TICKERS)

# ============================================================
# Date Ranges
# ============================================================
DATA_START = "2010-06-01"   # extra buffer before 2011
DATA_END = "2021-06-30"
IN_SAMPLE_START = "2011-01-03"
IN_SAMPLE_END = "2016-12-31"
TRAIN_END = "2014-12-31"
VALIDATION_END = "2016-12-31"
OUT_SAMPLE_START = "2017-01-03"
OUT_SAMPLE_END = "2021-06-30"

# ============================================================
# Feature Engineering (Algorithm 1)
# ============================================================
PAST_RETURNS_DAYS = 5           # past 5 daily returns
AVG_WINDOWS = [10, 20, 30]      # past 10, 20, 30 day avg returns & volatilities
COV_WINDOW = 30                 # sample covariance estimation window

# ============================================================
# Neural Network Architecture (Section 3.5.2)
# ============================================================
HIDDEN_DIM = 32                 # neurons in hidden layer
LEAKY_RELU_ALPHA = 0.1          # leaky ReLU parameter α = 0.1

# ============================================================
# Training — Simulation (Section 4.2)
# ============================================================
SIM_LOOKBACK = 150              # rolling window K for training
SIM_REBALANCE = 5               # rebalance every 5 days
SIM_LR = 10.0                   # learning rate (paper says "10" in Section 4.2)
SIM_N_STEPS = 50                # training steps per batch
SIM_DAYS = 175                  # total simulation days
SIM_N_SEEDS = 100               # number of random seeds

# ============================================================
# Training — Market Data (Section 5.1)
# ============================================================
MKT_LOOKBACK = 150              # rolling window K
MKT_REBALANCE = 25              # rebalance every 25 days
MKT_LR_SHARPE = 150.0           # optimal LR for Sharpe-based (Table 2)
MKT_STEPS_SHARPE = 10           # optimal steps for Sharpe-based
MKT_LR_RETURN = 300.0           # optimal LR for return-based (Table 3)
MKT_STEPS_RETURN = 25           # optimal steps for return-based
LR_DECAY_FACTOR = 0.9           # decay lr by 0.9
LR_DECAY_EVERY = 3              # every 3 steps

# ============================================================
# Risk Budgeting Layer (Problem 4)
# ============================================================
RB_CONSTANT_C = 1.0             # arbitrary positive constant c in Problem (4)
RB_EPSILON = 1e-6               # regularization for Σ: Σ + εI

# ============================================================
# Stochastic Gates (Section 6)
# ============================================================
GATE_SIGMA = 0.1                # randomness parameter σ for gates
GATE_MU_INIT = 0.5              # initialize μ with 0.5
GATE_THRESHOLD = 0.5            # threshold for asset inclusion
GATE_LR = 10.0                  # learning rate for gates (Table 6)
