# Uysal 2021 — E2E Risk Budgeting 代码复现

## 快速开始（服务器部署）

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 单元测试 — 验证 CvxpyLayers 工作正常
python risk_budget_layer.py

# 4. 模型前向传播测试
python model_based_net.py
python model_free_net.py

# 5. Smoke test — 小规模仿真（~2分钟）
python run_simulation.py --seeds 1 --loss sharpe

# 6. 完整仿真实验（~数小时，论文 Section 4）
python run_simulation.py --seeds 100 --loss sharpe
python run_simulation.py --seeds 100 --loss cumret

# 7. 真实市场数据实验（论文 Section 5）
python run_market.py --loss sharpe --period out_sample --verbose
python run_market.py --loss sharpe --period in_sample --verbose
```

## 项目结构

| 文件 | 对应论文 | 说明 |
|------|---------|------|
| `config.py` | §3.5, §4.2, §5.1 | 全部超参数 |
| `data_loader.py` | §4.1, §5 | yfinance下载 + 仿真数据 |
| `features.py` | Algorithm 1 | 特征工程 + 协方差估计 |
| `risk_budget_layer.py` | Problem (4) | CvxpyLayers 风险预算优化层 |
| `model_based_net.py` | Figure 2, §3.4.2 | NN→softmax→CvxpyLayer |
| `model_free_net.py` | Figure 1, §3.4.1 | NN→softmax→直接输出权重 |
| `train.py` | Algorithm 1 | Rolling window 训练循环 |
| `evaluate.py` | Table 1,4,5 | 评估指标 + Loss函数 |
| `baselines.py` | §5.2, §6.1 | Nominal RP, 1/N |
| `run_simulation.py` | Section 4 | 仿真实验 |
| `run_market.py` | Section 5 | 真实数据实验 |

## 建议运行顺序

1. 先跑 **单元测试**（步骤 3-4），确认环境没问题
2. 再跑 **1 seed smoke test**（步骤 5），确认 pipeline 跑通
3. 最后跑 **完整实验**（步骤 6-7）

## 预估运行时间（单核 CPU）

| 实验 | 预估时间 |
|------|---------|
| 单元测试 | ~30秒 |
| 1 seed 仿真 | ~5-10分钟 |
| 100 seeds 仿真 | ~8-15小时 |
| 市场数据 out-sample | ~2-4小时 |
