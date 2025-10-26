# DRL Ethereum Trading Agent - Training Environment

## 1. Overview

This document describes the training environment for a Deep Reinforcement Learning (DRL) agent designed to trade Ethereum (ETH/USDT). The system is built with a modular architecture to facilitate experimentation and reproducibility, as required for dissertation research.

The core of the project is the Jupyter Notebook `drl_training/ethTradingDrlAgent_modular.ipynb`, which orchestrates the entire training and evaluation pipeline. The agent learns a trading strategy based on a 15-dimensional state space that includes price-derived features, technical indicators, and real-time Reddit sentiment.

## 2. System Architecture

The training pipeline is composed of several independent modules located in `drl_training/modules/`. This design promotes separation of concerns and maintainability.

- **`ConfigManager`**: Centralizes all system configurations, from data paths to model hyperparameters.
- **`DataProcessor`**: Handles loading of raw data, comprehensive feature engineering for the 15D state space, and temporal data splitting (train/validation/test).
- **`StateActionReward`**: Defines the environment's core logic, including the 15D state space, the continuous action space (target portfolio allocation), and the multi-component reward function.
- **`TradingEnvironment`**: A custom `gymnasium` environment that simulates the trading process, using the logic from `StateActionReward`.
- **`ModelTrainer`**: Manages the training of DRL agents (A2C and TD3 models are supported) using the `stable-baselines3` library.
- **`HyperparameterOptimizer`**: Implements a sophisticated hyperparameter search strategy. It uses a rapid rolling-window diagnostic to cull unstable configurations before running a full, in-depth optimization on the most promising candidates.
- **`RollingWindowTrainer`**: Manages the rolling-window training and validation process, crucial for assessing model stability and performance over time.
- **`PerformanceAnalyzer`**: Conducts in-depth performance analysis of trained models, generating metrics (Sharpe ratio, drawdown, etc.) and visualizations.

## 3. Methodology

### State Space (15-Dimensional)
The agent's observations are based on a rich 15D state vector:
- **Core (6D)**: `position`, `position_change`, `z_score`, `zone_norm`, `price_momentum`, `z_score_momentum`.
- **Technical Indicators (8D)**: `MACD` (line, signal, histogram), `RSI`, `Bollinger Bands` (mid, high, low), and `On-Balance Volume (OBV)`.
- **Sentiment (1D)**: An aggregated sentiment score derived from Reddit posts.

### Action Space
The agent outputs a single continuous value in `[-1.0, 1.0]`, representing the desired target position in the portfolio (e.g., `1.0` for fully long, `-1.0` for fully short). The environment constrains the rate of change to prevent unrealistic portfolio shifts.

### Reward Function
The reward is a hybrid function designed to balance profitability and risk, composed of:
1.  Profit-and-Loss (PnL)
2.  Risk-Adjusted Return (Differential Sharpe Ratio)
3.  Transaction Cost Penalty
4.  Drawdown Penalty
5.  Position Holding Reward/Penalty
6.  Trading Activity Incentive

## 4. Prerequisites

Ensure the following packages are installed. You can install them using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Key packages include: `pandas`, `numpy`, `torch`, `gymnasium`, `stable-baselines3`, `matplotlib`, `seaborn`, `scipy`, `tqdm`, and `ta`.

## 5. Setup & Data

1.  **Price Data**: Place the historical 1-minute ETH/USDT data in the root directory of the project. The default expected file is `ETHUSDT_1m_with_indicators.parquet`. This path can be changed in `drl_training/modules/config_manager.py`.

2.  **Sentiment Data**: Place the aggregated 1-minute sentiment data in the root directory. The default expected file is `sentiment_1min_vader_s1_s5.csv`. This path can also be configured in the `ConfigManager`.

3.  **Hyperparameters**: The configurations for hyperparameter optimization are stored in `drl_training/drl_training_configs.json`. You can define different sets of hyperparameters to be tested by the `HyperparameterOptimizer`.

## 6. How to Run

The entire training and evaluation process is managed within the `drl_training/ethTradingDrlAgent_modular.ipynb` Jupyter Notebook.

1.  Launch Jupyter Notebook or Jupyter Lab from the project's root directory.
2.  Open `drl_training/ethTradingDrlAgent_modular.ipynb`.
3.  Execute the cells sequentially.

The notebook is structured into the following major steps:
- **Step 1: Data Loading & Feature Engineering**: Loads raw data and generates the 15D feature set.
- **Step 2: Temporal Data Splitting**: Splits the data into training, validation, and test sets chronologically.
- **Step 3: Full Training with Hyperparameter Optimization**: Runs the two-phase optimization process to find the best model configuration.
- **Step 4: Final Testing & Comprehensive Analysis**: Takes the best-performing model, retrains it on the full training set, and evaluates it against the held-out test set, generating final performance plots and metrics.

## 7. Outputs

The training pipeline generates several artifacts, primarily within the `drl_training/` directory:

- **Processed Data**: The fully-featured dataset and temporal splits are saved in `drl_training/processed_data_15d/`.
- **Trained Models**: The best-performing models are saved in the `drl_training/models/` directory.
- **Optimization Results**: Logs and results from the hyperparameter search are stored in `drl_training/processed_data_15d/optimization_results/`.
- **Analysis Plots**: Final performance charts and visualizations are saved to `drl_training/processed_data_15d/analysis_plots/`.
- **Logs**: Training logs, including evaluation metrics, are stored in `drl_training/logs/`.
