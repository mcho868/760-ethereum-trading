# DRL Ethereum Trading Agent - Real-Time Execution Environment

## 1. Overview

This document details the real-time trading environment for the DRL Ethereum trading agent. This system is designed to deploy a pre-trained DRL model to execute trades on the Binance exchange (hardcoded to the Testnet for safety). It serves as the practical application of the models developed in the training environment and is a critical component for the dissertation's results.

The system's entry point is `drl_testing_enrvironment/run_binance_realtime_trading.py`. It integrates real-time market data, on-the-fly feature calculation, and a sophisticated sentiment analysis pipeline to make autonomous trading decisions.

## 2. System Architecture

The real-time system is composed of several key components that work in concert:

- **`run_binance_realtime_trading.py` (Orchestrator)**: The main script that initializes all components, connects to the Binance WebSocket for live market data, and manages the primary trading loop.

- **`realtimeSentimentPipeline.py` (Sentiment Engine)**: A powerful, asynchronous pipeline that performs sentiment analysis. Its workflow is as follows:
    1.  **Fetch**: Collects posts from specified subreddits (e.g., r/ethereum, r/ethtrader) for the previous day.
    2.  **Clean**: Normalizes and cleans the text data.
    3.  **Score**: Scores sentiment using both the VADER lexicon and a sequence of local Large Language Models (LLMs) via an LM Studio server.
    4.  **Aggregate**: Computes a weighted daily average sentiment and expands it into a minute-by-minute timeseries file (`sentiment_1min_vader_s1_s5.csv`) for the feature engine.

- **`online_feature_engine.py` (Feature Engine)**: A stateful engine that calculates the 15-dimensional state vector in real-time. For each new 1-minute candlestick, it updates its internal state (EMAs, rolling means, etc.) and generates the feature set required by the DRL agent. This allows for efficient, O(1) feature updates. Its state is persisted in `state/indicators_ETHUSDT.json`.

- **`model_loader.py` (Model Loader)**: A simple utility to load a trained `stable-baselines3` model (e.g., A2C) and any associated data scaler (e.g., `MinMaxScaler`) from the training phase.

- **`binance_client.py` (Exchange Interface)**: A client that handles all communication with the Binance API. It is used to fetch account balances, get exchange trading rules (e.g., lot size), and place market orders. **It is hardcoded to use the Binance Testnet.**

## 3. Prerequisites

### 3.1. Software & Services

- **Python Packages**: All required packages are listed in `requirements.txt`. Install them with `pip install -r requirements.txt`.
- **LM Studio **: The application must be serving the models specified in the configuration 
"meta-llama-3.1-8b-instruct", 
"google/gemma-2-9b", "qwen2.5-7b-instruct-1m", 
"mistralai/mistral-7b-instruct-v0.3", 
"nous-hermes-2-mistral-7b-dpo" 
and be accessible at the configured URL (default: `http://127.0.0.1:1234/v1`).

### 3.2. API Keys & Credentials

You must configure the following credentials:

1.  **Binance API Keys**: The system requires Binance Testnet API keys. **For security, you should not hardcode your keys.** The provided `binance_client.py` contains placeholder keys. It is strongly recommended to modify it to load keys from environment variables.
    ```
    API_KEY='Your Binance Testnet API key'
    API_SECRET='Your Binance Testnet API secret'
    ```

2.  **Reddit API Credentials**: The sentiment pipeline requires Reddit API credentials to fetch posts. Create a `.env` file in the project root directory with the following content:
    ```
    REDDIT_CLIENT_ID="your_reddit_client_id"
    REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    REDDIT_USER_AGENT="your_custom_user_agent"
    ```

## 4. Setup & Configuration

1.  **Install Dependencies**: Run `pip install -r requirements.txt`.

2.  **Configure API Keys**: Set up your `binance_client.py` and `.env` file as described above.

3.  **Place Trained Model**: Ensure you have a trained model from  `drl_training/ethTradingDrlAgent_modular.ipynb` (e.g., `a2c_0001_final.zip`) 

4.  **Configure Trading**: Edit `drl_testing_enrvironment/trading_config.json` to point to your desired model and sentiment data files.

5.  **Start LM Studio**: If using LLMs for sentiment, start the LM Studio server and ensure the required models are loaded and ready to serve requests.

## 5. How to Run

Execute the main script from the root directory of the project:

```bash
python drl_testing_enrvironment/run_binance_realtime_trading.py
```

- The script will first initialize all components. This includes a check to see if sentiment data is up-to-date. If it detects missing days, it will automatically run the `RealTimeSentimentPipeline` to backfill them.
- It will then connect to the Binance WebSocket and begin the trading loop.
- Press `Ctrl+C` to gracefully shut down the system. On shutdown, it will attempt to flatten any open positions and generate a final performance report.

### Debug Mode

To force the sentiment pipeline to re-run for the previous day, use the `--debug` flag:

```bash
python drl_testing_enrvironment/run_binance_realtime_trading.py --debug
```

## 6. Real-Time Workflow

1.  **Initialization**: The system starts, loads the configuration, and initializes the sentiment pipeline, feature engine, model loader, and Binance client. It syncs exchange rules and the current portfolio state.
2.  **Data Backfilling**: It checks for the latest price and sentiment data, backfilling any missing days to ensure the feature engine is properly warmed up.
3.  **WebSocket Connection**: It connects to the Binance `kline_1m` WebSocket stream for `ETHUSDT`.
4.  **Trading Loop (On New Bar)**: For each 1-minute candlestick received from the stream:
    a. The `OnlineFeatureEngine` ingests the new OHLCV data and updates its internal state to compute a fresh 15D state vector.
    b. The `ModelLoader` feeds this state vector to the trained DRL agent, which deterministically predicts an action (a target position from -1.0 to 1.0).
    c. The system calculates the required trade (BUY or SELL) to move from the current position to the target position, respecting the exchange's lot size and notional value rules.
    d. If a trade is warranted, a market order is placed via the `binance_client`.
    e. The outcome is logged, and the process repeats for the next minute.

## 7. Outputs

The real-time system generates the following logs and reports:

- **Console Output**: Live status updates are printed to the console.
- **System Log (`trading_system.log`)**: A detailed log of all system operations, including connections, status updates, and errors.
- **Trading Reports (`drl_testing_enrvironment/trading_reports/`)**: This directory contains two types of reports generated at the end of each session:
    - A `.log` file with a human-readable summary of every trade decision and a final P&L report.
    - A `.json` file with a structured summary of the session's performance metrics (P&L, return percentage, trade count, etc.).
- **Engine State (`state/indicators_ETHUSDT.json`)**: A JSON file that persists the internal state of the `OnlineFeatureEngine`, allowing for seamless restarts.
