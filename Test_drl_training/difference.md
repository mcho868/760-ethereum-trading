# Differences Between drl.ipynb and drl_no_sent.ipynb

## Overview
Both notebooks implement reinforcement learning for Ethereum trading, but with fundamentally different approaches:

- **drl.ipynb**: Uses complex technical indicators with MinMax normalization
- **drl_no_sent.ipynb**: Implements a paper-based approach with simplified pseudo-spread methodology

## Detailed Differences

### Part 1: Data Preprocessing

#### drl.ipynb:
- Uses **32+ complex features**: OHLCV + technical indicators (RSI, MACD, ATR, Bollinger Bands, EMA, etc.)
- **MinMax normalization** with rolling 60-minute windows
- Features include: `open_norm`, `high_norm`, `low_norm`, `close_norm`, `volume_norm`, `RSI_norm`, `MACD_norm`, `BB_*_norm`, etc.
- Exports to `./processed_data_minmax/` directory
- Complex feature engineering with multiple indicator calculations

#### drl_no_sent.ipynb:
- Uses **only 2 simplified features**: `z_score` and `zone`
- **Pseudo-spread calculation** following research paper methodology:
  - Moving average baseline (60 periods)
  - Z-score normalization over 120-period window
  - 5 trading zones based on z-score thresholds
- Exports to `./processed_data_paper/` directory
- Anti-overfitting approach with dramatically reduced feature complexity

### Part 2: Trading Environment

#### drl.ipynb:
- **Complex `MinuteTradingEnv`** with 60×32+ = 1920+ dimensional observation space
- Advanced risk controls: stop-loss, trailing-stop, feature weighting
- Complex reward structure with multiple penalties
- Window-based observations with extensive feature history

#### drl_no_sent.ipynb:
- **Simplified `PaperBasedTradingEnv`** with only 3-dimensional observation space: `[position, z_score, zone]`
- **Three-component reward structure**:
  - Portfolio reward: position × price movement
  - Action reward: rewards for taking appropriate actions in corresponding zones
  - Transaction punishment: reduced penalties to enable actual trading
- No complex risk controls or feature weighting
- Much simpler state representation

### Part 3: Training Configuration

#### drl.ipynb:
- **1,000,000 timesteps** training
- Complex A2C setup with feature weight scheduling
- Dynamic weight adjustment during training
- `FeatureWeightScheduler` callback for runtime weight updates
- High entropy coefficient (0.01) for exploration

#### drl_no_sent.ipynb:
- **50,000 timesteps** training (20x less to prevent overfitting)
- Simplified A2C configuration without complex callbacks
- No dynamic weight adjustment
- Episode randomization for robust training
- Focus on anti-overfitting methodology

### Part 4: Evaluation and Results

#### drl.ipynb:
- Standard evaluation with NAV tracking and position monitoring
- Complex metrics calculation
- Uses same 1920+ dimensional observation space for evaluation

#### drl_no_sent.ipynb:
- **Enhanced evaluation** with comprehensive visualization matching reference style
- **Detailed performance comparison** against original results
- Shows improvements over original poor performance (80% loss)
- Creates enhanced trading plots with portfolio value, drawdown analysis, and position tracking
- Demonstrates actual trading behavior vs position H 0 in original

## Key Philosophical Differences

### Complexity vs Simplicity:
- **drl.ipynb**: Complex feature engineering approach with many technical indicators
- **drl_no_sent.ipynb**: Simplified approach following research paper methodology to prevent overfitting

### Observation Space:
- **drl.ipynb**: 1920+ dimensions (60 timesteps × 32+ features)
- **drl_no_sent.ipynb**: 3 dimensions (position, z_score, zone)

### Training Philosophy:
- **drl.ipynb**: Extensive training (1M timesteps) with complex feature weighting
- **drl_no_sent.ipynb**: Reduced training (50k timesteps) with anti-overfitting focus

### Performance Goals:
- **drl.ipynb**: Focuses on comprehensive technical analysis
- **drl_no_sent.ipynb**: Focuses on avoiding overfitting and achieving actual trading behavior

## File Structure Differences

### drl.ipynb outputs:
- `./processed_data_minmax/train_minmax.csv`
- `./processed_data_minmax/test_minmax.csv`
- `./processed_data_minmax/combined_minmax.csv`
- `a2c_ethusdt_1m_no_sentiment.zip` (model)

### drl_no_sent.ipynb outputs:
- `./processed_data_paper/train_paper.csv`
- `./processed_data_paper/test_paper.csv`
- `./processed_data_paper/combined_paper.csv`
- `paper_based_a2c_model.zip` (model)
- `paper_based_performance_analysis.png` (visualization)

## Summary

The notebooks represent two contrasting approaches to the same problem:
1. **Complex engineering** (drl.ipynb) vs **Simplified methodology** (drl_no_sent.ipynb)
2. **High-dimensional features** vs **Low-dimensional signals**  
3. **Extensive training** vs **Anti-overfitting training**
4. **Technical analysis focus** vs **Research paper methodology focus**

The paper-based approach (drl_no_sent.ipynb) explicitly aims to address overfitting issues and achieve actual trading behavior rather than staying at position H 0.