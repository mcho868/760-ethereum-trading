"""
Configuration Manager for DRL Trading System

This module provides centralized configuration management for all components
of the Deep Reinforcement Learning trading system.

Author: DRL Trading Team
"""

import os
import json
from typing import Dict, List, Any, Optional
from multiprocessing import cpu_count
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_path: str = "../ETHUSDT_1m_with_indicators.parquet"
    timestamp_col: str = "ts"
    price_col: str = "close"
    output_dir: str = "./processed_data_15d"
    model_dir: str = "./models"
    config_file: str = "./drl_training_configs.json"


@dataclass
class StateSpaceConfig:
    """State space configuration for 15D state representation."""
    # Core Features (6 dimensions)
    ma_period: int = 60  # Moving average period for Z-score baseline
    z_score_window: int = 120  # Rolling window for Z-score calculation (2 hours)
    lookback_window: int = 120  # State representation lookback (2 hours minimum)
    
    # Zone thresholds for normalized zone calculation
    open_threshold: float = 2.0  # Z-score threshold to open positions (±2.0)
    close_threshold: float = 0.5  # Z-score threshold to close positions (±0.5)
    
    # Sentiment Data (1 dimension)
    sentiment_weight: float = 0.15  # Weight for sentiment in Fear & Greed Index


@dataclass
class ActionSpaceConfig:
    """Action space configuration."""
    max_position_shift: float = 0.1  # Maximum position change per minute
    action_noise_std: float = 0.1  # Action noise for exploration


@dataclass
class RewardConfig:
    """Multi-component reward system configuration."""
    # Primary Reward: Profit-and-Loss
    pnl_scale: float = 100.0
    pnl_normalization: str = 'nav'  # 'nav', 'portfolio', 'none'
    
    # Risk-Adjusted Return (Differential Sharpe Ratio)
    sharpe_weight: float = 0.2
    sharpe_window: int = 1440  # Window for Sharpe calculation (24 hours)
    
    # Transaction Costs Penalty
    transaction_penalty: float = 1.0
    fee_rate: float = 0.001  # Trading fee rate (0.1%)
    slippage: float = 0.0005  # Market impact slippage (0.05%)
    
    # Drawdown Penalty
    drawdown_threshold: float = 0.10  # Drawdown threshold (10%)
    drawdown_penalty: float = 50.0
    
    # Holding Reward/Penalty
    holding_penalty: float = 0.001
    holding_reward: float = 0.0005
    max_hold_periods: int = 1440  # Maximum hold duration (24 hours)
    
    # Activity Incentives
    activity_reward: float = 0.1
    inactivity_penalty: float = 0.005
    
    # Sentiment Reward Feedback
    enable_sentiment_reward: bool = True
    sentiment_reward_weight: float = 0.0


@dataclass
class TradingConfig:
    """Trading parameters configuration."""
    initial_capital: float = 10000.0
    episode_length: int = 1440 * 7  # Episode length in minutes (7 days)
    position_limits: tuple = (-1.0, 1.0)


@dataclass
class TrainingProtocolConfig:
    """Training protocol configuration."""
    # Data Splitting Strategy
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Rolling Window Training
    rolling_window_months: int = 6
    evaluation_period_months: int = 1
    rolling_step_months: int = 1
    
    # Combinatorial Purged Cross-Validation (CPCV)
    n_purged_segments: int = 10
    purge_periods: int = 1440  # 24 hours
    embargo_periods: int = 720  # 12 hours
    
    # Episode and State Design
    state_lookback_hours: int = 2


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    a2c_params: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'normalize_advantage': True,
        'use_rms_prop': True,
        'use_sde': False,
    })
    
    td3_params: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'noise_std': 0.1,
        'target_noise': 0.2,
        'noise_clip': 0.5,
        'policy_delay': 2,
    })
    
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'total_timesteps': 200000,
        'eval_freq': 10000,
        'n_eval_episodes': 10,
        'eval_log_path': './logs',
        'save_freq': 25000,
        'verbose': 1,
    })


@dataclass
class BulkTestingConfig:
    """Bulk testing configuration."""
    max_parallel_jobs: int = field(default_factory=lambda: min(cpu_count() - 1, 8))
    config_batch_size: int = 10
    early_stopping_patience: int = 3
    performance_metric: str = 'sharpe_ratio'
    min_evaluation_episodes: int = 5
    results_file: str = 'bulk_results.json'


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""
    normalization_method: str = 'robust'  # 'standard', 'minmax', 'robust'
    feature_selection: bool = True
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    pca_components: Optional[int] = None


@dataclass
class SentimentConfig:
    """Sentiment integration configuration."""
    enabled: bool = True
    data_path: str = '../sentiment_1min_vader_s1_s5.csv'
    aggregation_method: str = 'weighted_mean'
    time_window: int = 60  # minutes
    smoothing_factor: float = 0.1
    sentiment_features: List[str] = field(default_factory=lambda: [
        'reddit_sentiment_score',
        'reddit_compound_score',
        'reddit_post_volume',
        'reddit_comment_sentiment'
    ])


class ConfigManager:
    """
    Centralized configuration manager for the DRL trading system.
    
    This class provides a single point of access for all configuration
    parameters used throughout the trading system. It supports:
    - Loading configurations from files
    - Saving configurations to files
    - Overriding configurations programmatically
    - Validation of configuration parameters
    - Environment-specific configurations
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = 'default'):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON)
            environment: Environment name for environment-specific configs
        """
        self.environment = environment
        self.config_path = config_path
        
        # Initialize default configurations
        self.data = DataConfig()
        self.state_space = StateSpaceConfig()
        self.action_space = ActionSpaceConfig()
        self.reward = RewardConfig()
        self.trading = TradingConfig()
        self.training_protocol = TrainingProtocolConfig()
        self.model = ModelConfig()
        self.bulk_testing = BulkTestingConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        self.sentiment = SentimentConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load environment-specific configuration if available
            if self.environment in config_data:
                config_data = config_data[self.environment]
            
            # Update configurations
            self._update_configs_from_dict(config_data)
            
            print(f"✅ Configuration loaded from: {config_path}")
            
        except FileNotFoundError:
            print(f"⚠️ Configuration file not found: {config_path}")
            print("🔧 Using default configuration")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            print("🔧 Using default configuration")
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("\n🔧 Configuration Summary")
        print("=" * 50)
        print(f"📊 State Space Dimensions: 15D")
        print(f"   - Core Features: 6D (Position, Z-score, Zone, Price Momentum, Z-score Momentum, Position Change)")
        print(f"   - Technical Indicators: 8D (MACD×3, RSI×1, BB×3, OBV×1)")
        print(f"   - Sentiment Data: 1D (Reddit Sentiment)")
        print(f"💰 Trading Configuration:")
        print(f"   - Initial Capital: ${self.trading.initial_capital:,.0f}")
        print(f"   - Episode Length: {self.trading.episode_length:,} minutes")
        print(f"   - Max Position Shift: {self.action_space.max_position_shift} per minute")
        print(f"🎯 Reward Function: Multi-component hybrid (6 components)")
        print(f"📅 Training Protocol: Rolling window ({self.training_protocol.rolling_window_months} months)")
        print(f"💻 Parallel Processing: {self.bulk_testing.max_parallel_jobs} cores")
        print(f"🔄 Sentiment Integration: {'Enabled' if self.sentiment.enabled else 'Disabled'}")
        print(f"📁 Output Directory: {self.data.output_dir}")
        print(f"🔍 Environment: {self.environment}")
        print("=" * 50)
