"""
Modular Trading Environment for DRL Trading System

This module provides a refactored, modular trading environment that integrates
with other components like ConfigManager, DataProcessor, and StateActionReward.

Author: DRL Trading Team
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from .config_manager import ConfigManager
from .state_action_reward import StateActionReward


class TradingEnvironment(gym.Env):
    """
    Modular Ethereum Trading Environment implementing 15D state space methodology.
    
    This environment provides:
    - 15-dimensional state space (6 core + 8 technical + 1 sentiment)
    - Multi-component reward function (6+ components)
    - Continuous action space [-1, 1] with position shift constraints
    - Rolling window training support
    - Advanced risk management
    - Modular design using ConfigManager and StateActionReward
    
    State Space (15D):
    1. Position: Current portfolio exposure [-1, 1]
    2. Z-score: Standardized mean-reversion signal
    3. Normalized Zone: Discrete trading signal [-1, 1]
    4. Price Momentum: Last-minute price return
    5. Z-score Momentum: Change in Z-score
    6. Position Change: Change in portfolio exposure
    7-9. MACD: Line, Signal, Histogram (normalized)
    10. RSI: Relative Strength Index (normalized)
    11-13. Bollinger Bands: Mid, High, Low (normalized)
    14. OBV: On-Balance Volume (normalized)
    15. Sentiment: Reddit sentiment score (normalized)
    """
    
    metadata = {'render.modes': []}
    
    def __init__(self,
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 config: ConfigManager,
                 reward_config: Optional[Dict[str, float]] = None,
                 episode_length: Optional[int] = None,
                 lookback_window: Optional[int] = None,
                 initial_capital: Optional[float] = None,
                 random_start: bool = True):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with price and feature data
            feature_columns: List of feature column names (13D)
            config: ConfigManager instance
            reward_config: Optional reward configuration override
            episode_length: Episode length in minutes (uses config default if None)
            lookback_window: State lookback window (uses config default if None)
            initial_capital: Starting capital (uses config default if None)
            random_start: Whether to randomize episode start positions
        """
        super().__init__()
        
        # Store configuration and components
        self.config = config
        self.data = data
        self.feature_columns = feature_columns
        self.random_start = random_start
        
        # Override configuration if provided
        self.episode_length = episode_length or config.trading.episode_length
        self.lookback_window = lookback_window or config.state_space.lookback_window
        self.initial_capital = initial_capital or config.trading.initial_capital
        
        # Initialize state-action-reward calculator
        self.sar_calculator = StateActionReward(config)
        if reward_config:
            # Override reward configuration
            for key, value in reward_config.items():
                if hasattr(self.sar_calculator.config.reward, key):
                    setattr(self.sar_calculator.config.reward, key, value)
        
        # Prepare data
        self._prepare_data()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Episode state
        self.reset()
        
        print(f"🏛️ TradingEnvironment initialized:")
        print(f"   📊 Data shape: {self.data.shape}")
        print(f"   📋 Features: {len(feature_columns)}D")
        print(f"   🎯 State space: 15D")
        print(f"   ⏱️ Episode length: {self.episode_length:,} minutes")
        print(f"   🔄 Random start: {random_start}")
        print(f"   💰 Initial capital: ${self.initial_capital:,.0f}")
    
    def _prepare_data(self) -> None:
        """Prepare and validate data for the environment."""
        # Validate required columns
        timestamp_col = self.config.data.timestamp_col
        price_col = self.config.data.price_col
        
        required_cols = [timestamp_col, price_col] + self.feature_columns
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp and reset index
        self.data = self.data.sort_values(timestamp_col).reset_index(drop=True)
        
        # Extract arrays for fast access
        self.prices = self.data[price_col].values
        self.timestamps = self.data[timestamp_col].values
        self.features = self.data[self.feature_columns].values
        
        # Calculate valid episode start range
        self.min_start_idx = self.lookback_window
        self.max_start_idx = len(self.data) - self.episode_length - 1
        
        if self.max_start_idx <= self.min_start_idx:
            raise ValueError(
                f"Insufficient data for episodes. Need at least "
                f"{self.lookback_window + self.episode_length + 1} rows, "
                f"got {len(self.data)}"
            )
        
        print(f"   📅 Valid episode range: [{self.min_start_idx}, {self.max_start_idx}]")
    
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces."""
        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: 15D state vector
        low_bounds, high_bounds = self.sar_calculator.get_state_bounds()
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32
        )
        
        print(f"   🎮 Action space: {self.action_space}")
        print(f"   👁️ Observation space: {self.observation_space.shape}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Randomize episode start if enabled
        if self.random_start:
            self.current_idx = self.np_random.integers(self.min_start_idx, self.max_start_idx)
        else:
            self.current_idx = self.min_start_idx
        
        # Initialize episode state
        self.step_count = 0
        
        # Reset state-action-reward calculator
        self.sar_calculator.reset_episode_state(self.initial_capital)
        
        # Get initial state
        initial_state = self._get_current_state()
        
        # Episode information
        info = {
            'step_count': self.step_count,
            'current_idx': self.current_idx,
            'episode_progress': 0.0,
            'episode_length': self.episode_length,
            'price': float(self.prices[self.current_idx]),
            'timestamp': self.timestamps[self.current_idx],
            'portfolio_value': self.sar_calculator.portfolio_value,
            'position': self.sar_calculator.position,
            'nav': 1.0
        }
        
        return initial_state, info
    
    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step."""
        # Handle action input
        if isinstance(action, np.ndarray):
            action_value = float(action[0])
        else:
            action_value = float(action)
        
        # Validate action
        if not self.sar_calculator.validate_action(action_value):
            warnings.warn(f"Invalid action received: {action_value}, clipping to [-1, 1]")
            action_value = np.clip(action_value, -1.0, 1.0)
        
        # Calculate reward and update state
        reward, reward_breakdown = self.sar_calculator.calculate_multi_component_reward(
            action_value, self.prices, self.features[self.current_idx], self.current_idx
        )
        
        # Move to next time step
        self.current_idx += 1
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False
        
        # Get next state
        if not terminated:
            next_state = self._get_current_state()
        else:
            next_state = self._get_current_state(use_last_valid=True)
        
        # Get current feature values for info
        feature_idx = min(self.current_idx - 1, len(self.features) - 1)
        current_features = self.features[feature_idx]
        sentiment_val = current_features[-1] if len(current_features) > 0 else 0.0
        
        # Prepare info dictionary
        price_idx = min(self.current_idx - 1, len(self.prices) - 1)
        timestamp_idx = min(self.current_idx - 1, len(self.timestamps) - 1)
        
        info = {
            **reward_breakdown,
            'step_count': self.step_count,
            'current_idx': self.current_idx,
            'episode_progress': self.step_count / self.episode_length,
            'price': float(self.prices[price_idx]),
            'timestamp': self.timestamps[timestamp_idx],
            'sentiment': sentiment_val,
            'terminated_reason': self._get_termination_reason() if terminated else None
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_current_state(self, use_last_valid: bool = False) -> np.ndarray:
        """Get current state representation."""
        if use_last_valid:
            # Use last valid index if episode is terminated
            feature_idx = min(self.current_idx - 1, len(self.features) - 1)
        else:
            feature_idx = self.current_idx
        
        feature_idx = max(0, min(feature_idx, len(self.features) - 1))
        current_features = self.features[feature_idx]
        
        state = self.sar_calculator.construct_state(current_features, feature_idx)
        return state
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Episode length reached
        if self.step_count >= self.episode_length:
            return True
        
        # Data exhausted
        if self.current_idx >= len(self.data) - 1:
            return True
        
        # Stop loss triggered (90% loss)
        if self.sar_calculator.portfolio_value <= 0.1 * self.initial_capital:
            return True
        
        return False
    
    def _get_termination_reason(self) -> str:
        """Get reason for episode termination."""
        if self.step_count >= self.episode_length:
            return "episode_length_reached"
        elif self.current_idx >= len(self.data) - 1:
            return "data_exhausted"
        elif self.sar_calculator.portfolio_value <= 0.1 * self.initial_capital:
            return "stop_loss_triggered"
        else:
            return "unknown"
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (optional implementation)."""
        if mode == 'human':
            metrics = self.sar_calculator.get_current_metrics()
            print(f"Step: {self.step_count}, "
                  f"Position: {metrics['position']:.3f}, "
                  f"Portfolio: ${metrics['portfolio_value']:.2f}, "
                  f"NAV: {metrics['nav']:.4f}")
        return None
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode."""
        metrics = self.sar_calculator.get_current_metrics()
        
        total_return = (metrics['portfolio_value'] - self.initial_capital) / self.initial_capital
        max_drawdown = metrics['drawdown']
        
        # Calculate Sharpe ratio if we have return history
        returns = list(self.sar_calculator.return_history)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe_ratio = 0.0
        
        return {
            'episode_length': self.step_count,
            'final_portfolio_value': metrics['portfolio_value'],
            'total_return': total_return,
            'final_nav': metrics['nav'],
            'peak_nav': metrics['peak_nav'],
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_position': metrics['position'],
            'steps_in_position': metrics['steps_in_position'],
            'inactive_steps': metrics['inactive_steps'],
            'price_start': float(self.prices[self.current_idx - self.step_count]) if self.step_count > 0 else 0.0,
            'price_end': float(self.prices[min(self.current_idx - 1, len(self.prices) - 1)]),
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information."""
        return self.sar_calculator.get_action_space_info()
    
    def get_state_description(self) -> List[str]:
        """Get description of state dimensions."""
        return self.sar_calculator.get_state_description()
    
    def get_reward_description(self) -> Dict[str, str]:
        """Get description of reward components."""
        return self.sar_calculator.get_reward_description()
    
    def set_reward_config(self, reward_config: Dict[str, float]) -> None:
        """
        Update reward configuration.
        
        Args:
            reward_config: Dictionary with reward configuration updates
        """
        for key, value in reward_config.items():
            if hasattr(self.sar_calculator.config.reward, key):
                setattr(self.sar_calculator.config.reward, key, value)
            else:
                print(f"⚠️ Unknown reward config key: {key}")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features used."""
        return {
            'feature_columns': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'data_shape': self.data.shape,
            'price_column': self.config.data.price_col,
            'timestamp_column': self.config.data.timestamp_col,
            'lookback_window': self.lookback_window
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate environment setup and return diagnostics.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'data_valid': True,
            'spaces_valid': True,
            'config_valid': True,
            'issues': []
        }
        
        try:
            # Check data integrity
            if len(self.data) == 0:
                validation_results['data_valid'] = False
                validation_results['issues'].append("Empty dataset")
            
            if np.any(np.isnan(self.prices)):
                validation_results['data_valid'] = False
                validation_results['issues'].append("NaN values in prices")
            
            if np.any(np.isnan(self.features)):
                validation_results['data_valid'] = False
                validation_results['issues'].append("NaN values in features")
            
            # Check action space
            test_action = self.action_space.sample()
            if not self.action_space.contains(test_action):
                validation_results['spaces_valid'] = False
                validation_results['issues'].append("Action space sampling failed")
            
            # Check observation space
            test_state = self._get_current_state()
            if not self.observation_space.contains(test_state):
                validation_results['spaces_valid'] = False
                validation_results['issues'].append("State outside observation space bounds")
            
            # Check configuration
            if not self.config.validate_configuration():
                validation_results['config_valid'] = False
                validation_results['issues'].append("Configuration validation failed")
            
        except Exception as e:
            validation_results['data_valid'] = False
            validation_results['spaces_valid'] = False
            validation_results['config_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        validation_results['overall_valid'] = (
            validation_results['data_valid'] and 
            validation_results['spaces_valid'] and 
            validation_results['config_valid']
        )
        
        return validation_results
    
    def close(self) -> None:
        """Clean up environment resources."""
        # Clean up any resources if needed
        pass


def create_trading_environment(
    data: pd.DataFrame,
    feature_columns: List[str],
    config: Optional[ConfigManager] = None,
    **kwargs
) -> TradingEnvironment:
    """
    Convenience function to create a trading environment.
    
    Args:
        data: DataFrame with price and feature data
        feature_columns: List of feature column names
        config: ConfigManager instance (creates default if None)
        **kwargs: Additional arguments passed to TradingEnvironment
        
    Returns:
        TradingEnvironment instance
    """
    if config is None:
        config = ConfigManager()
    
    return TradingEnvironment(data, feature_columns, config, **kwargs)


if __name__ == "__main__":
    # Example usage
    from .config_manager import ConfigManager
    from .data_processor import DataProcessor
    
    # Create configuration
    config = ConfigManager()
    
    # Load and process data (mock example)
    processor = DataProcessor(config)
    # processed_data, features, splits = processor.run_full_pipeline()
    
    # For this example, create dummy data
    dummy_data = pd.DataFrame({
        'ts': np.arange(10000),
        'close': 100 + np.random.randn(10000).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, 10000)
    })
    
    # Add dummy features
    feature_cols = [
        'z_score', 'zone_norm', 'price_momentum', 'z_score_momentum',
        'macd', 'macd_signal', 'macd_histogram', 'rsi',
        'bb_middle', 'bb_upper', 'bb_lower', 'obv', 'sentiment_score'
    ]
    for col in feature_cols:
        dummy_data[col] = np.random.randn(10000)
    
    # Create environment
    env = create_trading_environment(dummy_data, feature_cols, config)
    
    # Validate environment
    validation = env.validate_environment()
    print(f"Environment validation: {validation}")
    
    # Test episode
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial info: {info}")
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}")
        
        if terminated:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print(f"Episode summary: {summary}")
    
    env.close()
