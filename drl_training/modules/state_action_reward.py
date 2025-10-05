"""
State, Action, and Reward Components for DRL Trading System

This module implements the core logic for state space representation, 
action space constraints, and multi-component reward function calculation
for the Deep Reinforcement Learning trading system.

Author: DRL Trading Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from .config_manager import ConfigManager


class StateActionReward:
    """
    Core logic for state space, action space, and reward function calculation.
    
    This class implements:
    - 15D state space construction
    - Action space constraints and position shift limits
    - Multi-component reward function (6+ components)
    - Portfolio and risk tracking
    - State normalization and bounds checking
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the state-action-reward calculator.
        
        Args:
            config: ConfigManager instance with all configuration parameters
        """
        self.config = config
        
        # Trading state tracking
        self.position = 0.0
        self.position_change = 0.0
        self.portfolio_value = config.trading.initial_capital
        self.initial_capital = config.trading.initial_capital
        self.peak_nav = 1.0
        self.steps_in_position = 0
        self.inactive_steps = 0
        
        # Return history for Sharpe ratio calculation
        self.return_history = deque(maxlen=config.reward.sharpe_window)
        
        # State space bounds
        self.state_bounds = self._initialize_state_bounds()
        
    def _initialize_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state space bounds for 15D state vector."""
        # Define bounds for each state dimension
        low_bounds = np.array([
            -1.0,   # Position [-1, 1]
            -5.0,   # Z-score (typically [-3, 3] but allow wider range)
            -1.0,   # Zone normalized [-1, 1]
            -0.1,   # Price momentum (clipped to [-0.1, 0.1])
            -2.0,   # Z-score momentum (clipped to [-2, 2])
            -0.2,   # position change
            -1.0,   # MACD (normalized to [-1, 1])
            -1.0,   # MACD Signal (normalized to [-1, 1])
            -1.0,   # MACD Histogram (normalized to [-1, 1])
            -1.0,   # RSI (normalized to [-1, 1])
            -1.0,   # BB Middle (normalized to [-1, 1])
            -1.0,   # BB Upper (normalized to [-1, 1])
            -1.0,   # BB Lower (normalized to [-1, 1])
            -1.0,   # OBV (normalized to [-1, 1])
            -1.0,   # Sentiment score (normalized to [-1, 1])
        ], dtype=np.float32)
        
        high_bounds = np.array([
            1.0,    # Position [1, 1]
            5.0,    # Z-score
            1.0,    # Zone normalized [1, 1]
            0.1,    # Price momentum
            2.0,    # Z-score momentum
            0.2,    # Position change 
            1.0,    # MACD
            1.0,    # MACD Signal
            1.0,    # MACD Histogram
            1.0,    # RSI
            1.0,    # BB Middle
            1.0,    # BB Upper
            1.0,    # BB Lower
            1.0,    # OBV
            1.0,    # Sentiment score
        ], dtype=np.float32)
        
        return low_bounds, high_bounds
    
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state space bounds."""
        return self.state_bounds
    
    def construct_state(self, features: np.ndarray, current_idx: int) -> np.ndarray:
        """
        Construct 15D state representation.
        
        Args:
            features: Feature array (13D: core + technical + sentiment)
            current_idx: Current time index
            
        Returns:
            15D state vector
        """
        # Initialize 15D state vector
        state = np.zeros(15, dtype=np.float32)
        
        # 1. Position (current portfolio exposure)
        state[0] = self.position
        
        # 2-5. Core features (z_score, zone_norm, price_momentum, z_score_momentum)
        state[1:5] = features[:4]
        
        # 6. Position change (change in portfolio exposure)
        state[5] = self.position_change
        
        # 7-14. Technical indicators (8D)
        state[6:14] = features[4:12]
        
        # 15. Sentiment score
        state[14] = features[12]
        
        # Ensure state is within bounds
        low_bounds, high_bounds = self.state_bounds
        state = np.clip(state, low_bounds, high_bounds)
        
        return state
    
    def apply_action_constraints(self, action: float) -> Tuple[float, float, float]:
        """
        Apply action space constraints and position shift limits.
        
        Args:
            action: Raw action value [-1, 1]
            
        Returns:
            Tuple of (target_position, position_change, constrained_action)
        """
        # Clip action to valid range
        action_clipped = np.clip(action, -1.0, 1.0)
        
        # Calculate target position
        target_position = action_clipped
        position_change = target_position - self.position
        
        # Enforce maximum position shift per minute
        max_shift = self.config.action_space.max_position_shift
        if abs(position_change) > max_shift:
            position_change = np.sign(position_change) * max_shift
            target_position = self.position + position_change
            
        # Ensure final position is within limits
        pos_limits = self.config.trading.position_limits
        target_position = np.clip(target_position, pos_limits[0], pos_limits[1])
        position_change = target_position - self.position
        
        return target_position, position_change, action_clipped
    
    def calculate_multi_component_reward(
        self, 
        action: float, 
        prices: np.ndarray,
        features: np.ndarray,
        current_idx: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-component reward as per methodology.
        
        Components:
        1. Primary: Profit-and-Loss (PnL)
        2. Risk-Adjusted: Differential Sharpe Ratio
        3. Transaction Costs Penalty
        4. Drawdown Penalty
        5. Holding Reward/Penalty
        6. Activity Incentives
        7. Sentiment Alignment (optional)
        
        Args:
            action: Target position [-1, 1]
            prices: Price array
            features: Feature array
            current_idx: Current time index
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Validate inputs to prevent None comparison errors
        if action is None or prices is None or len(prices) == 0 or features is None or len(features) == 0 or current_idx is None:
            raise ValueError("Invalid input: action, prices, features, or current_idx is None/empty")
            
        # Apply action constraints
        target_position, position_change, _ = self.apply_action_constraints(action)
        
        # Calculate price change for PnL
        if current_idx + 1 < len(prices):
            current_price = prices[current_idx]
            next_price = prices[current_idx + 1]
            # Ensure price values are valid
            if current_price is None or next_price is None or current_price == 0:
                price_change = 0.0
            else:
                price_change = (next_price - current_price) / current_price
        else:
            price_change = 0.0
        
        # Component 1: Primary Reward - Profit-and-Loss
        pnl_reward = self._calculate_pnl_reward(price_change)
        
        # Component 2: Risk-Adjusted Return (Sharpe Ratio)
        sharpe_reward = self._calculate_sharpe_reward()
        
        # Component 3: Transaction Costs Penalty
        transaction_penalty = self._calculate_transaction_costs(position_change)
        
        # Component 4: Drawdown Penalty
        drawdown_penalty = self._calculate_drawdown_penalty()
        
        # Component 5: Holding Reward/Penalty
        holding_reward = self._calculate_holding_reward(price_change)
        
        # Component 6: Activity Incentives
        activity_reward = self._calculate_activity_reward(position_change)
        
        # Component 7: Sentiment Alignment (optional)
        sentiment_reward = self._calculate_sentiment_reward(
            features, current_idx, target_position, position_change
        )
        
        # Total reward
        total_reward = (
            pnl_reward + sharpe_reward + transaction_penalty + 
            drawdown_penalty + holding_reward + activity_reward + sentiment_reward
        )
        
        # Update internal state
        self._update_internal_state(target_position, position_change, price_change)
        
        # Component breakdown for analysis
        reward_breakdown = {
            'pnl_reward': pnl_reward,
            'sharpe_reward': sharpe_reward,
            'transaction_penalty': transaction_penalty,
            'drawdown_penalty': drawdown_penalty,
            'holding_reward': holding_reward,
            'activity_reward': activity_reward,
            'sentiment_reward': sentiment_reward,
            'total_reward': total_reward,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'nav': self.portfolio_value / self.initial_capital
        }
        
        return total_reward, reward_breakdown
    
    def _calculate_pnl_reward(self, price_change: float) -> float:
        """Calculate profit-and-loss reward component."""
        position_return = self.position * price_change
        portfolio_change = position_return * self.portfolio_value
        
        reward_config = self.config.reward
        if reward_config.pnl_normalization == 'nav':
            pnl_reward = (portfolio_change / self.initial_capital) * reward_config.pnl_scale
        else:
            pnl_reward = portfolio_change * reward_config.pnl_scale
        
        return pnl_reward
    
    def _calculate_sharpe_reward(self) -> float:
        """Calculate risk-adjusted Sharpe ratio reward component."""
        if len(self.return_history) > 1:
            returns_array = np.array(list(self.return_history))
            if len(returns_array) > 1 and np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
                sharpe_reward = sharpe_ratio * self.config.reward.sharpe_weight
            else:
                sharpe_reward = 0.0
        else:
            sharpe_reward = 0.0
        
        return sharpe_reward
    
    def _calculate_transaction_costs(self, position_change: float) -> float:
        """Calculate transaction costs penalty component."""
        reward_config = self.config.reward
        trade_cost = abs(position_change) * (reward_config.fee_rate + reward_config.slippage)
        transaction_penalty = -trade_cost * self.portfolio_value * reward_config.transaction_penalty
        
        return transaction_penalty
    
    def _calculate_drawdown_penalty(self) -> float:
        """Calculate drawdown penalty component."""
        current_nav = self.portfolio_value / self.initial_capital
        reward_config = self.config.reward
        
        if current_nav < (self.peak_nav * (1 - reward_config.drawdown_threshold)):
            drawdown_penalty = -reward_config.drawdown_penalty
        else:
            drawdown_penalty = 0.0
        
        return drawdown_penalty
    
    def _calculate_holding_reward(self, price_change: float) -> float:
        """Calculate holding reward/penalty component."""
        reward_config = self.config.reward
        
        if abs(self.position) > 0.1:  # In position
            position_return = self.position * price_change
            if position_return > 0:
                holding_reward = reward_config.holding_reward
            else:
                holding_reward = -reward_config.holding_penalty
        else:
            holding_reward = 0.0
        
        # Add penalty for holding too long
        if self.steps_in_position > reward_config.max_hold_periods:
            holding_reward -= reward_config.holding_penalty * 2
        
        return holding_reward
    
    def _calculate_activity_reward(self, position_change: float) -> float:
        """Calculate activity incentives reward component."""
        reward_config = self.config.reward
        
        if abs(position_change) > 0.01:  # Taking action
            activity_reward = reward_config.activity_reward
            self.inactive_steps = 0
        else:
            self.inactive_steps += 1
            # Escalating penalty for inactivity
            activity_reward = -reward_config.inactivity_penalty * (1 + self.inactive_steps * 0.1)
        
        return activity_reward
    
    def _calculate_sentiment_reward(
        self, 
        features: np.ndarray, 
        current_idx: int, 
        target_position: float,
        position_change: float
    ) -> float:
        """Calculate sentiment alignment reward component."""
        if not self.config.reward.enable_sentiment_reward:
            return 0.0
        
        sentiment_score = float(features[-1])  # Last feature is sentiment
        weight = self.config.reward.sentiment_reward_weight
        max_shift = self.config.action_space.max_position_shift
        
        if max_shift > 0:
            normalized_position = position_change / max_shift
            sentiment_reward = sentiment_score * weight * normalized_position
        else:
            sentiment_reward = 0.0
        
        return sentiment_reward
    
    def _update_internal_state(
        self, 
        target_position: float, 
        position_change: float, 
        price_change: float
    ) -> None:
        """Update internal tracking variables."""
        # Calculate portfolio change
        position_return = self.position * price_change
        portfolio_change = position_return * self.portfolio_value
        
        # Calculate trade costs
        reward_config = self.config.reward
        trade_cost = abs(position_change) * (reward_config.fee_rate + reward_config.slippage)
        
        # Update position and portfolio
        self.position = target_position
        self.position_change = position_change
        self.portfolio_value += portfolio_change - abs(trade_cost * self.portfolio_value)
        
        # Update return history
        self.return_history.append(position_return)
        
        # Update peak NAV
        current_nav = self.portfolio_value / self.initial_capital
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav
        
        # Update position tracking
        if abs(self.position) > 0.1:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0
    
    def reset_episode_state(self, initial_capital: Optional[float] = None) -> None:
        """
        Reset state for new episode.
        
        Args:
            initial_capital: Starting capital for episode (uses config default if None)
        """
        if initial_capital is None:
            initial_capital = self.config.trading.initial_capital
        
        self.position = 0.0
        self.position_change = 0.0
        self.portfolio_value = initial_capital
        self.initial_capital = initial_capital
        self.peak_nav = 1.0
        self.steps_in_position = 0
        self.inactive_steps = 0
        self.return_history.clear()
    
    def get_state_description(self) -> List[str]:
        """Get description of each state dimension."""
        return [
            "Position: Current portfolio exposure [-1, 1]",
            "Z-score: Standardized mean-reversion signal",
            "Normalized Zone: Discrete trading signal [-1, 1]",
            "Price Momentum: Last-minute price return",
            "Z-score Momentum: Change in Z-score",
            "Position Change: Change in portfolio exposure",
            "MACD Line: MACD indicator (normalized)",
            "MACD Signal: MACD signal line (normalized)",
            "MACD Histogram: MACD histogram (normalized)",
            "RSI: Relative Strength Index (normalized)",
            "BB Middle: Bollinger Bands middle (normalized)",
            "BB Upper: Bollinger Bands upper (normalized)",
            "BB Lower: Bollinger Bands lower (normalized)",
            "OBV: On-Balance Volume (normalized)",
            "Sentiment Score: Reddit sentiment (normalized)"
        ]
    
    def get_reward_description(self) -> Dict[str, str]:
        """Get description of each reward component."""
        return {
            'pnl_reward': 'Primary profit-and-loss reward',
            'sharpe_reward': 'Risk-adjusted Sharpe ratio reward',
            'transaction_penalty': 'Trading cost penalty',
            'drawdown_penalty': 'Maximum drawdown penalty',
            'holding_reward': 'Holding position reward/penalty',
            'activity_reward': 'Action/inaction incentive',
            'sentiment_reward': 'Sentiment alignment reward (optional)',
        }
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current state metrics for monitoring."""
        return {
            'position': self.position,
            'position_change': self.position_change,
            'portfolio_value': self.portfolio_value,
            'nav': self.portfolio_value / self.initial_capital,
            'peak_nav': self.peak_nav,
            'steps_in_position': self.steps_in_position,
            'inactive_steps': self.inactive_steps,
            'return_history_length': len(self.return_history),
            'drawdown': 1 - (self.portfolio_value / self.initial_capital) / self.peak_nav
        }
    
    def validate_action(self, action: float) -> bool:
        """
        Validate if action is within acceptable bounds.
        
        Args:
            action: Action value to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        if not isinstance(action, (int, float)):
            return False
        
        if np.isnan(action) or np.isinf(action):
            return False
        
        # Action should be in reasonable range (will be clipped anyway)
        if abs(action) > 10:  # Allow some tolerance beyond [-1, 1]
            return False
        
        return True
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space configuration information."""
        return {
            'action_space': 'Continuous',
            'action_range': [-1.0, 1.0],
            'max_position_shift': self.config.action_space.max_position_shift,
            'position_limits': self.config.trading.position_limits,
            'constraint_type': 'Position shift per minute',
            'noise_std': self.config.action_space.action_noise_std
        }


if __name__ == "__main__":
    # Example usage
    from .config_manager import ConfigManager
    
    config = ConfigManager()
    sar = StateActionReward(config)
    
    # Print information
    print("State Space Dimensions:", len(sar.get_state_description()))
    print("Reward Components:", len(sar.get_reward_description()))
    print("Action Space Info:", sar.get_action_space_info())
    
    # Example state construction
    dummy_features = np.random.randn(13)  # 13D features
    state = sar.construct_state(dummy_features, 0)
    print(f"State shape: {state.shape}, bounds: {sar.get_state_bounds()}")
    
    # Example reward calculation
    dummy_prices = np.array([100.0, 101.0, 100.5])
    action = 0.5
    reward, breakdown = sar.calculate_multi_component_reward(
        action, dummy_prices, dummy_features, 0
    )
    print(f"Total reward: {reward:.4f}")
    print("Reward breakdown:", {k: f"{v:.4f}" for k, v in breakdown.items()})
