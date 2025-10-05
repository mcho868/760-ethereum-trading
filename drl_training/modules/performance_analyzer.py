"""
Performance Analyzer for DRL Trading System

This module provides comprehensive performance analysis and visualization
capabilities for Deep Reinforcement Learning trading models.

Author: DRL Trading Team
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Set up plotting style (with error handling)
try:
    plt.style.use('default')
    sns.set_palette("husl")
except Exception as e:
    print(f"⚠️ Warning: Could not set plotting style: {e}")
    
from .config_manager import ConfigManager
from .trading_environment import TradingEnvironment


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and visualization for DRL trading models.
    
    Features:
    - Portfolio performance metrics calculation
    - Risk analysis (Sharpe ratio, drawdown, VaR, volatility)
    - Trading behavior analysis and statistics
    - Reward component breakdown and analysis
    - State space analysis and feature importance
    - Model comparison dashboards
    - Advanced visualizations and plots
    - Export capabilities for reports
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize PerformanceAnalyzer.
        
        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.results_cache = {}
        self.plot_style = {
            'figsize': (15, 10),
            'dpi': 100,
            'style': 'seaborn-v0_8',
            'color_palette': 'husl'
        }
        
        # Create output directory for plots
        self.output_dir = Path(config.data.output_dir) / 'analysis_plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ PerformanceAnalyzer initialized")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def calculate_comprehensive_metrics(self,
                                      portfolio_values: np.ndarray,
                                      positions: np.ndarray,
                                      rewards: np.ndarray,
                                      prices: Optional[np.ndarray] = None,
                                      timestamps: Optional[np.ndarray] = None,
                                      initial_capital: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive trading performance metrics.
        
        Args:
            portfolio_values: Array of portfolio values over time
            positions: Array of position values over time
            rewards: Array of reward values over time
            prices: Array of price values over time (optional)
            timestamps: Array of timestamps (optional)
            initial_capital: Initial portfolio value (uses config if None)
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio_values) == 0:
            return {}
        
        if initial_capital is None:
            initial_capital = self.config.trading.initial_capital
        
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values)
        positions = np.array(positions)
        rewards = np.array(rewards)
        
        # Basic metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # NAV calculation
        nav_values = portfolio_values / initial_capital
        
        # Returns calculation
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]  # Remove inf/nan values
        
        # Risk metrics
        if len(returns) > 1:
            return_volatility = np.std(returns)
            sharpe_ratio = np.mean(returns) / (return_volatility + 1e-8)
            # Annualized metrics (assuming daily returns)
            annualized_return = np.mean(returns) * 252
            annualized_volatility = return_volatility * np.sqrt(252)
            annualized_sharpe = annualized_return / (annualized_volatility + 1e-8)
        else:
            return_volatility = 0.0
            sharpe_ratio = 0.0
            annualized_return = 0.0
            annualized_volatility = 0.0
            annualized_sharpe = 0.0
        
        # Drawdown calculation
        peak_values = np.maximum.accumulate(nav_values)
        drawdowns = (nav_values - peak_values) / peak_values
        max_drawdown = np.min(drawdowns)
        
        # Value at Risk (95% confidence)
        if len(returns) > 10:
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        # Trading statistics
        position_changes = np.diff(positions)
        n_trades = np.sum(np.abs(position_changes) > 0.01)
        avg_position = np.mean(np.abs(positions))
        
        # Win rate calculation
        if len(returns) > 0:
            winning_periods = np.sum(returns > 0)
            win_rate = winning_periods / len(returns)
        else:
            win_rate = 0.0
        
        # Reward statistics
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)
        reward_volatility = np.std(rewards)
        
        # Calmar ratio (annualized return / max drawdown)
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-8)
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_deviation = np.std(negative_returns)
            sortino_ratio = np.mean(returns) / (downside_deviation + 1e-8)
        else:
            sortino_ratio = 0.0
        
        metrics = {
            # Basic performance
            'total_return': float(total_return),
            'final_portfolio_value': float(final_value),
            'final_nav': float(nav_values[-1]),
            
            # Risk metrics
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'return_volatility': float(return_volatility),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            
            # Annualized metrics
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(annualized_volatility),
            'annualized_sharpe': float(annualized_sharpe),
            
            # Trading statistics
            'win_rate': float(win_rate),
            'n_trades': int(n_trades),
            'avg_position': float(avg_position),
            
            # Reward statistics
            'total_reward': float(total_reward),
            'avg_reward': float(avg_reward),
            'reward_volatility': float(reward_volatility),
            
            # Episode statistics
            'n_periods': len(portfolio_values),
            'initial_capital': float(initial_capital)
        }
        
        return metrics
    
    def analyze_model_performance(self,
                                model,
                                env: TradingEnvironment,
                                n_episodes: int = 5,
                                config_info: Optional[Dict[str, Any]] = None,
                                deterministic: bool = True) -> Dict[str, Any]:
        """
        Analyze model performance over multiple episodes.
        
        Args:
            model: Trained model to analyze
            env: Trading environment
            n_episodes: Number of episodes to run
            config_info: Additional configuration information
            deterministic: Whether to use deterministic actions
            
        Returns:
            Comprehensive analysis results
        """
        print(f"🔍 Analyzing model performance over {n_episodes} episodes...")
        
        episode_data = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            print(f"   📊 Running episode {episode + 1}/{n_episodes}")
            
            # Run episode
            episode_results = self._run_analysis_episode(model, env, deterministic)
            
            if episode_results:
                episode_data.append(episode_results)
                
                # Calculate metrics for this episode
                metrics = self.calculate_comprehensive_metrics(
                    episode_results['portfolio_values'],
                    episode_results['positions'],
                    episode_results['rewards'],
                    episode_results.get('prices'),
                    episode_results.get('timestamps')
                )
                
                metrics['episode'] = episode + 1
                episode_metrics.append(metrics)
        
        # Aggregate metrics across episodes
        aggregate_metrics = self._aggregate_episode_metrics(episode_metrics)
        
        # Compile analysis results
        analysis_results = {
            'episode_data': episode_data,
            'episode_metrics': episode_metrics,
            'aggregate_metrics': aggregate_metrics,
            'config_info': config_info or {},
            'n_episodes': n_episodes,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        if episode_metrics:
            avg_return = aggregate_metrics.get('mean_total_return', 0)
            avg_sharpe = aggregate_metrics.get('mean_sharpe_ratio', 0)
            avg_drawdown = aggregate_metrics.get('mean_max_drawdown', 0)
            
            print(f"   ✅ Analysis complete:")
            print(f"      📈 Average Total Return: {avg_return:.2%}")
            print(f"      📊 Average Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"      📉 Average Max Drawdown: {avg_drawdown:.2%}")
        
        return analysis_results
    
    def _run_analysis_episode(self, model, env: TradingEnvironment, deterministic: bool) -> Dict[str, Any]:
        """Run a single episode for analysis."""
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}  # Default empty info dict

            if isinstance(info, (list, tuple)):
                info = info[0] if info else {}
            
            # Storage for episode data
            portfolio_values = [info.get('portfolio_value', self.config.trading.initial_capital)]
            positions = [info.get('position', 0.0)]
            actions = []
            rewards = []
            prices = [info.get('price', 0.0)]
            timestamps = [info.get('timestamp', 0)]
            nav_values = [1.0]
            sentiment_values = [info.get('sentiment', 0.0)]
            
            # Reward components tracking
            reward_components = {
                'pnl_reward': [],
                'sharpe_reward': [],
                'transaction_penalty': [],
                'drawdown_penalty': [],
                'holding_reward': [],
                'activity_reward': [],
                'sentiment_reward': []
            }
            
            done = False
            step_count = 0
            max_steps = 10000  # Safety limit
            
            while not done and step_count < max_steps:
                # Get action from model
                prediction = model.predict(obs, deterministic=deterministic)
                if isinstance(prediction, tuple):
                    action = prediction[0]  # Unpack tuple
                else:
                    action = prediction  # Use direct value
                
                # Take step
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                if isinstance(info, (list, tuple)):
                    info = info[0] if info else {}
                
                # Store data
                actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
                rewards.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
                portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
                positions.append(info.get('position', positions[-1]))
                prices.append(info.get('price', prices[-1]))
                timestamps.append(info.get('timestamp', timestamps[-1]))
                nav_values.append(info.get('nav', nav_values[-1]))
                sentiment_values.append(info.get('sentiment', sentiment_values[-1]))
                
                # Store reward components if available
                for component in reward_components.keys():
                    value = info.get(component, 0.0)
                    reward_components[component].append(float(value))
                
                step_count += 1
            
            # Get episode summary from base environment
            episode_summary = None
            if hasattr(env, "get_episode_summary"):
                episode_summary = env.get_episode_summary()
            elif hasattr(env, "envs") and env.envs:
                base_env = env.envs[0]
                if hasattr(base_env, "get_episode_summary"):
                    episode_summary = base_env.get_episode_summary()

            
            return {
                'portfolio_values': np.array(portfolio_values),
                'positions': np.array(positions),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'prices': np.array(prices),
                'timestamps': np.array(timestamps),
                'nav_values': np.array(nav_values),
                'sentiment': np.array(sentiment_values),
                'reward_components': reward_components,
                'episode_summary': episode_summary,
                'step_count': step_count
            }
            
        except Exception as e:
            print(f"      ⚠️ Episode failed: {str(e)}")
            return None
    
    def _aggregate_episode_metrics(self, episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple episodes."""
        if not episode_metrics:
            return {}
        
        # Extract numeric metrics only
        numeric_metrics = {}
        for episode in episode_metrics:
            for key, value in episode.items():
                if isinstance(value, (int, float)) and key != 'episode':
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Calculate aggregated statistics
        aggregated = {}
        for key, values in numeric_metrics.items():
            values = np.array(values)
            aggregated.update({
                f'mean_{key}': float(np.mean(values)),
                f'std_{key}': float(np.std(values)),
                f'min_{key}': float(np.min(values)),
                f'max_{key}': float(np.max(values)),
                f'median_{key}': float(np.median(values))
            })
        
        return aggregated
    
    def create_performance_plots(
        self,
        analysis_results: Dict[str, Any],
        save_plots: bool = True,
        show_plots: bool = True,
        style: str = "classic"
    ) -> None:
        """
        Create comprehensive performance visualizations.

        Args:
            analysis_results: Results from analyze_model_performance
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
            style: "classic" for the legacy 3×3 dashboard, anything else
                   falls back to the modular dashboards.
        """
        print("🎨 Creating performance visualizations...")
        
        episode_data = analysis_results.get("episode_data", [])
        if not episode_data:
            print("   ⚠️ No episode data available for visualization")
            return

        # Use first episode for detailed plotting
        main_episode = episode_data[0]
        
        # Create comprehensive dashboard (exact copy from non-modular version)
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Portfolio Value Evolution
        ax1 = plt.subplot(3, 3, 1)
        for i, episode in enumerate(episode_data[:3]):  # Show first 3 episodes
            plt.plot(episode['portfolio_values'], label=f'Episode {i+1}', alpha=0.7)
        plt.axhline(y=self.config.trading.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        plt.title('Portfolio Value Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. NAV Evolution
        ax2 = plt.subplot(3, 3, 2)
        for i, episode in enumerate(episode_data[:3]):
            plt.plot(episode['nav_values'], label=f'Episode {i+1}', alpha=0.7)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        plt.title('NAV Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('NAV')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Position Evolution
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(main_episode['positions'], label='Position', color='green', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Max Long')
        plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Max Short')
        plt.title('Position Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Reward Distribution
        ax4 = plt.subplot(3, 3, 4)
        all_rewards = []
        for episode in episode_data:
            all_rewards.extend(episode['rewards'])
        plt.hist(all_rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Reward')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Cumulative Rewards
        ax5 = plt.subplot(3, 3, 5)
        cumulative_rewards = np.cumsum(main_episode['rewards'])
        plt.plot(cumulative_rewards, color='purple', alpha=0.8)
        plt.title('Cumulative Rewards')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, alpha=0.3)
        
        # 6. Action Distribution
        ax6 = plt.subplot(3, 3, 6)
        all_actions = []
        for episode in episode_data:
            all_actions.extend(episode['actions'])
        plt.hist(all_actions, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral Action')
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Performance Metrics Comparison
        ax7 = plt.subplot(3, 3, 7)
        episode_metrics = analysis_results['episode_metrics']
        if episode_metrics:
            returns = [m['total_return'] for m in episode_metrics]
            sharpes = [m['sharpe_ratio'] for m in episode_metrics]
            episodes = list(range(1, len(returns) + 1))
            
            ax7_twin = ax7.twinx()
            bars1 = ax7.bar([x - 0.2 for x in episodes], returns, 0.4, label='Total Return', alpha=0.7, color='green')
            bars2 = ax7_twin.bar([x + 0.2 for x in episodes], sharpes, 0.4, label='Sharpe Ratio', alpha=0.7, color='blue')
            
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Total Return', color='green')
            ax7_twin.set_ylabel('Sharpe Ratio', color='blue')
            ax7.set_title('Episode Performance Metrics')
            ax7.grid(True, alpha=0.3)
        
        # 8. Drawdown Analysis
        ax8 = plt.subplot(3, 3, 8)
        portfolio_values = np.array(main_episode['portfolio_values'])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red', label='Drawdown')
        plt.plot(drawdown, color='darkred', alpha=0.8)
        plt.title('Drawdown Analysis')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Risk-Return Scatter
        ax9 = plt.subplot(3, 3, 9)
        if episode_metrics:
            returns = [m['total_return'] for m in episode_metrics]
            volatilities = [m['return_volatility'] for m in episode_metrics]
            plt.scatter(volatilities, returns, alpha=0.7, s=100, c=range(len(returns)), cmap='viridis')
            plt.xlabel('Return Volatility')
            plt.ylabel('Total Return')
            plt.title('Risk-Return Profile')
            plt.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Episode')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'performance_analysis_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   💾 Performance analysis saved to: {filepath}")
        
        if show_plots:
            plt.show()

        # 10. Price vs Actions vs Portfolio (exact copy from non-modular version)
        fig2 = plt.figure(figsize=(16, 10))
        gs = fig2.add_gridspec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.25)

        steps = np.arange(len(main_episode["prices"]))
        prices = np.asarray(main_episode["prices"], dtype=float)
        portfolio_vals = np.asarray(main_episode["portfolio_values"], dtype=float)
        actions = np.asarray(main_episode["actions"], dtype=float)

        # Get sentiment values if available
        if 'sentiment' in main_episode:
            sentiment_vals = np.asarray(main_episode["sentiment"], dtype=float)
        else:
            # If no sentiment data, use zeros
            sentiment_vals = np.zeros_like(actions)

        # Align all sequences to the shortest length to avoid matplotlib shape errors
        min_len = min(len(steps), len(prices), len(portfolio_vals), len(actions), len(sentiment_vals))
        if min_len == 0:
            print("⚠️ Insufficient data to plot price-action relationship.")
            return

        if len(steps) != min_len:
            steps = steps[:min_len]
        if len(prices) != min_len:
            prices = prices[:min_len]
        if len(portfolio_vals) != min_len:
            portfolio_vals = portfolio_vals[:min_len]
        if len(actions) != min_len:
            actions = actions[:min_len]
        if len(sentiment_vals) != min_len:
            sentiment_vals = sentiment_vals[:min_len]

        # --- top subplot: price + portfolio ---
        ax_price = fig2.add_subplot(gs[0])
        ax_price.plot(steps, prices, color="black", linewidth=1.5, label="ETH Price")

        action_colors = np.where(actions > 0, "green", np.where(actions < 0, "red", "gray"))
        ax_price.scatter(steps, prices, c=action_colors, s=10, alpha=0.6,
                        label="Agent Action (long/short/flat)")

        ax_portfolio = ax_price.twinx()
        ax_portfolio.plot(steps, portfolio_vals, color="royalblue", linewidth=1.2,
                        label="Portfolio Value")

        ax_price.set_title("ETH Price vs Agent Actions and Portfolio Response")
        ax_price.set_ylabel("ETH Price ($)")
        ax_price.grid(True, alpha=0.3)
        ax_portfolio.set_ylabel("Portfolio Value ($)", color="royalblue")
        ax_portfolio.tick_params(axis="y", labelcolor="royalblue")

        lines1, labels1 = ax_price.get_legend_handles_labels()
        lines2, labels2 = ax_portfolio.get_legend_handles_labels()
        ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

        # --- middle subplot: sentiment on its own scale ---
        ax_sentiment = fig2.add_subplot(gs[1], sharex=ax_price)
        ax_sentiment.plot(steps, sentiment_vals, color="darkorange", linewidth=1.2,
                        label="Sentiment Score")
        ax_sentiment.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_sentiment.set_ylabel("Sentiment")
        ax_sentiment.grid(True, alpha=0.25)
        ax_sentiment.legend(loc="upper left")

        # --- bottom subplot: action trace ---
        ax_actions = fig2.add_subplot(gs[2], sharex=ax_price)
        ax_actions.step(steps, actions, where="post", color="purple",
                        linewidth=1.2, label="Agent Action")
        ax_actions.fill_between(steps, 0, actions, where=(actions >= 0),
                                color="green", alpha=0.2, step="post")
        ax_actions.fill_between(steps, 0, actions, where=(actions < 0),
                                color="red", alpha=0.2, step="post")
        ax_actions.set_ylim(-1.05, 1.05)
        ax_actions.set_ylabel("Action [-1, 1]")
        ax_actions.set_xlabel("Episode Time Step")
        ax_actions.grid(True, alpha=0.25)
        ax_actions.legend(loc="upper right")

        fig2.tight_layout()
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"price_action_portfolio_{timestamp}.png"
            filepath = self.output_dir / filename
            fig2.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"   💾 Price/action/portfolio chart saved to: {filepath}")
        
        if show_plots:
            plt.show()
                
        # Create reward component analysis
        self._create_reward_component_analysis(analysis_results, save_plots)

    def _create_reward_component_analysis(self, 
                                        analysis_results: Dict[str, Any],
                                        save_plots: bool = True) -> None:
        """Create reward component breakdown analysis (exact copy from non-modular version)."""
        print("   🎯 Creating reward component analysis...")
        
        # Check if reward components are available
        if not analysis_results.get('episode_data') or not analysis_results['episode_data']:
            print("      ⚠️ No episode data available for reward component analysis")
            return
            
        breakdown = analysis_results['episode_data'][0].get('reward_components', {})
        if not breakdown:
            print("      ⚠️ No reward components data available")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Get reward components from episode data (exact same logic as non-modular version)
        components = []
        values = []

        for key, label in [
            ('pnl_reward', 'PnL Reward'),
            ('sharpe_reward', 'Sharpe Reward'),
            ('transaction_penalty', 'Transaction Penalty'),
            ('drawdown_penalty', 'Drawdown Penalty'),
            ('holding_reward', 'Holding Reward'),
            ('activity_reward', 'Activity Reward'),
            ('sentiment_reward', 'Sentiment Reward'),
        ]:
            series = breakdown.get(key, [])
            if series and len(series) > 0:
                components.append(label)
                values.append(np.mean(series))
        
        if not components:
            print("      ⚠️ No valid reward components found")
            plt.close(fig)
            return
        
        colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink'][:len(components)]
        
        bars = ax.bar(components, values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Multi-Component Reward Function Analysis')
        ax.set_ylabel('Contribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'reward_components_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   💾 Reward component analysis saved to: {filepath}")
        
        plt.show()

