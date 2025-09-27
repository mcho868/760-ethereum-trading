"""
Model Trainer for DRL Trading System

This module provides a modular model trainer that integrates with other
components for training A2C and TD3 models using rolling window protocols.

Author: DRL Trading Team
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings

# Stable Baselines3 imports
from stable_baselines3 import A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from .config_manager import ConfigManager
from .trading_environment import TradingEnvironment


class ModelTrainer:
    """
    Model trainer implementing rolling window training protocol as per methodology.
    
    Features:
    - Rolling window training (configurable window sizes)
    - A2C and TD3 model support with full parameter configuration
    - Hyperparameter configuration loading and management
    - Performance tracking and evaluation
    - Model persistence and versioning
    - Integration with modular trading environment
    - Comprehensive logging and monitoring
    """
    
    def __init__(self,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 feature_columns: List[str],
                 config: ConfigManager):
        """
        Initialize ModelTrainer.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            feature_columns: List of feature columns (13D)
            config: ConfigManager instance
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.feature_columns = feature_columns
        self.config = config
        
        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🖥️ Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print(f"🖥️ Using CPU device")
        
        # Set random seed for reproducibility
        if hasattr(config, 'random_seed'):
            set_random_seed(config.random_seed)
        
        # Training history
        self.training_history = []
        
        print(f"✅ ModelTrainer initialized")
        print(f"   📊 Training data: {len(train_data):,} rows")
        print(f"   📊 Validation data: {len(val_data):,} rows")
        print(f"   📊 Test data: {len(test_data):,} rows")
        print(f"   🎯 Features: {len(feature_columns)} dimensions")
        print(f"   🏛️ Environment: Modular TradingEnvironment")
    
    def create_environment(self,
                          data: pd.DataFrame,
                          reward_config: Optional[Dict[str, float]] = None,
                          random_start: bool = True,
                          normalize: bool = False) -> Union[DummyVecEnv, VecNormalize]:
        """
        Create a trading environment with specified data and configuration.
        
        Args:
            data: DataFrame with trading data
            reward_config: Optional reward configuration override
            random_start: Whether to randomize episode start positions
            normalize: Whether to apply observation normalization
            
        Returns:
            Vectorized environment (optionally normalized)
        """
        # Create base environment
        env = TradingEnvironment(
            data=data,
            feature_columns=self.feature_columns,
            config=self.config,
            reward_config=reward_config,
            random_start=random_start
        )
        
        # Wrap in Monitor for logging
        env = Monitor(env)
        
        # Vectorize environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Apply normalization if requested
        if normalize:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
        
        return vec_env
    
    def _train_model(self,
                       model_class,
                       policy: str,
                       config: Dict[str, Any],
                       train_env,
                       val_env,
                       model_params: Dict[str, Any],
                       training_params: Dict[str, Any],
                       action_noise: Optional[NormalActionNoise] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Generic internal training function for A2C and TD3 models.
        """
        config_id = config.get('config_id', 'default')
        algo_name = model_class.__name__

        # Create model
        model_kwargs = {
            'policy': policy,
            'env': train_env,
            'device': self.device,
            'verbose': training_params.get('verbose', 0),
            'seed': config.get('seed', 42),
            **model_params
        }
        if action_noise and algo_name == 'TD3':
            model_kwargs['action_noise'] = action_noise
        
        model = model_class(**model_kwargs)

        # Set up evaluation callback
        eval_callback = None
        if config.get('use_eval_callback', True):
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path=None,
                log_path=training_params.get('eval_log_path'),
                eval_freq=training_params.get('eval_freq', 1000),
                n_eval_episodes=training_params.get('n_eval_episodes', 5),
                deterministic=True,
                render=False,
                verbose=0
            )

        # Train model
        total_timesteps = training_params.get('total_timesteps')
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback,
                tb_log_name=f"{algo_name.lower()}_{config_id}",
                progress_bar=config.get('show_progress', False)
            )
        except Exception as e:
            print(f"❌ {algo_name} training failed: {str(e)}")
            raise
        
        training_time = time.time() - start_time
        
        # Evaluate model performance
        mean_reward, reward_std = self.evaluate_model(model, val_env, n_episodes=5)
        
        print(f"   ✅ {algo_name} training complete: {training_time:.1f}s, "
              f"Mean reward: {mean_reward:.4f} ± {reward_std:.4f}")

        # Compile performance metrics
        performance_metrics = {
            'algorithm': algo_name,
            'config_id': config_id,
            'mean_reward': mean_reward,
            'reward_std': reward_std,
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }
        
        return model, performance_metrics

    def train_a2c_model(self,
                       config: Dict[str, Any],
                       reward_config: Optional[Dict[str, float]] = None,
                       train_data: Optional[pd.DataFrame] = None,
                       val_data: Optional[pd.DataFrame] = None,
                       save_model: bool = True,
                       save_path: Optional[str] = None) -> Tuple[A2C, Dict[str, Any]]:
        train_data = train_data if train_data is not None else self.train_data
        val_data = val_data if val_data is not None else self.val_data
        
        config_id = config.get('config_id', 'default')
        print(f"🚀 Training A2C model: {config_id}")

        train_env = self.create_environment(train_data, reward_config or config.get('reward_components'), True, config.get('normalize_env', False))
        val_env = self.create_environment(val_data, reward_config or config.get('reward_components'), False, False)
        
        model_params = config.get('model_params', self.config.model.a2c_params)
        training_params = config.get('training', self.config.model.training_config)

        model, metrics = self._train_model(A2C, "MlpPolicy", config, train_env, val_env, model_params, training_params)

        if save_model:
            path = save_path or os.path.join(self.config.data.model_dir, f"{config_id}_final.zip")
            self._save_model(model, path)
            metrics['model_path'] = path

        self.training_history.append(metrics)
        return model, metrics

    def train_td3_model(self,
                       config: Dict[str, Any],
                       reward_config: Optional[Dict[str, float]] = None,
                       train_data: Optional[pd.DataFrame] = None,
                       val_data: Optional[pd.DataFrame] = None,
                       save_model: bool = True,
                       save_path: Optional[str] = None) -> Tuple[TD3, Dict[str, Any]]:
        train_data = train_data if train_data is not None else self.train_data
        val_data = val_data if val_data is not None else self.val_data
        
        config_id = config.get('config_id', 'default')
        print(f"🚀 Training TD3 model: {config_id}")

        train_env = self.create_environment(train_data, reward_config or config.get('reward_components'), True, config.get('normalize_env', False))
        val_env = self.create_environment(val_data, reward_config or config.get('reward_components'), False, False)
        
        model_params = config.get('model_params', self.config.model.td3_params)
        training_params = config.get('training', self.config.model.training_config)

        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=model_params.get('noise_std', 0.1) * np.ones(1)) if model_params.get('noise_std', 0) > 0 else None

        model, metrics = self._train_model(TD3, "MlpPolicy", config, train_env, val_env, model_params, training_params, action_noise)

        if save_model:
            path = save_path or os.path.join(self.config.data.model_dir, f"{config_id}_final.zip")
            self._save_model(model, path)
            metrics['model_path'] = path

        self.training_history.append(metrics)
        return model, metrics
    
    def evaluate_model(self,
                      model,
                      env,
                      n_episodes: int = 10,
                      deterministic: bool = True) -> Tuple[float, float]:
        """
        Evaluate model performance over multiple episodes.
        
        Args:
            model: Trained model to evaluate
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (mean_reward, reward_std)
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 10000  # Safety limit
            
            while not done and steps < max_steps:
                try:
                    prediction = model.predict(obs, deterministic=deterministic)
                    if isinstance(prediction, tuple):
                        action = prediction[0]  # Unpack tuple
                    else:
                        action = prediction  # Use direct value
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0]  # Extract scalar from array
                    steps += 1
                except Exception as e:
                    print(f"⚠️ Evaluation error in episode {episode + 1}: {str(e)}")
                    break
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        
        return mean_reward, reward_std
    
    def rolling_window_training(self,
                               config: Dict[str, Any],
                               rolling_months: Optional[int] = None,
                               eval_months: Optional[int] = None,
                               step_months: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Implement rolling window training as per methodology.
        
        Args:
            config: Model configuration dictionary
            rolling_months: Rolling window size in months (uses config default if None)
            eval_months: Evaluation period in months (uses config default if None)
            step_months: Step size in months (uses config default if None)
            
        Returns:
            List of performance results for each window
        """
        # Use configuration defaults if not provided
        protocol_config = self.config.training_protocol
        rolling_months = rolling_months or protocol_config.rolling_window_months
        eval_months = eval_months or protocol_config.evaluation_period_months
        step_months = step_months or protocol_config.rolling_step_months
        
        print(f"🔄 Starting rolling window training")
        print(f"   📅 Rolling window: {rolling_months} months")
        print(f"   📊 Evaluation period: {eval_months} months")
        print(f"   📈 Step size: {step_months} months")
        
        # Calculate window sizes (simplified - using row indices)
        total_rows = len(self.train_data)
        minutes_per_month = 30 * 24 * 60  # Approximate
        
        rolling_window_size = min(rolling_months * minutes_per_month, total_rows // 2)
        eval_window_size = min(eval_months * minutes_per_month, total_rows // 10)
        step_size = step_months * minutes_per_month
        
        results = []
        window_start = 0
        
        while window_start + rolling_window_size + eval_window_size <= total_rows:
            # Define current windows
            train_window_end = window_start + rolling_window_size
            eval_window_end = train_window_end + eval_window_size
            
            # Extract data for current window
            current_train_data = self.train_data.iloc[window_start:train_window_end].copy()
            current_eval_data = self.train_data.iloc[train_window_end:eval_window_end].copy()
            
            window_idx = len(results) + 1
            print(f"   📊 Window {window_idx}: Training [{window_start}:{train_window_end}], "
                  f"Eval [{train_window_end}:{eval_window_end}]")
            
            # Train model for current window
            algorithm = config.get('algorithm', 'A2C').upper()
            
            try:
                if algorithm == 'A2C':
                    model, metrics = self.train_a2c_model(
                        config, 
                        train_data=current_train_data, 
                        val_data=current_eval_data
                    )
                elif algorithm == 'TD3':
                    model, metrics = self.train_td3_model(
                        config,
                        train_data=current_train_data,
                        val_data=current_eval_data
                    )
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Add window information to metrics
                metrics.update({
                    'window_index': window_idx,
                    'train_start_idx': window_start,
                    'train_end_idx': train_window_end,
                    'eval_start_idx': train_window_end,
                    'eval_end_idx': eval_window_end,
                    'train_rows': len(current_train_data),
                    'eval_rows': len(current_eval_data),
                    'rolling_months': rolling_months,
                    'eval_months': eval_months
                })
                
                results.append(metrics)
                
                print(f"      ✅ Window {window_idx} complete, "
                      f"Mean reward: {metrics['mean_reward']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Window {window_idx} failed: {str(e)}")
                # Add failed window info
                results.append({
                    'window_index': window_idx,
                    'status': 'failed',
                    'error': str(e),
                    'train_start_idx': window_start,
                    'train_end_idx': train_window_end,
                    'eval_start_idx': train_window_end,
                    'eval_end_idx': eval_window_end
                })
            
            # Move to next window
            window_start += step_size
        
        print(f"🎉 Rolling window training complete! {len(results)} windows processed")
        
        return results
    
    def load_model(self, model_path: str, algorithm: str) -> Union[A2C, TD3]:
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
            algorithm: Algorithm type ('A2C' or 'TD3')
            
        Returns:
            Loaded model
        """
        try:
            if algorithm.upper() == 'A2C':
                model = A2C.load(model_path, device=self.device)
            elif algorithm.upper() == 'TD3':
                model = TD3.load(model_path, device=self.device)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            print(f"✅ Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {str(e)}")
            raise
    
    def _save_model(self, model, path: str) -> str:
        """Save trained model to disk."""
        try:
            model.save(path)
            print(f"   💾 Model saved: {path}")
            return path
        except Exception as e:
            print(f"   ⚠️ Failed to save model: {str(e)}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training sessions."""
        if not self.training_history:
            return {'message': 'No training sessions completed'}
        
        algorithms = [h['algorithm'] for h in self.training_history]
        rewards = [h['mean_reward'] for h in self.training_history]
        times = [h['training_time'] for h in self.training_history]
        
        return {
            'total_sessions': len(self.training_history),
            'algorithms_used': list(set(algorithms)),
            'best_mean_reward': max(rewards),
            'worst_mean_reward': min(rewards),
            'average_reward': np.mean(rewards),
            'total_training_time': sum(times),
            'average_training_time': np.mean(times),
            'last_session': self.training_history[-1] if self.training_history else None
        }
    
    def compare_models(self, model_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model configurations by training and evaluating them.
        
        Args:
            model_configs: List of model configuration dictionaries
            
        Returns:
            DataFrame with comparison results
        """
        print(f"🔍 Comparing {len(model_configs)} model configurations...")
        
        comparison_results = []
        
        for i, config in enumerate(model_configs):
            print(f"\n📊 Training configuration {i+1}/{len(model_configs)}: {config.get('config_id', f'config_{i+1}')}")
            
            try:
                algorithm = config.get('algorithm', 'A2C').upper()
                
                if algorithm == 'A2C':
                    model, metrics = self.train_a2c_model(config, save_model=False)
                elif algorithm == 'TD3':
                    model, metrics = self.train_td3_model(config, save_model=False)
                else:
                    print(f"   ⚠️ Unsupported algorithm: {algorithm}")
                    continue
                
                comparison_results.append(metrics)
                
            except Exception as e:
                print(f"   ❌ Configuration failed: {str(e)}")
                comparison_results.append({
                    'config_id': config.get('config_id', f'config_{i+1}'),
                    'algorithm': config.get('algorithm', 'Unknown'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Convert to DataFrame for easy comparison
        df = pd.DataFrame(comparison_results)
        
        if len(df) > 0 and 'mean_reward' in df.columns:
            df = df.sort_values('mean_reward', ascending=False)
            print(f"\n🏆 Best performing configuration: {df.iloc[0]['config_id']} "
                  f"(Mean reward: {df.iloc[0]['mean_reward']:.4f})")
        
        return df


if __name__ == "__main__":
    # Example usage
    from .config_manager import ConfigManager
    from .data_processor import DataProcessor
    
    # Create configuration and data
    config = ConfigManager()
    processor = DataProcessor(config)
    
    # Mock data for testing
    dummy_data = pd.DataFrame({
        'ts': np.arange(10000),
        'close': 100 + np.random.randn(10000).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, 10000)
    })
    
    # Add features
    feature_cols = config.get_feature_columns()
    for col in feature_cols:
        dummy_data[col] = np.random.randn(10000)
    
    # Create data splits
    train_split = int(0.7 * len(dummy_data))
    val_split = int(0.85 * len(dummy_data))
    
    train_data = dummy_data[:train_split].copy()
    val_data = dummy_data[train_split:val_split].copy()
    test_data = dummy_data[val_split:].copy()
    
    # Initialize trainer
    trainer = ModelTrainer(train_data, val_data, test_data, feature_cols, config)
    
    # Example configuration
    test_config = {
        'config_id': 'test_a2c',
        'algorithm': 'A2C',
        'model_params': config.model.a2c_params,
        'training': {
            'total_timesteps': 10000,
            'eval_freq': 2000,
            'n_eval_episodes': 3
        }
    }
    
    # Train model
    model, metrics = trainer.train_a2c_model(test_config, save_model=False)
    print(f"\nTraining completed: {metrics}")
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"\nTraining summary: {summary}")
