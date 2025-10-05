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

# Stable Baselines3 imports
from stable_baselines3 import A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
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
                 config: ConfigManager,
                 device: str = "cpu",
        ):
        """
        Initialize ModelTrainer.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            feature_columns: List of feature columns (13D)
            config: ConfigManager instance
            device: Device to use for training
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.feature_columns = feature_columns
        self.config = config
        self.device = device
        
        if self.device == "gpu":
            # Set up device
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"🖥️ Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print(f"🖥️ Using MPS device")

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
    def _save_model(self, model, path: str) -> str:
        """Save trained model to disk."""
        try:
            model.save(path)
            print(f"   💾 Model saved: {path}")
            return path
        except Exception as e:
            print(f"   ⚠️ Failed to save model: {str(e)}")
            return None
