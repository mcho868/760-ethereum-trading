"""
Hyperparameter Optimizer for DRL Trading System

This module provides comprehensive hyperparameter optimization capabilities
for Deep Reinforcement Learning trading models using various optimization strategies.

Author: DRL Trading Team
"""


import json
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
from tqdm.auto import tqdm

from .config_manager import ConfigManager
from .model_trainer import ModelTrainer
from .rolling_window_trainer import RollingWindowTrainer


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for DRL trading models.
    
    Features:
    - Grid search optimization
    - Random search optimization  
    - Bayesian optimization (with Optuna)
    - Parallel processing support
    - Early stopping mechanisms
    - Performance tracking and analysis
    - Configuration persistence
    - Rolling window optimization
    - Multi-objective optimization support
    """
    
    def __init__(self,
                 trainer: ModelTrainer,
                 config: ConfigManager,
                 config_file: Optional[str] = None):
        """
        Initialize HyperparameterOptimizer.
        
        Args:
            trainer: ModelTrainer instance
            config: ConfigManager instance
            config_file: Path to hyperparameter configuration file
        """
        self.trainer = trainer
        self.config = config
        self.config_file = config_file or config.data.config_file
        
        # Optimization settings
        if self.trainer.device != 'cpu':
            self.max_parallel_jobs = 1
            print(f"   ⚠️ GPU {self.trainer.device}) detected. Setting max_parallel_jobs to 1 to ensure efficient GPU utilization.")
        else:
            # If using CPU, parallelize across available cores
            self.max_parallel_jobs = max(mp.cpu_count() - 1, 1)

        self.early_stopping_patience = config.bulk_testing.early_stopping_patience
        self.performance_metric = config.bulk_testing.performance_metric
        self.min_evaluation_episodes = config.bulk_testing.min_evaluation_episodes
        
        # Results storage
        self.optimization_results = []
        self.best_configurations = []
        self.optimization_history = []
        
        # Setup output directory
        self.output_dir = Path(config.data.output_dir) / 'optimization_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ HyperparameterOptimizer initialized")
        print(f"   💻 Max parallel jobs: {self.max_parallel_jobs}")
        print(f"   📊 Performance metric: {self.performance_metric}")
        print(f"   📁 Output directory: {self.output_dir}")

    def _evaluate_config_stability(self, config: Dict[str, Any], fast_training_timesteps: int, max_windows: int) -> Dict[str, Any]:
        """Helper function to evaluate a single config's stability using rolling windows."""
        try:
            # Use a temporary, faster training config for diagnostics
            diag_config = config.copy()
            diag_config['training'] = diag_config.get('training', self.config.model.training_config.copy())
            diag_config['training']['total_timesteps'] = fast_training_timesteps
            diag_config['training']['verbose'] = 0 # Suppress verbose output

            rolling_trainer = RollingWindowTrainer(self.trainer, self.config)
            
            window_results = rolling_trainer.run_rolling_window_diagnostics(
                sample_config=diag_config,
                max_windows=max_windows,
                save_models=False
            )

            successful_results = [r for r in window_results if r.get('status') != 'failed']
            
            if not successful_results:
                return {'config_id': config.get('config_id'), 'stability_score': -1, 'mean_performance': -float('inf'), 'config': config}

            mean_rewards = [r['mean_reward'] for r in successful_results if 'mean_reward' in r]
            
            if not mean_rewards:
                return {'config_id': config.get('config_id'), 'stability_score': -1, 'mean_performance': -float('inf'), 'config': config}

            std_dev = np.std(mean_rewards)
            mean_perf = np.mean(mean_rewards)
            stability_score = 1.0 / (1.0 + std_dev) if std_dev > 0 else 1.0

            return {
                'config_id': config.get('config_id'),
                'stability_score': stability_score,
                'mean_performance': mean_perf,
                'config': config
            }
        except Exception as e:
            return {'config_id': config.get('config_id'), 'stability_score': -1, 'mean_performance': -float('inf'), 'config': config}


    def filter_and_optimize(self,
                            configurations: List[Dict[str, Any]],
                            survival_rate: float = 0.1,
                            fast_training_timesteps: int = 10000,
                            max_windows: int = 3,
                            use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Filters configurations by stability, then runs full optimization on the survivors.
        """
        print(f"🔪 Starting Phase 1: Culling {len(configurations)} configurations with rolling window diagnostics...")
        
        all_stability_results = []
        
        if use_parallel and self.max_parallel_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
                future_to_config = {
                    executor.submit(self._evaluate_config_stability, config, fast_training_timesteps, max_windows): config
                    for config in configurations
                }
                
                for future in tqdm(as_completed(future_to_config), total=len(configurations), desc="Evaluating Stability"):
                    all_stability_results.append(future.result())
        else:
            for config in tqdm(configurations, desc="Evaluating Stability"):
                all_stability_results.append(self._evaluate_config_stability(config, fast_training_timesteps, max_windows))

        # Filter out failed evaluations
        successful_evals = [res for res in all_stability_results if res['stability_score'] > -1]
        
        if not successful_evals:
            print("   ❌ No configurations survived the stability evaluation. Aborting.")
            return []

        # Rank by stability score (higher is better)
        ranked_configs = sorted(successful_evals, key=lambda x: x['stability_score'], reverse=True)

        # Select the top N survivors
        num_survivors = min(int(len(ranked_configs) * survival_rate), 5)
        survivors = ranked_configs[:num_survivors]
        
        print(f"\n✅ Phase 1 Complete: {len(survivors)} configurations survived (top {survival_rate:.0%}).")
        
        survivor_configs = [res['config'] for res in survivors]

        print("\n🚀 Starting Phase 2: Full optimization on surviving configurations...")
        
        final_results = self.optimize_configurations(
            configurations=survivor_configs,
            use_parallel=use_parallel
        )
        
        return final_results

    def load_configurations(self, config_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load hyperparameter configurations from file.
        
        Args:
            config_file: Path to configuration file (uses instance file if None)
            
        Returns:
            List of configuration dictionaries
        """
        config_file = config_file or self.config_file
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                configurations = config_data.get('configurations', [])
                metadata = config_data.get('metadata', {})
                
            print(f"   ✅ Loaded {len(configurations)} configurations from {config_file}")
            print(f"   📊 Algorithms: {metadata.get('algorithms', ['Unknown'])}")
            
            return configurations
            
        except FileNotFoundError:
            print(f"   ⚠️ Configuration file not found: {config_file}")
            print(f"   🔄 Consider using generate_configurations() to create configurations")
            return []
        except json.JSONDecodeError as e:
            print(f"   ❌ Error parsing configuration file: {e}")
            return []
    def get_best_configurations(self, n_best: int = 10) -> List[Dict[str, Any]]:
        """Get top N best configurations from optimization results."""
        if not self.optimization_results:
            print("   ⚠️ No optimization results available")
            return []
        
        # Filter successful results
        successful_results = [
            r for r in self.optimization_results 
            if self.performance_metric in r and 'error' not in r
        ]
        
        if not successful_results:
            print("   ⚠️ No successful optimization results available")
            return []
        
        # Sort by performance metric
        successful_results.sort(
            key=lambda x: x.get(self.performance_metric, float('-inf')), 
            reverse=True
        )
        
        return successful_results[:n_best]
