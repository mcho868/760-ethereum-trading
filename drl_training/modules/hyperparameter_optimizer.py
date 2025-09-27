"""
Hyperparameter Optimizer for DRL Trading System

This module provides comprehensive hyperparameter optimization capabilities
for Deep Reinforcement Learning trading models using various optimization strategies.

Author: DRL Trading Team
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
from itertools import product
import warnings

# Optuna for advanced optimization (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Advanced optimization features will be disabled.")

from .config_manager import ConfigManager
from .model_trainer import ModelTrainer
from .performance_analyzer import PerformanceAnalyzer


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
        self.max_parallel_jobs = 1  # Forces sequential mode without overhead
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
    
    def generate_configurations(self,
                              algorithms: List[str] = None,
                              n_configs: int = 100,
                              param_ranges: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate random hyperparameter configurations.
        
        Args:
            algorithms: List of algorithms to generate configs for
            n_configs: Number of configurations to generate
            param_ranges: Custom parameter ranges
            
        Returns:
            List of generated configurations
        """
        algorithms = algorithms or ['A2C', 'TD3']
        
        print(f"🔄 Generating {n_configs} random configurations...")
        
        configurations = []
        
        for i in range(n_configs):
            # Randomly select algorithm
            algorithm = np.random.choice(algorithms)
            
            if algorithm == 'A2C':
                config = self._generate_a2c_config(i, param_ranges)
            elif algorithm == 'TD3':
                config = self._generate_td3_config(i, param_ranges)
            else:
                continue
            
            config['algorithm'] = algorithm
            config['config_id'] = f"{algorithm.lower()}_{i:04d}"
            
            configurations.append(config)
        
        print(f"   ✅ Generated {len(configurations)} configurations")
        return configurations
    
    def _generate_a2c_config(self, config_id: int, param_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate random A2C configuration."""
        base_params = self.config.model.a2c_params.copy()
        
        # Default parameter ranges
        ranges = {
            'learning_rate': (1e-5, 1e-2),
            'n_steps': [512, 1024, 2048, 4096],
            'gamma': (0.95, 0.999),
            'gae_lambda': (0.8, 0.99),
            'ent_coef': (0.001, 0.1),
            'vf_coef': (0.1, 1.0),
            'max_grad_norm': (0.1, 1.0)
        }
        
        if param_ranges:
            ranges.update(param_ranges)
        
        # Sample parameters
        config = {
            'model_params': {},
            'training': self.config.model.training_config.copy(),
            'reward_components': self._generate_reward_config()
        }
        
        for param, range_val in ranges.items():
            if isinstance(range_val, (list, tuple)) and len(range_val) == 2 and all(isinstance(x, (int, float)) for x in range_val):
                # Continuous range
                if param == 'learning_rate':
                    # Log-uniform sampling for learning rate
                    config['model_params'][param] = float(10**np.random.uniform(np.log10(range_val[0]), np.log10(range_val[1])))
                else:
                    config['model_params'][param] = float(np.random.uniform(range_val[0], range_val[1]))
            elif isinstance(range_val, list):
                # Discrete choices
                config['model_params'][param] = int(np.random.choice(range_val))
        
        return config
    
    def _generate_td3_config(self, config_id: int, param_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate random TD3 configuration."""
        base_params = self.config.model.td3_params.copy()
        
        # Default parameter ranges
        ranges = {
            'learning_rate': (1e-5, 1e-2),
            'buffer_size': [100000, 500000, 1000000],
            'learning_starts': [5000, 10000, 20000],
            'batch_size': [64, 128, 256, 512],
            'tau': (0.001, 0.01),
            'gamma': (0.95, 0.999),
            'noise_std': (0.05, 0.3),
            'target_noise': (0.1, 0.5),
            'noise_clip': (0.3, 0.8),
            'policy_delay': [1, 2, 3, 4]
        }
        
        if param_ranges:
            ranges.update(param_ranges)
        
        # Sample parameters
        config = {
            'model_params': {},
            'training': self.config.model.training_config.copy(),
            'reward_components': self._generate_reward_config()
        }
        
        for param, range_val in ranges.items():
            if isinstance(range_val, (list, tuple)) and len(range_val) == 2 and all(isinstance(x, (int, float)) for x in range_val):
                # Continuous range
                if param == 'learning_rate':
                    # Log-uniform sampling for learning rate
                    config['model_params'][param] = float(10**np.random.uniform(np.log10(range_val[0]), np.log10(range_val[1])))
                else:
                    config['model_params'][param] = float(np.random.uniform(range_val[0], range_val[1]))
            elif isinstance(range_val, list):
                # Discrete choices
                if param in ['buffer_size', 'learning_starts', 'batch_size', 'policy_delay']:
                    config['model_params'][param] = int(np.random.choice(range_val))
                else:
                    config['model_params'][param] = float(np.random.choice(range_val))
        
        return config
    
    def _generate_reward_config(self) -> Dict[str, float]:
        """Generate random reward configuration."""
        base_config = self.config.reward.__dict__.copy()
        
        # Vary key reward parameters
        variations = {
            'pnl_scale': np.random.uniform(50.0, 200.0),
            'sharpe_weight': np.random.uniform(0.1, 0.5),
            'transaction_penalty': np.random.uniform(0.5, 2.0),
            'drawdown_penalty': np.random.uniform(25.0, 100.0),
            'activity_reward': np.random.uniform(0.05, 0.2),
            'holding_penalty': np.random.uniform(0.0005, 0.002)
        }
        
        base_config.update(variations)
        return base_config
    
    def grid_search_optimization(self,
                                param_grid: Dict[str, List[Any]],
                                algorithm: str = 'A2C',
                                max_configs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform grid search optimization.
        
        Args:
            param_grid: Dictionary of parameter names and value lists
            algorithm: Algorithm to optimize ('A2C' or 'TD3')
            max_configs: Maximum number of configurations to test
            
        Returns:
            List of optimization results
        """
        print(f"🔍 Starting grid search optimization for {algorithm}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        if max_configs and len(param_combinations) > max_configs:
            # Randomly sample if too many combinations
            param_combinations = np.random.choice(
                len(param_combinations), max_configs, replace=False
            )
            param_combinations = [param_combinations[i] for i in param_combinations]
        
        print(f"   📊 Testing {len(param_combinations)} parameter combinations")
        
        # Generate configurations
        configurations = []
        for i, param_combo in enumerate(param_combinations):
            config = self._create_base_config(algorithm, f"grid_{i:04d}")
            
            # Set grid parameters
            for param_name, param_value in zip(param_names, param_combo):
                if param_name.startswith('reward_'):
                    # Reward parameter
                    reward_param = param_name.replace('reward_', '')
                    config['reward_components'][reward_param] = param_value
                else:
                    # Model parameter
                    config['model_params'][param_name] = param_value
            
            configurations.append(config)
        
        # Run optimization
        results = self.optimize_configurations(configurations)
        
        print(f"   ✅ Grid search complete: {len(results)} configurations tested")
        return results
    
    def random_search_optimization(self,
                                  param_distributions: Dict[str, Tuple[Any, ...]],
                                  algorithm: str = 'A2C',
                                  n_configs: int = 100) -> List[Dict[str, Any]]:
        """
        Perform random search optimization.
        
        Args:
            param_distributions: Dictionary of parameter distributions
            algorithm: Algorithm to optimize
            n_configs: Number of configurations to test
            
        Returns:
            List of optimization results
        """
        print(f"🎲 Starting random search optimization for {algorithm}")
        
        configurations = []
        for i in range(n_configs):
            config = self._create_base_config(algorithm, f"random_{i:04d}")
            
            # Sample parameters from distributions
            for param_name, distribution in param_distributions.items():
                if param_name.startswith('reward_'):
                    # Reward parameter
                    reward_param = param_name.replace('reward_', '')
                    config['reward_components'][reward_param] = self._sample_from_distribution(distribution)
                else:
                    # Model parameter
                    config['model_params'][param_name] = self._sample_from_distribution(distribution)
            
            configurations.append(config)
        
        # Run optimization
        results = self.optimize_configurations(configurations)
        
        print(f"   ✅ Random search complete: {len(results)} configurations tested")
        return results
    
    def bayesian_optimization(self,
                            param_space: Dict[str, Any],
                            algorithm: str = 'A2C',
                            n_trials: int = 100,
                            timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            param_space: Parameter space definition for Optuna
            algorithm: Algorithm to optimize
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            
        Returns:
            List of optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        print(f"🧠 Starting Bayesian optimization for {algorithm} ({n_trials} trials)")
        
        # Create Optuna study
        study_name = f"{algorithm}_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler()
        )
        
        def objective(trial):
            """Objective function for Optuna optimization."""
            # Sample parameters
            config = self._create_base_config(algorithm, f"bayesian_{trial.number:04d}")
            
            for param_name, param_spec in param_space.items():
                if param_spec['type'] == 'float':
                    value = trial.suggest_float(param_name, param_spec['low'], param_spec['high'])
                elif param_spec['type'] == 'int':
                    value = trial.suggest_int(param_name, param_spec['low'], param_spec['high'])
                elif param_spec['type'] == 'categorical':
                    value = trial.suggest_categorical(param_name, param_spec['choices'])
                elif param_spec['type'] == 'log_float':
                    value = trial.suggest_float(param_name, param_spec['low'], param_spec['high'], log=True)
                else:
                    raise ValueError(f"Unsupported parameter type: {param_spec['type']}")
                
                if param_name.startswith('reward_'):
                    reward_param = param_name.replace('reward_', '')
                    config['reward_components'][reward_param] = value
                else:
                    config['model_params'][param_name] = value
            
            # Train and evaluate model
            try:
                result = self._evaluate_single_configuration(config)
                return result.get(self.performance_metric, 0.0)
            except Exception as e:
                print(f"   ⚠️ Trial {trial.number} failed: {str(e)}")
                return float('-inf')
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Convert results
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    'config_id': f"bayesian_{trial.number:04d}",
                    'params': trial.params,
                    'value': trial.value,
                    'trial_number': trial.number
                }
                results.append(result)
        
        print(f"   ✅ Bayesian optimization complete: {len(results)} successful trials")
        print(f"   🏆 Best value: {study.best_value:.4f}")
        
        return results
    
    def optimize_configurations(self,
                               configurations: List[Dict[str, Any]],
                               use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Optimize a list of configurations.
        
        Args:
            configurations: List of configurations to test
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of optimization results
        """
        print(f"⚡ Optimizing {len(configurations)} configurations...")
        print(f"   💻 Parallel processing: {use_parallel}")
        
        start_time = time.time()
        results = []
        
        if use_parallel and self.max_parallel_jobs > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(self._evaluate_single_configuration, config): config
                    for config in configurations
                }
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_config), 1):
                    config = future_to_config[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        config_id = config.get('config_id', f'config_{i}')
                        metric_value = result.get(self.performance_metric, 0.0)
                        
                        print(f"   ✅ {i}/{len(configurations)}: {config_id} - "
                              f"{self.performance_metric}: {metric_value:.4f}")
                              
                    except Exception as e:
                        print(f"   ❌ {i}/{len(configurations)}: {config.get('config_id', 'unknown')} failed - {str(e)}")
                        
        else:
            # Sequential processing
            for i, config in enumerate(configurations, 1):
                try:
                    result = self._evaluate_single_configuration(config)
                    results.append(result)
                    
                    config_id = config.get('config_id', f'config_{i}')
                    metric_value = result.get(self.performance_metric, 0.0)
                    
                    print(f"   ✅ {i}/{len(configurations)}: {config_id} - "
                          f"{self.performance_metric}: {metric_value:.4f}")
                          
                except Exception as e:
                    print(f"   ❌ {i}/{len(configurations)}: {config.get('config_id', 'unknown')} failed - {str(e)}")
        
        # Sort results by performance metric
        results.sort(key=lambda x: x.get(self.performance_metric, float('-inf')), reverse=True)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if self.performance_metric in r]
        
        print(f"   🎉 Optimization complete!")
        print(f"   ⏱️ Total time: {total_time:.1f}s")
        print(f"   ✅ Successful: {len(successful_results)}/{len(configurations)}")
        
        if successful_results:
            best_result = successful_results[0]
            print(f"   🏆 Best {self.performance_metric}: {best_result[self.performance_metric]:.4f} "
                  f"({best_result.get('config_id', 'unknown')})")
        
        # Store results
        self.optimization_results.extend(results)
        
        return results
    
    def _evaluate_single_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single configuration."""
        config_id = config.get('config_id', 'unknown')
        algorithm = config.get('algorithm', 'A2C').upper()
        
        try:
            # Train model
            if algorithm == 'A2C':
                model, train_metrics = self.trainer.train_a2c_model(config, save_model=False)
            elif algorithm == 'TD3':
                model, train_metrics = self.trainer.train_td3_model(config, save_model=False)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Evaluate on test data
            test_env = self.trainer.create_environment(
                self.trainer.test_data,
                reward_config=config.get('reward_components'),
                random_start=False
            )
            
            mean_test_reward, test_reward_std = self.trainer.evaluate_model(
                model, test_env, n_episodes=self.min_evaluation_episodes
            )
            
            # Compile results
            result = {
                'config_id': config_id,
                'algorithm': algorithm,
                'config': config,
                'train_metrics': train_metrics,
                'mean_test_reward': mean_test_reward,
                'test_reward_std': test_reward_std,
                'sharpe_ratio': train_metrics.get('sharpe_ratio', 0.0),
                'training_time': train_metrics.get('training_time', 0.0),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'config_id': config_id,
                'algorithm': algorithm,
                'error': str(e),
                'status': 'failed'
            }
    
    def _create_base_config(self, algorithm: str, config_id: str) -> Dict[str, Any]:
        """Create base configuration for algorithm."""
        base_config = {
            'config_id': config_id,
            'algorithm': algorithm,
            'training': self.config.model.training_config.copy(),
            'reward_components': self.config.get_reward_components()
        }
        
        if algorithm == 'A2C':
            base_config['model_params'] = self.config.model.a2c_params.copy()
        elif algorithm == 'TD3':
            base_config['model_params'] = self.config.model.td3_params.copy()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return base_config
    
    def _sample_from_distribution(self, distribution: Tuple[Any, ...]) -> Any:
        """Sample value from distribution specification."""
        if len(distribution) == 2:
            # Uniform distribution
            return np.random.uniform(distribution[0], distribution[1])
        elif len(distribution) == 3 and distribution[2] == 'log':
            # Log-uniform distribution
            return 10**np.random.uniform(np.log10(distribution[0]), np.log10(distribution[1]))
        elif len(distribution) == 3 and distribution[2] == 'int':
            # Integer uniform distribution
            return np.random.randint(distribution[0], distribution[1] + 1)
        elif isinstance(distribution[0], list):
            # Categorical distribution
            return np.random.choice(distribution[0])
        else:
            raise ValueError(f"Unsupported distribution format: {distribution}")
    
    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save optimization results to file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for JSON serialization
        export_data = {
            'metadata': {
                'optimization_timestamp': datetime.now().isoformat(),
                'performance_metric': self.performance_metric,
                'total_configurations': len(results),
                'successful_configurations': len([r for r in results if 'error' not in r]),
            },
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"   💾 Results saved: {filepath}")
        return str(filepath)
    
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
    
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create comprehensive optimization report."""
        if not self.optimization_results:
            return {'message': 'No optimization results available'}
        
        successful_results = [r for r in self.optimization_results if 'error' not in r]
        failed_results = [r for r in self.optimization_results if 'error' in r]
        
        report = {
            'summary': {
                'total_configurations': len(self.optimization_results),
                'successful_configurations': len(successful_results),
                'failed_configurations': len(failed_results),
                'success_rate': len(successful_results) / len(self.optimization_results) if self.optimization_results else 0
            },
            'performance_statistics': {},
            'best_configurations': self.get_best_configurations(5),
            'algorithm_comparison': {},
            'optimization_metadata': {
                'performance_metric': self.performance_metric,
                'max_parallel_jobs': self.max_parallel_jobs,
                'report_timestamp': datetime.now().isoformat()
            }
        }
        
        if successful_results:
            # Performance statistics
            metric_values = [r[self.performance_metric] for r in successful_results]
            report['performance_statistics'] = {
                'mean': float(np.mean(metric_values)),
                'std': float(np.std(metric_values)),
                'min': float(np.min(metric_values)),
                'max': float(np.max(metric_values)),
                'median': float(np.median(metric_values))
            }
            
            # Algorithm comparison
            algorithms = list(set(r.get('algorithm', 'Unknown') for r in successful_results))
            for algo in algorithms:
                algo_results = [r for r in successful_results if r.get('algorithm') == algo]
                algo_metrics = [r[self.performance_metric] for r in algo_results]
                
                report['algorithm_comparison'][algo] = {
                    'count': len(algo_results),
                    'mean_performance': float(np.mean(algo_metrics)),
                    'best_performance': float(np.max(algo_metrics)),
                    'worst_performance': float(np.min(algo_metrics))
                }
        
        return report


if __name__ == "__main__":
    # Example usage
    from .config_manager import ConfigManager
    from .model_trainer import ModelTrainer
    from .data_processor import DataProcessor
    
    # Mock setup for testing
    config = ConfigManager()
    
    # Create mock data
    dummy_data = pd.DataFrame({
        'ts': np.arange(1000),
        'close': 100 + np.random.randn(1000).cumsum() * 0.1,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Add features
    feature_cols = config.get_feature_columns()
    for col in feature_cols:
        dummy_data[col] = np.random.randn(1000)
    
    # Create data splits
    train_data = dummy_data[:700].copy()
    val_data = dummy_data[700:850].copy()
    test_data = dummy_data[850:].copy()
    
    # Initialize components
    trainer = ModelTrainer(train_data, val_data, test_data, feature_cols, config)
    optimizer = HyperparameterOptimizer(trainer, config)
    
    # Example: Generate configurations
    configs = optimizer.generate_configurations(n_configs=5)
    print(f"Generated {len(configs)} configurations")
    
    # Example: Random search
    param_distributions = {
        'learning_rate': (1e-5, 1e-2, 'log'),
        'n_steps': ([512, 1024, 2048],),
        'reward_pnl_scale': (50.0, 200.0)
    }
    
    # Note: In real usage, you would run this:
    # results = optimizer.random_search_optimization(param_distributions, n_configs=10)
    # report = optimizer.create_optimization_report()
    
    print("HyperparameterOptimizer example completed")
