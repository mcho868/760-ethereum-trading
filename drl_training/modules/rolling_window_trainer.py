import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm.auto import tqdm

from .config_manager import ConfigManager
from .model_trainer import ModelTrainer


class RollingWindowTrainer:
    """
    Rolling window training and temporal validation for DRL trading models.
    
    Features:
    - Rolling window diagnostics on training data
    - Temporal stability assessment
    - Performance consistency analysis across time periods
    - Model robustness evaluation
    """
    
    def __init__(self, model_trainer: ModelTrainer, config: ConfigManager):
        """
        Initialize RollingWindowTrainer.
        
        Args:
            model_trainer: ModelTrainer instance
            config: ConfigManager instance
        """
        self.model_trainer = model_trainer
        self.config = config
        self.training_protocol_config = config.training_protocol
        
        print("✅ RollingWindowTrainer initialized")
        
    def run_rolling_window_diagnostics(self,
                                     sample_config: Dict[str, Any],
                                     rolling_months: int = None,
                                     eval_months: int = None,
                                     max_windows: int = 5,
                                     save_models: bool = False) -> List[Dict[str, Any]]:
        """
        Implement rolling window training diagnostics on training data only.
        
        Args:
            sample_config: Configuration for testing
            rolling_months: Rolling window size in months (None = use config)
            eval_months: Evaluation period in months (None = use config)
            max_windows: Maximum number of windows to test
            save_models: Whether to save models from each window
            
        Returns:
            List of performance results for each window
        """
        # Use config defaults if not specified
        rolling_months = rolling_months or self.training_protocol_config.rolling_window_months
        eval_months = eval_months or self.training_protocol_config.evaluation_period_months
        
        print(f"🔄 Starting rolling window diagnostics")
        print(f"   📅 Rolling window: {rolling_months} months")
        print(f"   📊 Evaluation period: {eval_months} months")
        print(f"   🔢 Max windows: {max_windows}")
        
        # Calculate time windows (using row indices as proxy for time)
        total_rows = len(self.model_trainer.train_data)
        minutes_per_month = 30 * 24 * 60  # Approximate
        
        rolling_window_size = min(rolling_months * minutes_per_month, total_rows // 3)
        eval_window_size = min(eval_months * minutes_per_month, total_rows // 10)
        
        # Ensure minimum viable window sizes
        min_window_size = self.config.trading.episode_length * 2  # At least 2 episodes
        rolling_window_size = max(rolling_window_size, min_window_size)
        eval_window_size = max(eval_window_size, min_window_size // 2)
        
        results = []
        window_start = 0
        
        print(f"   📏 Window size: {rolling_window_size:,} rows")
        print(f"   📏 Eval size: {eval_window_size:,} rows")
        print(f"   📊 Total training data: {total_rows:,} rows")
        
        window_count = 0
        progress_bar = tqdm(desc="Rolling Windows", total=max_windows)
        
        while (window_start + rolling_window_size + eval_window_size <= total_rows and 
               window_count < max_windows):
            
            # Define current windows
            train_window_end = window_start + rolling_window_size
            eval_window_end = train_window_end + eval_window_size
            
            # Extract data for current window
            current_train_data = self.model_trainer.train_data.iloc[window_start:train_window_end].copy().reset_index(drop=True)
            current_eval_data = self.model_trainer.train_data.iloc[train_window_end:eval_window_end].copy().reset_index(drop=True)
            
            window_count += 1
            progress_bar.set_description(f"Window {window_count}/{max_windows}")
            
            print(f"\n   📊 Window {window_count}: Training [{window_start:,}:{train_window_end:,}], Eval [{train_window_end:,}:{eval_window_end:,}]")
            print(f"      📈 Train rows: {len(current_train_data):,}, Eval rows: {len(current_eval_data):,}")
            
            try:
                # Train model for current window
                algorithm = sample_config.get('algorithm', 'A2C')
                window_config = sample_config.copy()
                window_config['config_id'] = f"{sample_config.get('config_id', 'rolling')}_window_{window_count}"
                
                if algorithm == 'A2C':
                    model, metrics = self.model_trainer.train_a2c_model(
                        window_config, 
                        train_data=current_train_data, 
                        val_data=current_eval_data,
                        save_model=save_models
                    )
                elif algorithm == 'TD3':
                    model, metrics = self.model_trainer.train_td3_model(
                        window_config, 
                        train_data=current_train_data, 
                        val_data=current_eval_data,
                        save_model=save_models
                    )
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Add window information to metrics
                metrics.update({
                    'window_index': window_count,
                    'train_start_idx': window_start,
                    'train_end_idx': train_window_end,
                    'eval_start_idx': train_window_end,
                    'eval_end_idx': eval_window_end,
                    'train_rows': len(current_train_data),
                    'eval_rows': len(current_eval_data),
                    'algorithm': algorithm,
                    'rolling_window_months': rolling_months,
                    'eval_window_months': eval_months
                })
                
                # Save model if requested
                if save_models:
                    model_path = os.path.join(
                        self.config.data.model_dir, 
                        f"rolling_window_{window_count}_{window_config['config_id']}"
                    )
                    os.makedirs(self.config.data.model_dir, exist_ok=True)
                    model.save(model_path)
                    metrics['model_path'] = model_path
                    print(f"      💾 Model saved: {model_path}")
                
                results.append(metrics)
                print(f"      ✅ Window {window_count} complete: Mean reward = {metrics['mean_reward']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Window {window_count} failed: {str(e)}")
                # Add failed window info for completeness
                failed_metrics = {
                    'window_index': window_count,
                    'algorithm': algorithm,
                    'status': 'failed',
                    'error': str(e),
                    'mean_reward': float('-inf'),
                    'train_start_idx': window_start,
                    'train_end_idx': train_window_end,
                    'eval_start_idx': train_window_end,
                    'eval_end_idx': eval_window_end,
                }
                results.append(failed_metrics)
                continue
            
            # Move to next window (step by evaluation period for some overlap)
            step_size = self.training_protocol_config.rolling_step_months * minutes_per_month
            step_size = min(step_size, eval_window_size)  # Ensure reasonable step size
            window_start += step_size
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        print(f"\n🎉 Rolling window diagnostics complete! {len(results)} windows processed")
        
        # Analyze rolling window performance
        successful_results = [r for r in results if r.get('status') != 'failed']
        
        if successful_results:
            analysis = self._analyze_rolling_performance(successful_results)
            
            # Save results
            results_file = os.path.join(self.config.data.output_dir, 'rolling_window_results.json')
            self._save_rolling_results(results, analysis, results_file)
            
            return results
        else:
            print("   ⚠️ No successful windows to analyze")
            return results
    
    def _analyze_rolling_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze rolling window performance statistics."""
        mean_rewards = [r['mean_reward'] for r in results if 'mean_reward' in r]
        training_times = [r['training_time'] for r in results if 'training_time' in r]
        
        if not mean_rewards:
            return {}
        
        analysis = {
            'n_windows': len(results),
            'n_successful': len(mean_rewards),
            'mean_performance': np.mean(mean_rewards),
            'std_performance': np.std(mean_rewards),
            'min_performance': np.min(mean_rewards),
            'max_performance': np.max(mean_rewards),
            'performance_range': np.max(mean_rewards) - np.min(mean_rewards),
            'coefficient_of_variation': np.std(mean_rewards) / abs(np.mean(mean_rewards)) if np.mean(mean_rewards) != 0 else np.inf,
            'stability_score': 1.0 / (1.0 + np.std(mean_rewards)),  # Higher = more stable
        }
        
        if training_times:
            analysis.update({
                'mean_training_time': np.mean(training_times),
                'total_training_time': np.sum(training_times),
            })
        
        print(f"\n📊 Rolling Window Analysis:")
        print(f"   📈 Mean performance: {analysis['mean_performance']:.4f} ± {analysis['std_performance']:.4f}")
        print(f"   📊 Best window: {analysis['max_performance']:.4f}")
        print(f"   📉 Worst window: {analysis['min_performance']:.4f}")
        print(f"   📏 Performance range: {analysis['performance_range']:.4f}")
        print(f"   🎯 Stability score: {analysis['stability_score']:.3f}")
        print(f"   📏 Performance CV: {analysis['coefficient_of_variation']:.2%}")
        
        if 'mean_training_time' in analysis:
            print(f"   ⏱️ Avg training time: {analysis['mean_training_time']:.1f}s")
            print(f"   ⏱️ Total training time: {analysis['total_training_time']:.1f}s")
        
        return analysis
    
    def _save_rolling_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], filepath: str):
        """Save rolling window results to file."""
        import json
        from datetime import datetime
        
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_windows': len(results),
                'rolling_window_months': self.training_protocol_config.rolling_window_months,
                'evaluation_period_months': self.training_protocol_config.evaluation_period_months,
                'config_source': 'RollingWindowTrainer'
            },
            'analysis': analysis,
            'window_results': results
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"   💾 Rolling window results saved: {filepath}")
    
    def compare_rolling_configurations(self, 
                                     configurations: List[Dict[str, Any]], 
                                     max_configs: int = 3,
                                     windows_per_config: int = 2) -> Dict[str, Any]:
        """
        Compare multiple configurations using rolling window analysis.
        
        Args:
            configurations: List of configurations to compare
            max_configs: Maximum number of configurations to test
            windows_per_config: Number of rolling windows per configuration
            
        Returns:
            Comparison results
        """
        print(f"🔍 Comparing {min(len(configurations), max_configs)} configurations with rolling windows...")
        
        comparison_results = {}
        
        for i, config in enumerate(configurations[:max_configs]):
            config_id = config.get('config_id', f'config_{i+1}')
            print(f"\n📊 Testing configuration: {config_id}")
            
            try:
                window_results = self.run_rolling_window_diagnostics(
                    config, 
                    max_windows=windows_per_config,
                    save_models=False
                )
                
                successful_results = [r for r in window_results if r.get('status') != 'failed']
                if successful_results:
                    analysis = self._analyze_rolling_performance(successful_results)
                    comparison_results[config_id] = {
                        'config': config,
                        'window_results': window_results,
                        'analysis': analysis,
                        'status': 'success'
                    }
                else:
                    comparison_results[config_id] = {
                        'config': config,
                        'status': 'failed',
                        'error': 'No successful windows'
                    }
                    
            except Exception as e:
                print(f"   ❌ Configuration {config_id} failed: {str(e)}")
                comparison_results[config_id] = {
                    'config': config,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Rank configurations by stability and performance
        successful_configs = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
        
        if successful_configs:
            self._print_configuration_comparison(successful_configs)
            
            # Save comparison results
            comparison_file = os.path.join(self.config.data.output_dir, 'rolling_window_comparison.json')
            self._save_comparison_results(comparison_results, comparison_file)
        
        return comparison_results
    
    def _print_configuration_comparison(self, successful_configs: Dict[str, Dict[str, Any]]):
        """Print comparison table for successful configurations."""
        print(f"\n📊 ROLLING WINDOW CONFIGURATION COMPARISON")
        print("=" * 80)
        print(f"{'Config ID':<20} {'Mean Perf':<12} {'Stability':<12} {'CV':<10} {'Best':<10}")
        print("-" * 80)
        
        # Sort by stability score (higher is better)
        sorted_configs = sorted(
            successful_configs.items(),
            key=lambda x: x[1]['analysis']['stability_score'],
            reverse=True
        )
        
        for config_id, result in sorted_configs:
            analysis = result['analysis']
            print(f"{config_id:<20} {analysis['mean_performance']:<12.4f} "
                  f"{analysis['stability_score']:<12.3f} {analysis['coefficient_of_variation']:<10.2%} "
                  f"{analysis['max_performance']:<10.4f}")
        
        print("=" * 80)
        print("\n🏆 Best performing configuration (by stability):")
        best_config_id, best_result = sorted_configs[0]
        print(f"   🥇 {best_config_id}")
        print(f"   📊 Stability Score: {best_result['analysis']['stability_score']:.3f}")
        print(f"   📈 Mean Performance: {best_result['analysis']['mean_performance']:.4f}")
        print(f"   📏 Performance CV: {best_result['analysis']['coefficient_of_variation']:.2%}")
    
    def _save_comparison_results(self, comparison_results: Dict[str, Any], filepath: str):
        """Save configuration comparison results."""
        import json
        from datetime import datetime
        
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_configurations': len(comparison_results),
                'comparison_source': 'RollingWindowTrainer'
            },
            'comparison_results': comparison_results
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"   💾 Configuration comparison saved: {filepath}")
