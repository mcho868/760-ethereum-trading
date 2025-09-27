"""
Trading Orchestrator for DRL Trading System

This module provides a high-level orchestrator that coordinates all components
of the Deep Reinforcement Learning trading system for end-to-end workflows.

Author: DRL Trading Team
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import warnings

from .config_manager import ConfigManager
from .data_processor import DataProcessor
from .trading_environment import TradingEnvironment
from .model_trainer import ModelTrainer
from .performance_analyzer import PerformanceAnalyzer
from .hyperparameter_optimizer import HyperparameterOptimizer


class TradingOrchestrator:
    """
    High-level orchestrator for the DRL trading system.
    
    This class provides a unified interface for:
    - End-to-end pipeline execution
    - Component coordination and workflow management
    - Configuration management and validation
    - Data processing and feature engineering
    - Model training and hyperparameter optimization
    - Performance analysis and visualization
    - Results export and reporting
    - Pipeline persistence and resume capabilities
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 environment: str = 'default'):
        """
        Initialize the Trading Orchestrator.
        
        Args:
            config_path: Path to configuration file
            workspace_dir: Workspace directory for outputs
            environment: Environment configuration name
        """
        print("🚀 Initializing DRL Trading Orchestrator...")
        
        # Initialize configuration
        self.config = ConfigManager(config_path, environment)
        
        # Set up workspace
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
            self.config.data.output_dir = str(self.workspace_dir / 'outputs')
            self.config.data.model_dir = str(self.workspace_dir / 'models')
        else:
            self.workspace_dir = Path(self.config.data.output_dir).parent
        
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy initialization)
        self.data_processor = None
        self.model_trainer = None
        self.performance_analyzer = None
        self.hyperparameter_optimizer = None
        
        # Pipeline state
        self.pipeline_state = {
            'data_processed': False,
            'models_trained': False,
            'optimization_completed': False,
            'analysis_completed': False
        }
        
        # Results storage
        self.processed_data = None
        self.feature_columns = None
        self.data_splits = None
        self.trained_models = {}
        self.optimization_results = []
        self.analysis_results = {}
        
        print(f"   ✅ Orchestrator initialized")
        print(f"   📁 Workspace: {self.workspace_dir}")
        print(f"   🔧 Environment: {environment}")
        
        # Validate configuration
        if self.config.validate_configuration():
            print(f"   ✅ Configuration validated")
        else:
            print(f"   ⚠️ Configuration validation failed - some features may not work correctly")
    
    def run_complete_pipeline(self,
                            data_path: Optional[str] = None,
                            optimization_strategy: str = 'random_search',
                            n_optimization_trials: int = 50,
                            n_best_models: int = 5,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete DRL trading pipeline end-to-end.
        
        Args:
            data_path: Path to raw data file
            optimization_strategy: Optimization strategy ('grid_search', 'random_search', 'bayesian')
            n_optimization_trials: Number of optimization trials
            n_best_models: Number of best models to analyze
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with pipeline results
        """
        print("🔄 Running complete DRL trading pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Processing
            print("\n📊 Step 1: Data Processing and Feature Engineering")
            self.process_data(data_path)
            
            # Step 2: Hyperparameter Optimization
            print(f"\n🔍 Step 2: Hyperparameter Optimization ({optimization_strategy})")
            self.run_hyperparameter_optimization(
                strategy=optimization_strategy,
                n_trials=n_optimization_trials
            )
            
            # Step 3: Train Best Models
            print(f"\n🧠 Step 3: Training Best {n_best_models} Models")
            self.train_best_models(n_best_models)
            
            # Step 4: Performance Analysis
            print(f"\n📈 Step 4: Performance Analysis and Visualization")
            self.analyze_performance(create_visualizations=True)
            
            # Step 5: Generate Report
            print(f"\n📄 Step 5: Generating Final Report")
            report = self.generate_pipeline_report()
            
            # Save results if requested
            if save_results:
                self.save_pipeline_results()
            
            execution_time = datetime.now() - start_time
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   ⏱️ Total execution time: {execution_time}")
            
            return {
                'status': 'success',
                'execution_time': execution_time,
                'report': report,
                'workspace_dir': str(self.workspace_dir)
            }
            
        except Exception as e:
            execution_time = datetime.now() - start_time
            print(f"\n❌ Pipeline failed after {execution_time}")
            print(f"   Error: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': execution_time
            }
    
    def process_data(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Process data and engineer features.
        
        Args:
            data_path: Path to raw data file
            
        Returns:
            Tuple of (processed_data, feature_columns, data_splits)
        """
        if not self.data_processor:
            self.data_processor = DataProcessor(self.config)
        
        # Run data processing pipeline
        self.processed_data, self.feature_columns, self.data_splits = \
            self.data_processor.run_full_pipeline(data_path)
        
        self.pipeline_state['data_processed'] = True
        
        print(f"   ✅ Data processing completed")
        print(f"   📊 Processed data shape: {self.processed_data.shape}")
        print(f"   🎯 Features: {len(self.feature_columns)} dimensions")
        
        return self.processed_data, self.feature_columns, self.data_splits
    
    def setup_model_trainer(self) -> ModelTrainer:
        """Setup and return model trainer."""
        if not self.data_splits:
            raise RuntimeError("Data must be processed before setting up model trainer")
        
        if not self.model_trainer:
            self.model_trainer = ModelTrainer(
                train_data=self.data_splits['train'],
                val_data=self.data_splits['validation'],
                test_data=self.data_splits['test'],
                feature_columns=self.feature_columns,
                config=self.config
            )
        
        return self.model_trainer
    
    def run_hyperparameter_optimization(self,
                                       strategy: str = 'random_search',
                                       n_trials: int = 50,
                                       algorithms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run hyperparameter optimization.
        
        Args:
            strategy: Optimization strategy ('grid_search', 'random_search', 'bayesian')
            n_trials: Number of trials/configurations to test
            algorithms: List of algorithms to optimize
            
        Returns:
            List of optimization results
        """
        # Setup components
        trainer = self.setup_model_trainer()
        
        if not self.hyperparameter_optimizer:
            self.hyperparameter_optimizer = HyperparameterOptimizer(trainer, self.config)
        
        algorithms = algorithms or ['A2C', 'TD3']
        
        if strategy == 'random_search':
            # Define parameter distributions for random search
            param_distributions = {
                'learning_rate': (1e-5, 1e-2, 'log'),
                'reward_pnl_scale': (50.0, 200.0),
                'reward_sharpe_weight': (0.1, 0.5),
                'reward_transaction_penalty': (0.5, 2.0)
            }
            
            results = []
            for algorithm in algorithms:
                algo_results = self.hyperparameter_optimizer.random_search_optimization(
                    param_distributions=param_distributions,
                    algorithm=algorithm,
                    n_configs=n_trials // len(algorithms)
                )
                results.extend(algo_results)
        
        elif strategy == 'grid_search':
            # Define parameter grid for grid search
            param_grid = {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'reward_pnl_scale': [50.0, 100.0, 150.0],
                'reward_sharpe_weight': [0.1, 0.2, 0.3]
            }
            
            results = []
            for algorithm in algorithms:
                algo_results = self.hyperparameter_optimizer.grid_search_optimization(
                    param_grid=param_grid,
                    algorithm=algorithm,
                    max_configs=n_trials // len(algorithms)
                )
                results.extend(algo_results)
        
        elif strategy == 'bayesian':
            try:
                # Define parameter space for Bayesian optimization
                param_space = {
                    'learning_rate': {'type': 'log_float', 'low': 1e-5, 'high': 1e-2},
                    'reward_pnl_scale': {'type': 'float', 'low': 50.0, 'high': 200.0},
                    'reward_sharpe_weight': {'type': 'float', 'low': 0.1, 'high': 0.5}
                }
                
                results = []
                for algorithm in algorithms:
                    algo_results = self.hyperparameter_optimizer.bayesian_optimization(
                        param_space=param_space,
                        algorithm=algorithm,
                        n_trials=n_trials // len(algorithms)
                    )
                    results.extend(algo_results)
            except ImportError:
                print("   ⚠️ Bayesian optimization requires Optuna. Falling back to random search.")
                return self.run_hyperparameter_optimization('random_search', n_trials, algorithms)
        
        else:
            raise ValueError(f"Unsupported optimization strategy: {strategy}")
        
        self.optimization_results = results
        self.pipeline_state['optimization_completed'] = True
        
        print(f"   ✅ Hyperparameter optimization completed")
        print(f"   🔍 Tested configurations: {len(results)}")
        
        return results
    
    def train_best_models(self, n_best: int = 5) -> Dict[str, Any]:
        """
        Train the best models from optimization results.
        
        Args:
            n_best: Number of best configurations to train
            
        Returns:
            Dictionary of trained models
        """
        if not self.optimization_results:
            raise RuntimeError("Hyperparameter optimization must be completed first")
        
        # Get best configurations
        best_configs = self.hyperparameter_optimizer.get_best_configurations(n_best)
        
        if not best_configs:
            raise RuntimeError("No successful optimization results found")
        
        trainer = self.setup_model_trainer()
        trained_models = {}
        
        for i, result in enumerate(best_configs):
            config = result.get('config', {})
            config_id = config.get('config_id', f'best_model_{i}')
            algorithm = config.get('algorithm', 'A2C')
            
            print(f"   🧠 Training model {i+1}/{n_best}: {config_id}")
            
            try:
                if algorithm.upper() == 'A2C':
                    model, metrics = trainer.train_a2c_model(config, save_model=True)
                elif algorithm.upper() == 'TD3':
                    model, metrics = trainer.train_td3_model(config, save_model=True)
                else:
                    print(f"      ⚠️ Unsupported algorithm: {algorithm}")
                    continue
                
                trained_models[config_id] = {
                    'model': model,
                    'config': config,
                    'metrics': metrics,
                    'optimization_result': result
                }
                
                print(f"      ✅ Training completed: {metrics.get('mean_reward', 0):.4f}")
                
            except Exception as e:
                print(f"      ❌ Training failed: {str(e)}")
        
        self.trained_models = trained_models
        self.pipeline_state['models_trained'] = True
        
        print(f"   ✅ Model training completed")
        print(f"   🧠 Successfully trained: {len(trained_models)} models")
        
        return trained_models
    
    def analyze_performance(self, create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Analyze performance of trained models.
        
        Args:
            create_visualizations: Whether to create visualization plots
            
        Returns:
            Dictionary of analysis results
        """
        if not self.trained_models:
            raise RuntimeError("Models must be trained before performance analysis")
        
        if not self.performance_analyzer:
            self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        analysis_results = {}
        
        # Analyze each trained model
        for model_id, model_data in self.trained_models.items():
            print(f"   📊 Analyzing model: {model_id}")
            
            try:
                # Create test environment
                test_env = TradingEnvironment(
                    data=self.data_splits['test'],
                    feature_columns=self.feature_columns,
                    config=self.config,
                    reward_config=model_data['config'].get('reward_components'),
                    random_start=False
                )
                
                # Analyze model performance
                analysis = self.performance_analyzer.analyze_model_performance(
                    model=model_data['model'],
                    env=test_env,
                    n_episodes=5,
                    config_info={'config_id': model_id, **model_data['config']}
                )
                
                analysis_results[model_id] = analysis
                
                # Create individual visualizations if requested
                if create_visualizations:
                    self.performance_analyzer.create_performance_plots(
                        analysis, save_plots=True, show_plots=False
                    )
                
                print(f"      ✅ Analysis completed")
                
            except Exception as e:
                print(f"      ❌ Analysis failed: {str(e)}")
        
        # Create comparison dashboard if multiple models
        if len(analysis_results) > 1 and create_visualizations:
            print(f"   📊 Creating model comparison dashboard...")
            self.performance_analyzer.create_model_comparison_dashboard(
                analysis_results, save_plots=True, show_plots=False
            )
        
        self.analysis_results = analysis_results
        self.pipeline_state['analysis_completed'] = True
        
        print(f"   ✅ Performance analysis completed")
        print(f"   📈 Analyzed models: {len(analysis_results)}")
        
        return analysis_results
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        print("   📄 Generating comprehensive pipeline report...")
        
        # Basic pipeline information
        report = {
            'pipeline_metadata': {
                'execution_timestamp': datetime.now().isoformat(),
                'workspace_dir': str(self.workspace_dir),
                'configuration': self.config.to_dict(),
                'pipeline_state': self.pipeline_state
            },
            'data_summary': {},
            'optimization_summary': {},
            'model_summary': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # Data summary
        if self.processed_data is not None:
            report['data_summary'] = {
                'total_rows': len(self.processed_data),
                'feature_count': len(self.feature_columns),
                'train_rows': len(self.data_splits['train']) if self.data_splits else 0,
                'validation_rows': len(self.data_splits['validation']) if self.data_splits else 0,
                'test_rows': len(self.data_splits['test']) if self.data_splits else 0,
                'feature_columns': self.feature_columns
            }
        
        # Optimization summary
        if self.optimization_results:
            successful_results = [r for r in self.optimization_results if 'error' not in r]
            
            if successful_results:
                performance_values = [
                    r.get(self.hyperparameter_optimizer.performance_metric, 0) 
                    for r in successful_results
                ]
                
                report['optimization_summary'] = {
                    'total_configurations': len(self.optimization_results),
                    'successful_configurations': len(successful_results),
                    'success_rate': len(successful_results) / len(self.optimization_results),
                    'performance_metric': self.hyperparameter_optimizer.performance_metric,
                    'best_performance': max(performance_values) if performance_values else 0,
                    'mean_performance': np.mean(performance_values) if performance_values else 0,
                    'std_performance': np.std(performance_values) if performance_values else 0
                }
        
        # Model summary
        if self.trained_models:
            model_metrics = []
            for model_id, model_data in self.trained_models.items():
                metrics = model_data.get('metrics', {})
                model_metrics.append({
                    'model_id': model_id,
                    'algorithm': model_data['config'].get('algorithm', 'Unknown'),
                    'mean_reward': metrics.get('mean_reward', 0),
                    'training_time': metrics.get('training_time', 0)
                })
            
            report['model_summary'] = {
                'total_models_trained': len(self.trained_models),
                'model_metrics': model_metrics,
                'best_model': max(model_metrics, key=lambda x: x['mean_reward']) if model_metrics else None
            }
        
        # Performance summary
        if self.analysis_results:
            performance_summary = {}
            for model_id, analysis in self.analysis_results.items():
                aggregate_metrics = analysis.get('aggregate_metrics', {})
                performance_summary[model_id] = {
                    'total_return': aggregate_metrics.get('mean_total_return', 0),
                    'sharpe_ratio': aggregate_metrics.get('mean_sharpe_ratio', 0),
                    'max_drawdown': aggregate_metrics.get('mean_max_drawdown', 0),
                    'win_rate': aggregate_metrics.get('mean_win_rate', 0)
                }
            
            report['performance_summary'] = performance_summary
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        print(f"      ✅ Pipeline report generated")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        # Data recommendations
        if self.processed_data is not None and len(self.processed_data) < 10000:
            recommendations.append("Consider using more training data for better model performance")
        
        # Optimization recommendations
        if self.optimization_results:
            successful_rate = len([r for r in self.optimization_results if 'error' not in r]) / len(self.optimization_results)
            if successful_rate < 0.8:
                recommendations.append("High failure rate in optimization - consider reviewing parameter ranges")
        
        # Model recommendations
        if self.trained_models:
            algorithms = [m['config'].get('algorithm') for m in self.trained_models.values()]
            if len(set(algorithms)) == 1:
                recommendations.append("Consider testing multiple algorithms (A2C, TD3) for comparison")
        
        # Performance recommendations
        if self.analysis_results:
            avg_sharpe = np.mean([
                analysis.get('aggregate_metrics', {}).get('mean_sharpe_ratio', 0)
                for analysis in self.analysis_results.values()
            ])
            if avg_sharpe < 0.5:
                recommendations.append("Low Sharpe ratios detected - consider tuning reward function parameters")
        
        if not recommendations:
            recommendations.append("Pipeline completed successfully with good results")
        
        return recommendations
    
    def save_pipeline_results(self) -> Dict[str, str]:
        """Save all pipeline results to files."""
        print("   💾 Saving pipeline results...")
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save configuration
        config_file = self.workspace_dir / f'pipeline_config_{timestamp}.json'
        self.config.save_to_file(str(config_file))
        saved_files['config'] = str(config_file)
        
        # Save optimization results
        if self.optimization_results:
            opt_file = self.workspace_dir / f'optimization_results_{timestamp}.json'
            with open(opt_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            saved_files['optimization'] = str(opt_file)
        
        # Save model metrics
        if self.trained_models:
            models_file = self.workspace_dir / f'model_metrics_{timestamp}.json'
            model_data = {
                model_id: {
                    'config': data['config'],
                    'metrics': data['metrics']
                }
                for model_id, data in self.trained_models.items()
            }
            with open(models_file, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            saved_files['models'] = str(models_file)
        
        # Save analysis results
        if self.analysis_results:
            for model_id, analysis in self.analysis_results.items():
                analysis_file = self.workspace_dir / f'analysis_{model_id}_{timestamp}.json'
                self.performance_analyzer.export_analysis_report(analysis, 'json')
                saved_files[f'analysis_{model_id}'] = str(analysis_file)
        
        # Save pipeline report
        report = self.generate_pipeline_report()
        report_file = self.workspace_dir / f'pipeline_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        saved_files['report'] = str(report_file)
        
        print(f"      ✅ Results saved to {len(saved_files)} files")
        return saved_files
    
    def load_pipeline_state(self, state_file: str) -> bool:
        """Load pipeline state from file."""
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore pipeline state
            self.pipeline_state = state_data.get('pipeline_state', {})
            
            # TODO: Implement full state restoration
            print(f"   ✅ Pipeline state loaded from: {state_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load pipeline state: {str(e)}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'pipeline_state': self.pipeline_state,
            'components_initialized': {
                'data_processor': self.data_processor is not None,
                'model_trainer': self.model_trainer is not None,
                'performance_analyzer': self.performance_analyzer is not None,
                'hyperparameter_optimizer': self.hyperparameter_optimizer is not None
            },
            'data_status': {
                'processed_data_available': self.processed_data is not None,
                'data_splits_available': self.data_splits is not None,
                'feature_columns_count': len(self.feature_columns) if self.feature_columns else 0
            },
            'results_status': {
                'optimization_results_count': len(self.optimization_results),
                'trained_models_count': len(self.trained_models),
                'analysis_results_count': len(self.analysis_results)
            },
            'workspace_dir': str(self.workspace_dir)
        }


def create_orchestrator(config_path: Optional[str] = None,
                       workspace_dir: Optional[str] = None,
                       environment: str = 'default') -> TradingOrchestrator:
    """
    Convenience function to create a TradingOrchestrator.
    
    Args:
        config_path: Path to configuration file
        workspace_dir: Workspace directory
        environment: Environment name
        
    Returns:
        TradingOrchestrator instance
    """
    return TradingOrchestrator(config_path, workspace_dir, environment)


if __name__ == "__main__":
    # Example usage
    orchestrator = create_orchestrator()
    
    # Print status
    status = orchestrator.get_pipeline_status()
    print(f"Pipeline Status: {status}")
    
    # Example: Run mini pipeline (commented out for safety)
    # results = orchestrator.run_complete_pipeline(
    #     data_path='../ETHUSDT_1m_with_indicators.parquet',
    #     optimization_strategy='random_search',
    #     n_optimization_trials=10,
    #     n_best_models=2
    # )
    
    print("TradingOrchestrator example completed")
