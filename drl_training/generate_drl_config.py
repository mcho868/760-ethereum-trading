#!/usr/bin/env python3
"""
Hyperparameter Configuration Generator for DRL Ethereum Trading
================================================================

This script generates 1000 unique hyperparameter combinations for fine-tuning
the DRL trading agent's reward function and model parameters.
"""

import json
import random
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_drl_configs(num_configs: int = 1000):
    """Generate 1000 hyperparameter configurations for reward function optimization."""
    
    # Reward function parameters
    pnl_reward_scales = [50.0, 80.0, 100.0, 120.0, 150.0, 200.0]
    sharpe_reward_weights = [0.0, 0.1, 0.2, 0.3, 0.5]
    transaction_penalty_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
    drawdown_thresholds = [0.05, 0.10, 0.15, 0.20]
    drawdown_penalties = [10.0, 20.0, 50.0, 100.0, 200.0]
    holding_penalties = [0.0, 0.001, 0.005, 0.01, 0.02]
    activity_reward_weights = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    # Model hyperparameters
    a2c_learning_rates = [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    a2c_n_steps = [512, 1024, 2048, 4096]
    a2c_gamma = [0.95, 0.99, 0.995]
    a2c_gae_lambda = [0.9, 0.95, 0.98]
    a2c_ent_coef = [0.001, 0.01, 0.02, 0.05]
    
    # Training parameters
    training_timesteps = [50000, 100000, 150000, 200000, 300000]
    episode_lengths = [40320]  # Fixed to 28 days
    rolling_window_months = [6, 9, 12]
    max_position_changes = [0.05, 0.1, 0.15, 0.2]
    
    configurations = []
    
    for i in range(num_configs):
        algorithm = random.choice(['A2C', 'TD3'])
        
        config = {
            'config_id': f'{algorithm.lower()}_{i+1:04d}',
            'algorithm': algorithm,
            'description': f'{algorithm} configuration {i+1} for reward function optimization',
            'reward_components': {
                'pnl_reward_scale': random.choice(pnl_reward_scales),
                'sharpe_reward_weight': random.choice(sharpe_reward_weights),
                'transaction_penalty_multiplier': random.choice(transaction_penalty_multipliers),
                'drawdown_threshold': random.choice(drawdown_thresholds),
                'drawdown_penalty': random.choice(drawdown_penalties),
                'holding_penalty': random.choice(holding_penalties),
                'activity_reward_weight': random.choice(activity_reward_weights)
            },
            'training': {
                'total_timesteps': random.choice(training_timesteps),
                'episode_length': random.choice(episode_lengths),
                'rolling_window_months': random.choice(rolling_window_months)
            },
            'risk_management': {
                'max_position_change': random.choice(max_position_changes),
                'max_drawdown_limit': random.choice([0.15, 0.20, 0.25, 0.30]),
                'position_size_limit': random.choice([0.8, 1.0, 1.2])
            }
        }
        
        if algorithm == 'A2C':
            config['model_params'] = {
                'learning_rate': random.choice(a2c_learning_rates),
                'n_steps': random.choice(a2c_n_steps),
                'gamma': random.choice(a2c_gamma),
                'gae_lambda': random.choice(a2c_gae_lambda),
                'ent_coef': random.choice(a2c_ent_coef),
                'vf_coef': random.choice([0.25, 0.5, 0.75]),
                'max_grad_norm': random.choice([0.1, 0.5, 1.0])
            }
        else:  # TD3
            config['model_params'] = {
                'learning_rate': random.choice([1e-5, 5e-5, 1e-4, 3e-4, 5e-4]),
                'buffer_size': random.choice([100000, 500000, 1000000]),
                'batch_size': random.choice([64, 128, 256, 512]),
                'tau': random.choice([0.001, 0.005, 0.01]),
                'gamma': random.choice(a2c_gamma),
                'noise_std': random.choice([0.05, 0.1, 0.2]),
                'target_noise': random.choice([0.1, 0.2, 0.3]),
                'policy_delay': random.choice([1, 2, 3])
            }
        
        configurations.append(config)
    
    output = {
        'metadata': {
            'total_configurations': len(configurations),
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'description': 'DRL hyperparameter configurations for reward function optimization',
            'methodology_version': '15D_state_space_v1.0',
            'algorithms': ['A2C', 'TD3']
        },
        'configurations': configurations
    }
    
    return output

if __name__ == "__main__":
    print("🎯 Generating DRL Training Configurations for Reward Function Optimization")
    print("=" * 80)
    
    configs = generate_drl_configs(1000)
    
    with open('drl_training_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    
    algorithms = [c['algorithm'] for c in configs['configurations']]
    a2c_count = algorithms.count('A2C')
    td3_count = algorithms.count('TD3')
    
    print(f"✅ Generated {len(configs['configurations'])} configurations")
    print(f"💾 Saved to: drl_training_configs.json")
    print(f"📊 Algorithm distribution:")
    print(f"   A2C: {a2c_count} configs ({a2c_count/len(algorithms)*100:.1f}%)")
    print(f"   TD3: {td3_count} configs ({td3_count/len(algorithms)*100:.1f}%)")
    print(f"\n🎉 Configuration generation complete!")
    print(f"⚡ Covers reward function fine-tuning across 1000 combinations")
