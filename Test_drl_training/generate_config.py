import json
import numpy as np
import itertools

def generate_1000_training_configs():
    """Generate 1000 diverse training configurations with systematic parameter sweeps"""
    
    # Define parameter ranges for systematic exploration
    param_ranges = {
        'transaction_penalty': [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
        'action_reward_scale': [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
        'zone_reward_multiplier': [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        'static_penalty': [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        'learning_rate': [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003],
        'total_timesteps': [50000, 75000, 100000, 125000, 150000],
        'ent_coef': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    }
    
    configurations = []
    
    # Generate base configurations with different strategies
    strategies = [
        {
            'name_prefix': 'ultra_conservative',
            'description': 'Very high penalties, low rewards - extreme anti-static',
            'param_weights': {
                'transaction_penalty': [0.02, 0.03, 0.05, 0.1],
                'action_reward_scale': [5.0, 10.0, 20.0],
                'zone_reward_multiplier': [1.0, 5.0, 10.0],
                'static_penalty': [20.0, 30.0, 50.0],
                'learning_rate': [0.0001, 0.0003],
                'total_timesteps': [100000, 125000, 150000],
                'ent_coef': [0.01, 0.02]
            }
        },
        {
            'name_prefix': 'high_reward',
            'description': 'High action rewards with moderate penalties',
            'param_weights': {
                'transaction_penalty': [0.001, 0.005, 0.01],
                'action_reward_scale': [60.0, 80.0, 100.0],
                'zone_reward_multiplier': [25.0, 30.0, 40.0, 50.0],
                'static_penalty': [5.0, 10.0, 15.0],
                'learning_rate': [0.0005, 0.0007, 0.001],
                'total_timesteps': [75000, 100000],
                'ent_coef': [0.03, 0.05, 0.07]
            }
        },
        {
            'name_prefix': 'balanced',
            'description': 'Balanced approach with moderate parameters',
            'param_weights': {
                'transaction_penalty': [0.005, 0.01, 0.02],
                'action_reward_scale': [30.0, 40.0, 50.0],
                'zone_reward_multiplier': [15.0, 20.0, 25.0],
                'static_penalty': [10.0, 15.0],
                'learning_rate': [0.0003, 0.0005, 0.0007],
                'total_timesteps': [75000, 100000],
                'ent_coef': [0.02, 0.03]
            }
        },
        {
            'name_prefix': 'aggressive',
            'description': 'Low penalties, high rewards - encourage active trading',
            'param_weights': {
                'transaction_penalty': [0.00001, 0.0001, 0.001],
                'action_reward_scale': [50.0, 60.0, 80.0, 100.0],
                'zone_reward_multiplier': [30.0, 40.0, 50.0],
                'static_penalty': [0.0, 1.0, 5.0],
                'learning_rate': [0.0007, 0.001, 0.002],
                'total_timesteps': [50000, 75000],
                'ent_coef': [0.05, 0.07, 0.1]
            }
        },
        {
            'name_prefix': 'exploration',
            'description': 'High exploration with varied parameters',
            'param_weights': {
                'transaction_penalty': [0.001, 0.005, 0.01, 0.02],
                'action_reward_scale': [20.0, 40.0, 60.0],
                'zone_reward_multiplier': [10.0, 20.0, 30.0],
                'static_penalty': [5.0, 10.0, 20.0],
                'learning_rate': [0.0005, 0.001, 0.002, 0.003],
                'total_timesteps': [100000, 125000, 150000],
                'ent_coef': [0.05, 0.07, 0.1]
            }
        }
    ]
    
    config_id = 1
    
    # Generate configurations for each strategy
    for strategy in strategies:
        strategy_configs = []
        
        # Generate all combinations for this strategy
        param_combinations = itertools.product(
            strategy['param_weights']['transaction_penalty'],
            strategy['param_weights']['action_reward_scale'],
            strategy['param_weights']['zone_reward_multiplier'],
            strategy['param_weights']['static_penalty'],
            strategy['param_weights']['learning_rate'],
            strategy['param_weights']['total_timesteps'],
            strategy['param_weights']['ent_coef']
        )
        
        for combo in param_combinations:
            if config_id > 1000:
                break
                
            config = {
                "name": f"{strategy['name_prefix']}_{config_id:04d}",
                "description": f"{strategy['description']} - Config {config_id}",
                "params": {
                    "transaction_penalty": combo[0],
                    "action_reward_scale": combo[1],
                    "zone_reward_multiplier": combo[2],
                    "static_penalty": combo[3],
                    "learning_rate": combo[4],
                    "total_timesteps": int(combo[5]),
                    "ent_coef": combo[6]
                }
            }
            
            configurations.append(config)
            config_id += 1
    
    # Fill remaining slots with random combinations if needed
    np.random.seed(42)  # For reproducible results
    
    while len(configurations) < 1000:
        config = {
            "name": f"random_{config_id:04d}",
            "description": f"Random parameter combination - Config {config_id}",
            "params": {
                "transaction_penalty": float(np.random.choice(param_ranges['transaction_penalty'])),
                "action_reward_scale": float(np.random.choice(param_ranges['action_reward_scale'])),
                "zone_reward_multiplier": float(np.random.choice(param_ranges['zone_reward_multiplier'])),
                "static_penalty": float(np.random.choice(param_ranges['static_penalty'])),
                "learning_rate": float(np.random.choice(param_ranges['learning_rate'])),
                "total_timesteps": int(np.random.choice(param_ranges['total_timesteps'])),
                "ent_coef": float(np.random.choice(param_ranges['ent_coef']))
            }
        }
        
        configurations.append(config)
        config_id += 1
    
    # Create the final JSON structure
    config_data = {
        "metadata": {
            "total_configurations": len(configurations),
            "generation_date": "2025-01-08",
            "description": "1000 diverse training configurations for comprehensive parameter sweep",
            "strategies": [s['name_prefix'] for s in strategies],
            "parameter_ranges": param_ranges
        },
        "training_configurations": configurations
    }
    
    return config_data

# Generate and save the configurations
print("🔄 Generating 1000 training configurations...")
config_data = generate_1000_training_configs()

# Save to JSON file
with open('training_config.json', 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"✅ Generated {len(config_data['training_configurations'])} configurations")
print(f"💾 Saved to: training_configs.json")
print(f"📊 File size: ~{len(json.dumps(config_data)) / 1024:.1f} KB")

# Print summary statistics
print(f"\n📈 CONFIGURATION SUMMARY:")
print("=" * 50)

strategies_count = {}
for config in config_data['training_configurations']:
    strategy = config['name'].split('_')[0]
    strategies_count[strategy] = strategies_count.get(strategy, 0) + 1

for strategy, count in strategies_count.items():
    print(f"   {strategy}: {count} configurations")

print(f"\n🎯 PARAMETER RANGES COVERED:")
for param, ranges in config_data['metadata']['parameter_ranges'].items():
    print(f"   {param}: {len(ranges)} values from {min(ranges)} to {max(ranges)}")