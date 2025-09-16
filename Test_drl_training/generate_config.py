import json
import numpy as np
import itertools
from datetime import datetime
from collections import Counter

def generate_training_configs(max_configs=1000):
    """Generate N (default 1000) diverse training configurations incl. ENV + A2C hyperparams
       ✅ 对齐 EnhancedTradingEnv(ACTIVE)：去掉 transaction_penalty/position_change_threshold
       ✅ 新增 static_delta_thresh / increment_step_size / pnl_reward_scale / momentum_reward_scale
    """

    # -------- 参数范围（ENV + A2C）--------
    param_ranges = {
        # —— 环境/奖励形状 ——  （与 ACTIVE 版本字段一致）
        'action_reward_scale':      [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
        'zone_reward_multiplier':   [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        'static_penalty':           [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        'momentum_reward_scale':    [10.0, 15.0, 20.0, 25.0],
        'static_delta_thresh':      [0.005, 0.01, 0.015, 0.02],   # 触发“静止惩罚”的动作阈值
        'increment_step_size':      [0.3, 0.4, 0.5, 0.6, 0.7],    # 每步增量映射：action∈[-1,1] -> Δpos ∈ [-k, k]
        'pnl_reward_scale':         [80.0, 100.0, 120.0, 150.0],  # ΔNAV/NAV 的缩放

        # —— A2C 超参 ——
        'learning_rate':            [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002],
        'ent_coef':                 [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'n_steps':                  [256, 512, 1024],
        'gamma':                    [0.95, 0.99],
        'gae_lambda':               [0.90, 0.95, 0.98],
        'vf_coef':                  [0.3, 0.5, 0.7],
        'max_grad_norm':            [0.5, 1.0],

        # 训练时长
        'total_timesteps':          [50000, 75000, 100000, 125000, 150000],
    }

    configurations = []

    # -------- 策略模板（覆盖不同风格）--------
    strategies = [
        {
            'name_prefix': 'ultra_conservative',
            'description': 'Very high static penalties, smaller steps, lower rewards',
            'param_weights': {
                'action_reward_scale':      [20.0, 30.0],
                'zone_reward_multiplier':   [10.0, 15.0],
                'static_penalty':           [30.0, 50.0],
                'momentum_reward_scale':    [10.0, 15.0],
                'static_delta_thresh':      [0.015, 0.02],
                'increment_step_size':      [0.3, 0.4],
                'pnl_reward_scale':         [80.0, 100.0],

                'learning_rate':            [0.0001, 0.0003],
                'ent_coef':                 [0.01, 0.02],
                'n_steps':                  [1024],
                'gamma':                    [0.99],
                'gae_lambda':               [0.95, 0.98],
                'vf_coef':                  [0.5, 0.7],
                'max_grad_norm':            [0.5],

                'total_timesteps':          [100000, 125000, 150000],
            }
        },
        {
            'name_prefix': 'high_reward',
            'description': 'High action rewards with moderate penalties',
            'param_weights': {
                'action_reward_scale':      [60.0, 80.0, 100.0],
                'zone_reward_multiplier':   [25.0, 40.0, 50.0],
                'static_penalty':           [5.0, 10.0, 15.0],
                'momentum_reward_scale':    [15.0, 25.0],
                'static_delta_thresh':      [0.01],
                'increment_step_size':      [0.5, 0.6],
                'pnl_reward_scale':         [120.0, 150.0],

                'learning_rate':            [0.0005, 0.0007, 0.001],
                'ent_coef':                 [0.03, 0.05, 0.07],
                'n_steps':                  [512, 1024],
                'gamma':                    [0.99],
                'gae_lambda':               [0.95],
                'vf_coef':                  [0.5],
                'max_grad_norm':            [0.5, 1.0],

                'total_timesteps':          [75000, 100000],
            }
        },
        {
            'name_prefix': 'balanced',
            'description': 'Balanced approach with moderate parameters',
            'param_weights': {
                'action_reward_scale':      [30.0, 40.0, 50.0],
                'zone_reward_multiplier':   [15.0, 20.0, 25.0],
                'static_penalty':           [10.0, 15.0],
                'momentum_reward_scale':    [10.0, 15.0, 20.0],
                'static_delta_thresh':      [0.01],
                'increment_step_size':      [0.4, 0.5],
                'pnl_reward_scale':         [100.0, 120.0],

                'learning_rate':            [0.0003, 0.0005, 0.0007],
                'ent_coef':                 [0.02, 0.03],
                'n_steps':                  [512, 1024],
                'gamma':                    [0.95, 0.99],
                'gae_lambda':               [0.90, 0.95],
                'vf_coef':                  [0.3, 0.5],
                'max_grad_norm':            [0.5, 1.0],

                'total_timesteps':          [75000, 100000],
            }
        },
        {
            'name_prefix': 'aggressive',
            'description': 'Low penalties, big steps, high rewards',
            'param_weights': {
                'action_reward_scale':      [50.0, 80.0, 100.0],
                'zone_reward_multiplier':   [30.0, 40.0, 50.0],
                'static_penalty':           [0.0, 1.0, 5.0],
                'momentum_reward_scale':    [20.0, 25.0],
                'static_delta_thresh':      [0.005],
                'increment_step_size':      [0.6, 0.7],
                'pnl_reward_scale':         [120.0, 150.0],

                'learning_rate':            [0.0007, 0.001, 0.002],
                'ent_coef':                 [0.05, 0.07, 0.1],
                'n_steps':                  [256, 512],
                'gamma':                    [0.95],
                'gae_lambda':               [0.90, 0.95],
                'vf_coef':                  [0.3, 0.5],
                'max_grad_norm':            [1.0],

                'total_timesteps':          [50000, 75000],
            }
        },
        {
            'name_prefix': 'exploration',
            'description': 'High exploration with varied parameters',
            'param_weights': {
                'action_reward_scale':      [20.0, 40.0, 60.0],
                'zone_reward_multiplier':   [10.0, 20.0, 30.0],
                'static_penalty':           [5.0, 10.0, 20.0],
                'momentum_reward_scale':    [10.0, 20.0, 25.0],
                'static_delta_thresh':      [0.005, 0.02],
                'increment_step_size':      [0.4, 0.5, 0.6],
                'pnl_reward_scale':         [100.0, 120.0, 150.0],

                'learning_rate':            [0.0005, 0.001, 0.002, 0.003],
                'ent_coef':                 [0.05, 0.07, 0.1],
                'n_steps':                  [512, 1024],
                'gamma':                    [0.95, 0.99],
                'gae_lambda':               [0.95, 0.98],
                'vf_coef':                  [0.5, 0.7],
                'max_grad_norm':            [0.5, 1.0],

                'total_timesteps':          [100000, 125000, 150000],
            }
        }
    ]

    configurations = []
    config_id = 1

    # -------- 先按策略遍历小组合，直到凑够 max_configs --------
    for strategy in strategies:
        keys = list(strategy['param_weights'].keys())
        values = [strategy['param_weights'][k] for k in keys]
        for combo in itertools.product(*values):
            if len(configurations) >= max_configs:
                break
            params = dict(zip(keys, combo))

            # 强制类型
            params['n_steps'] = int(params['n_steps'])
            params['total_timesteps'] = int(params['total_timesteps'])
            # 其余统一成 float
            for k, v in list(params.items()):
                if k not in ('n_steps', 'total_timesteps'):
                    params[k] = float(v)

            config = {
                "name": f"{strategy['name_prefix']}_{config_id:04d}",
                "description": f"{strategy['description']} - Config {config_id}",
                "params": params
            }
            configurations.append(config)
            config_id += 1
        if len(configurations) >= max_configs:
            break

    # -------- 还不够则随机补齐 --------
    np.random.seed(42)
    all_keys = list(param_ranges.keys())
    while len(configurations) < max_configs:
        params = {}
        for k in all_keys:
            choice = np.random.choice(param_ranges[k])
            if k in ('n_steps', 'total_timesteps'):
                params[k] = int(choice)
            else:
                params[k] = float(choice)

        config = {
            "name": f"random_{config_id:04d}",
            "description": f"Random parameter combination - Config {config_id}",
            "params": params
        }
        configurations.append(config)
        config_id += 1

    # -------- 输出 JSON --------
    config_data = {
        "metadata": {
            "total_configurations": len(configurations),
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "description": f"{len(configurations)} training configurations for parameter sweep (ENV + A2C)",
            "strategies": [s['name_prefix'] for s in strategies],
            "parameter_ranges": param_ranges
        },
        "training_configurations": configurations
    }
    return config_data


# 运行并保存
print("🔄 Generating 1000 training configurations...")
config_data = generate_training_configs(max_configs=1000)

with open('training_config.json', 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"✅ Generated {len(config_data['training_configurations'])} configurations")
print(f"💾 Saved to: training_config.json")
print(f"📊 File size: ~{len(json.dumps(config_data)) / 1024:.1f} KB")

print("\n📈 CONFIGURATION SUMMARY")
print("=" * 50)
c = Counter(cfg['name'].split('_')[0] for cfg in config_data['training_configurations'])
for k, v in c.items():
    print(f"   {k}: {v} configurations")