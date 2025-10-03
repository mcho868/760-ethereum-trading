# generate_a2c_configs_500.py
# -*- coding: utf-8 -*-


import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

def loguniform(low, high, rng):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="configs/a2c_configs_extreme_500.json",
                        help="输出 JSON 路径")
    parser.add_argument("--n", type=int, default=500, help="配置数量")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== 基准奖励权重（你给的极端版） =====
    BASE_RC = {
        "pnl_scale": 8000.0,
        "pnl_normalization": "nav",
        "transaction_penalty": 0.012,
        "fee_rate": 0.001,
        "slippage": 0.0005,
        "drawdown_threshold": 0.15,
        "drawdown_penalty": 0.10,
        "sharpe_weight": 0.30,
        "sharpe_window": 240,
        "sentiment_reward_weight": 0.0,
        "signal_k": 1.9,
        "big_move_k": 1.20,
        "small_move_k": 0.01,
        "misalign_k": 0.75,
        "pos_size_beta": 0.22,
        "pos_size_gamma": 2.0,
        "edge_start": 0.50,
        "edge_bonus_weight": 0.18,
        "activity_k": 0.50,
        "activity_warmup_steps": 2000,
        "activity_cap": 0.05,
        "activity_rel_cap": 0.90,
        "activity_signal_boost": 0.5,
        "inactivity_eps": 0.005,
        "inactivity_penalty": 0.005,
        "dither_eps": 0.05,
        "dither_penalty": 0.005,
        "underfill_weight": 0.02,
        "action_amp_k": 0.8,
        "underaction_k": 0.35,
        "step_lo": 0.35,
        "step_hi": 1.00,
        "step_momentum_beta": 0.55
    }

    BASE_TRAINING = {
        "total_timesteps": 300_000,
        "train_episode_length": 1440 * 7,
        "eval_episode_length": 10000,
        "rolling_window_months": 12
    }

    BASE_MODEL_PARAMS = {
        "learning_rate": 5e-5,
        "n_steps": 2048,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.2,
        "vf_coef": 0.25,
        "max_grad_norm": 0.7,
        "use_sde": True,
        "sde_sample_freq": 4,
        "policy_kwargs": {
            "log_std_init": 1.0,
            "ortho_init": False,
            "net_arch": [512, 512]
        }
    }

    def jitter(val, rel=0.2):
        return float(val * (1.0 + rng.uniform(-rel, rel)))

    def sample_reward(rc):
        out = dict(rc)
        # 举例：让 PnL scale 在 ±30% 内抖动
        out["pnl_scale"] = clamp(jitter(rc["pnl_scale"], 0.3), 1000, 20000)
        out["signal_k"] = clamp(jitter(rc["signal_k"], 0.3), 0.5, 3.5)
        out["big_move_k"] = clamp(jitter(rc["big_move_k"], 0.5), 0.0, 3.0)
        out["activity_k"] = clamp(jitter(rc["activity_k"], 0.5), 0.01, 2.0)
        out["pos_size_beta"] = clamp(jitter(rc["pos_size_beta"], 0.5), 0.01, 1.0)
        out["action_amp_k"] = clamp(jitter(rc["action_amp_k"], 0.5), 0.0, 2.0)
        # 其它参数可根据需要增加采样逻辑
        return out

    def sample_model_params(mp):
        out = dict(mp)
        out["learning_rate"] = loguniform(1e-5, 1e-3, rng)
        out["n_steps"] = int(rng.choice([512, 1024, 2048, 4096]))
        out["gamma"] = float(rng.uniform(0.98, 0.999))
        out["gae_lambda"] = float(rng.uniform(0.9, 0.99))
        out["ent_coef"] = float(rng.uniform(0.01, 0.3))
        out["vf_coef"] = float(rng.uniform(0.1, 0.5))
        out["max_grad_norm"] = float(rng.uniform(0.3, 1.0))
        return out

    configs = []
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for i in range(args.n):
        cfg_id = f"A2C_extreme_local_{timestamp}_{i:03d}"
        cfg = {
            "config_id": cfg_id,
            "algorithm": "A2C",
            "seed": int(rng.integers(0, 2**31 - 1)),
            "MAX_POSITION_SHIFT": 1.0,
            "reward_components": sample_reward(BASE_RC),
            "training": dict(BASE_TRAINING),
            "model_params": sample_model_params(BASE_MODEL_PARAMS),
            "notes": "Generated around extreme plan for A2C."
        }
        configs.append(cfg)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(configs)} A2C configs -> {out_path}")

if __name__ == "__main__":
    main()
