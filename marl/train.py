"""
MAPPO Training Script for Disaster Response Coordination System (FR4, FR5, NFR3, NFR7).
Directly trains MAPPO using PettingZoo env with CTDE, MongoDB logging, and resume support.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any, Tuple

from env.disaster_env import DisasterResponseEnv
from marl.mappo import MAPPOTrainer
from marl.buffer import MAPPOBuffer
from metrics_logging.db_logger import DBLogger

def construct_global_state(obs_dict: Dict[str, np.ndarray], possible_agents: list) -> np.ndarray:
    """Concatenates all agents' local observations into a 1D global state vector."""
    obs_list = [obs_dict[agent_id] for agent_id in possible_agents if agent_id in obs_dict]
    return np.concatenate(obs_list, axis=0).astype(np.float32)

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_mappo(
    config_path: str = "config/default_config.yaml",
    timesteps_override: int = None,
    resume: bool = False,
    smoke: bool = False
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if smoke:
        config["training"]["total_timesteps"] = 2000
        config["training"]["rollout_length"] = 32
        config["training"]["batch_size"] = 16
        config["training"]["checkpoint_freq"] = 5

    if timesteps_override is not None:
        config["training"]["total_timesteps"] = timesteps_override

    seed = config["environment"].get("seed", 42)
    set_seeds(seed)

    db_cfg = config.get("db", {})
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment
    env = DisasterResponseEnv(config)
    possible_agents = env.possible_agents
    
    # Reset env once to infer observation and global state shapes
    obs_dict, _ = env.reset(seed=seed)
    obs_dims = {agent_id: env.observation_space(agent_id).shape[0] for agent_id in possible_agents}
    action_dims = {agent_id: env.action_space(agent_id).n for agent_id in possible_agents}
    
    dummy_global_state = construct_global_state(obs_dict, possible_agents)
    global_state_dim = dummy_global_state.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MAPPO Trainer & Buffer
    trainer = MAPPOTrainer(config, possible_agents, obs_dims, global_state_dim, action_dims, device=device)
    rollout_length = config["training"].get("rollout_length", 128)
    buffer = MAPPOBuffer(rollout_length, possible_agents, obs_dims, global_state_dim, device=device)

    # DB Logger setup
    db = DBLogger(db_cfg)
    run_id = db.start_run(run_type="marl_mappo", config=config, notes="MAPPO CTDE Multi-Agent Training")

    start_update = 0
    total_timesteps = config["training"].get("total_timesteps", 200000)
    total_updates = total_timesteps // rollout_length
    best_reward = -float("inf")

    latest_ckpt_path = os.path.join(checkpoint_dir, "mappo_latest.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "mappo_best.pt")

    # Resume from checkpoint if requested
    if resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming MAPPO training from {latest_ckpt_path}...")
        extra = trainer.load_checkpoint(latest_ckpt_path)
        start_update = extra.get("update", 0) + 1
        best_reward = extra.get("best_reward", -float("inf"))
        if "rng_state" in extra:
            torch.set_rng_state(extra["rng_state"])
        print(f"Resumed successfully at update {start_update}/{total_updates}.")

    print(f"Starting MAPPO CTDE Training: {total_timesteps} timesteps ({total_updates} updates) [Run ID: {run_id}]...")

    curr_obs, _ = env.reset(seed=seed)
    curr_global_state = construct_global_state(curr_obs, possible_agents)
    
    episode_num = 0
    ep_reward = {a: 0.0 for a in possible_agents}
    ep_length = 0

    for update in range(start_update, total_updates):
        buffer.reset()

        for step in range(rollout_length):
            actions, log_probs = trainer.select_actions(curr_obs)
            val = trainer.get_value(curr_global_state)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            next_global_state = construct_global_state(next_obs if next_obs else curr_obs, possible_agents)
            done = any(terminations.values()) or any(truncations.values()) or len(env.agents) == 0

            buffer.insert(curr_obs, curr_global_state, actions, log_probs, rewards, done, val)

            for a, r in rewards.items():
                ep_reward[a] += r
            ep_length += 1

            if done:
                episode_num += 1
                state_data = env.state_data or {}
                render_meta = env.render_state()
                metrics = render_meta["metrics"]

                role_rewards = {"medical": 0.0, "rescue": 0.0, "logistics": 0.0}
                for a_id, r_val in ep_reward.items():
                    role = a_id.split("_")[0]
                    role_rewards[role] += r_val

                metrics["total_reward"] = sum(ep_reward.values())
                metrics["role_rewards"] = role_rewards

                db.log_episode(run_id, episode_num, metrics)

                # Reset episode tracking
                curr_obs, _ = env.reset()
                curr_global_state = construct_global_state(curr_obs, possible_agents)
                ep_reward = {a: 0.0 for a in possible_agents}
                ep_length = 0
            else:
                curr_obs = next_obs
                curr_global_state = next_global_state

        # Compute GAE and update policy
        last_val = trainer.get_value(curr_global_state)
        buffer.compute_gae(last_val, gamma=config["training"].get("gamma", 0.99), gae_lambda=config["training"].get("gae_lambda", 0.95))
        
        loss_dict = trainer.update(buffer)
        
        mean_update_reward = sum(np.mean(buffer.rewards[a]) for a in possible_agents) * rollout_length
        db.log_training_update(run_id, update, mean_update_reward, loss_dict["actor_loss"], loss_dict["critic_loss"], loss_dict["entropy"])

        # Checkpoint saving
        checkpoint_freq = config["training"].get("checkpoint_freq", 50)
        is_best = mean_update_reward > best_reward
        if is_best:
            best_reward = mean_update_reward

        if (update + 1) % checkpoint_freq == 0 or update == total_updates - 1 or is_best:
            extra_meta = {
                "update": update,
                "best_reward": best_reward,
                "seed": seed,
                "rng_state": torch.get_rng_state()
            }
            trainer.save_checkpoint(latest_ckpt_path, extra_meta)
            if is_best:
                trainer.save_checkpoint(best_ckpt_path, extra_meta)

        if (update + 1) % max(1, checkpoint_freq // 2) == 0 or update == total_updates - 1:
            print(f"Update [{update + 1}/{total_updates}] | Mean Reward: {mean_update_reward:.2f} | Actor Loss: {loss_dict['actor_loss']:.4f} | Critic Loss: {loss_dict['critic_loss']:.4f}")

    print(f"MAPPO CTDE Training Complete. Best Checkpoint saved to {best_ckpt_path}")
    return best_ckpt_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPPO Training Script")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2k-step smoke training")

    args = parser.parse_args()
    train_mappo(config_path=args.config, timesteps_override=args.timesteps, resume=args.resume, smoke=args.smoke)
