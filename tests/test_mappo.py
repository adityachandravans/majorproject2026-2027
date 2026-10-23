"""
Unit tests for MAPPO CTDE training, checkpoint save/resume, and decentralized inference (Stage 2).
"""

import os
import pytest
import yaml
import torch
import numpy as np

from env.disaster_env import DisasterResponseEnv
from marl.mappo import MAPPOTrainer
from marl.buffer import MAPPOBuffer
from marl.inference import load_policy, MAPPOPolicy

@pytest.fixture
def smoke_config():
    with open("config/smoke_config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_mappo_one_update_step(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    obs_dict, _ = env.reset(seed=42)
    
    possible_agents = env.possible_agents
    obs_dims = {a: env.observation_space(a).shape[0] for a in possible_agents}
    action_dims = {a: env.action_space(a).n for a in possible_agents}
    
    global_state_dim = sum(obs_dims.values())
    rollout_length = 16
    
    trainer = MAPPOTrainer(smoke_config, possible_agents, obs_dims, global_state_dim, action_dims)
    buffer = MAPPOBuffer(rollout_length, possible_agents, obs_dims, global_state_dim, torch.device("cpu"))

    # Initial param copy
    initial_param = trainer.actors["medical"].net[0].weight.clone()

    for _ in range(rollout_length):
        actions, log_probs = trainer.select_actions(obs_dict)
        val = trainer.get_value(np.concatenate(list(obs_dict.values())))
        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
        
        buffer.insert(obs_dict, np.concatenate(list(obs_dict.values())), actions, log_probs, rewards, done, val)
        if done:
            obs_dict, _ = env.reset()
        else:
            obs_dict = next_obs

    buffer.compute_gae(0.0)
    loss_metrics = trainer.update(buffer)

    assert "actor_loss" in loss_metrics
    assert "critic_loss" in loss_metrics
    assert np.isfinite(loss_metrics["actor_loss"])
    assert np.isfinite(loss_metrics["critic_loss"])

    # Parameters should change after optimization update
    updated_param = trainer.actors["medical"].net[0].weight
    assert not torch.equal(initial_param, updated_param)

def test_mappo_checkpoint_save_resume(smoke_config, tmp_path):
    env = DisasterResponseEnv(smoke_config)
    obs_dict, _ = env.reset(seed=42)
    possible_agents = env.possible_agents
    obs_dims = {a: env.observation_space(a).shape[0] for a in possible_agents}
    action_dims = {a: env.action_space(a).n for a in possible_agents}
    global_state_dim = sum(obs_dims.values())

    trainer1 = MAPPOTrainer(smoke_config, possible_agents, obs_dims, global_state_dim, action_dims)
    ckpt_path = str(tmp_path / "test_ckpt.pt")
    
    trainer1.save_checkpoint(ckpt_path, {"update": 10, "best_reward": 50.0})
    assert os.path.exists(ckpt_path)

    trainer2 = MAPPOTrainer(smoke_config, possible_agents, obs_dims, global_state_dim, action_dims)
    extra = trainer2.load_checkpoint(ckpt_path)

    assert extra["update"] == 10
    assert extra["best_reward"] == 50.0
    
    for role in trainer1.actors:
        for p1, p2 in zip(trainer1.actors[role].parameters(), trainer2.actors[role].parameters()):
            assert torch.equal(p1, p2)

def test_decentralized_inference(smoke_config, tmp_path):
    env = DisasterResponseEnv(smoke_config)
    obs_dict, _ = env.reset(seed=42)
    possible_agents = env.possible_agents
    obs_dims = {a: env.observation_space(a).shape[0] for a in possible_agents}
    action_dims = {a: env.action_space(a).n for a in possible_agents}
    global_state_dim = sum(obs_dims.values())

    trainer = MAPPOTrainer(smoke_config, possible_agents, obs_dims, global_state_dim, action_dims)
    ckpt_path = str(tmp_path / "test_inference_ckpt.pt")
    trainer.save_checkpoint(ckpt_path)

    policy = load_policy(ckpt_path)
    actions = policy.act(obs_dict, deterministic=True)

    assert len(actions) == len(possible_agents)
    for agent_id, act_val in actions.items():
        assert isinstance(act_val, int)
        assert 0 <= act_val < action_dims[agent_id]
