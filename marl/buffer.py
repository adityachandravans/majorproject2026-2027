"""
Rollout Buffer for Centralized Training with Decentralized Execution (MAPPO Buffer).
Stores trajectories and computes Generalized Advantage Estimation (GAE).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

class MAPPOBuffer:
    def __init__(self, rollout_length: int, possible_agents: List[str], obs_dims: Dict[str, int], global_state_dim: int, device: torch.device):
        self.rollout_length = rollout_length
        self.possible_agents = possible_agents
        self.obs_dims = obs_dims
        self.global_state_dim = global_state_dim
        self.device = device
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.obs = {agent_id: np.zeros((self.rollout_length, self.obs_dims[agent_id]), dtype=np.float32) for agent_id in self.possible_agents}
        self.global_state = np.zeros((self.rollout_length, self.global_state_dim), dtype=np.float32)
        self.actions = {agent_id: np.zeros(self.rollout_length, dtype=np.int64) for agent_id in self.possible_agents}
        self.log_probs = {agent_id: np.zeros(self.rollout_length, dtype=np.float32) for agent_id in self.possible_agents}
        self.rewards = {agent_id: np.zeros(self.rollout_length, dtype=np.float32) for agent_id in self.possible_agents}
        self.dones = np.zeros(self.rollout_length, dtype=np.float32)
        self.values = np.zeros(self.rollout_length, dtype=np.float32)
        
        self.advantages = {agent_id: np.zeros(self.rollout_length, dtype=np.float32) for agent_id in self.possible_agents}
        self.returns = {agent_id: np.zeros(self.rollout_length, dtype=np.float32) for agent_id in self.possible_agents}

    def insert(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state_vec: np.ndarray,
        actions_dict: Dict[str, int],
        log_probs_dict: Dict[str, float],
        rewards_dict: Dict[str, float],
        done: bool,
        value: float
    ):
        idx = self.step_idx
        for agent_id in self.possible_agents:
            if agent_id in obs_dict:
                self.obs[agent_id][idx] = obs_dict[agent_id]
                self.actions[agent_id][idx] = actions_dict.get(agent_id, 0)
                self.log_probs[agent_id][idx] = log_probs_dict.get(agent_id, 0.0)
                self.rewards[agent_id][idx] = rewards_dict.get(agent_id, 0.0)
            
        self.global_state[idx] = global_state_vec
        self.dones[idx] = float(done)
        self.values[idx] = float(value)
        self.step_idx += 1

    def compute_gae(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Computes Generalized Advantage Estimation (GAE) for each agent and the critic."""
        for agent_id in self.possible_agents:
            last_gae_lam = 0.0
            for t in reversed(range(self.rollout_length)):
                if t == self.rollout_length - 1:
                    next_non_terminal = 1.0 - self.dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t]
                    next_val = self.values[t + 1]

                delta = self.rewards[agent_id][t] + gamma * next_val * next_non_terminal - self.values[t]
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[agent_id][t] = last_gae_lam
                self.returns[agent_id][t] = self.advantages[agent_id][t] + self.values[t]

    def get_batches(self, batch_size: int):
        """Generates PyTorch mini-batches for PPO updates."""
        indices = np.arange(self.rollout_length)
        np.random.shuffle(indices)

        for start in range(0, self.rollout_length, batch_size):
            batch_idx = indices[start:start + batch_size]
            
            b_obs = {
                agent_id: torch.tensor(self.obs[agent_id][batch_idx], dtype=torch.float32, device=self.device)
                for agent_id in self.possible_agents
            }
            b_global_state = torch.tensor(self.global_state[batch_idx], dtype=torch.float32, device=self.device)
            b_actions = {
                agent_id: torch.tensor(self.actions[agent_id][batch_idx], dtype=torch.int64, device=self.device)
                for agent_id in self.possible_agents
            }
            b_log_probs = {
                agent_id: torch.tensor(self.log_probs[agent_id][batch_idx], dtype=torch.float32, device=self.device)
                for agent_id in self.possible_agents
            }
            b_advantages = {
                agent_id: torch.tensor(self.advantages[agent_id][batch_idx], dtype=torch.float32, device=self.device)
                for agent_id in self.possible_agents
            }
            b_returns = {
                agent_id: torch.tensor(self.returns[agent_id][batch_idx], dtype=torch.float32, device=self.device)
                for agent_id in self.possible_agents
            }
            b_values = torch.tensor(self.values[batch_idx], dtype=torch.float32, device=self.device)

            yield b_obs, b_global_state, b_actions, b_log_probs, b_advantages, b_returns, b_values
