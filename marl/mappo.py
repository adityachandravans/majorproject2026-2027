"""
MAPPO (Multi-Agent PPO) Algorithm Implementation for CTDE (FR4, NFR3, NFR7).
Implements clipped PPO objective, GAE, parameter sharing per role, and centralized critic updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any

from marl.networks import RoleActor, CentralizedCritic
from marl.buffer import MAPPOBuffer

class MAPPOTrainer:
    def __init__(self, config: dict, possible_agents: List[str], obs_dims: Dict[str, int], global_state_dim: int, action_dims: Dict[str, int], device: torch.device = None):
        self.config = config
        self.possible_agents = possible_agents
        self.obs_dims = obs_dims
        self.global_state_dim = global_state_dim
        self.action_dims = action_dims
        self.device = device or torch.device("cpu")

        train_cfg = config.get("training", {})
        self.lr = train_cfg.get("learning_rate", 0.0003)
        self.clip_eps = train_cfg.get("clip_eps", 0.2)
        self.entropy_coef = train_cfg.get("entropy_coef", 0.01)
        self.value_coef = train_cfg.get("value_coef", 0.5)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
        self.epochs = train_cfg.get("epochs", 10)
        self.batch_size = train_cfg.get("batch_size", 64)

        # Role mapping & per-role parameter shared actors
        self.roles = sorted(list(set(a.split("_")[0] for a in possible_agents)))
        self.actors: Dict[str, RoleActor] = {}
        self.actor_optimizers: Dict[str, torch.optim.Adam] = {}

        for role in self.roles:
            # Find representative agent to get obs_dim & action_dim
            rep_agent = next(a for a in possible_agents if a.startswith(role))
            o_dim = obs_dims[rep_agent]
            a_dim = action_dims[rep_agent]
            
            actor = RoleActor(o_dim, a_dim).to(self.device)
            self.actors[role] = actor
            self.actor_optimizers[role] = torch.optim.Adam(actor.parameters(), lr=self.lr)

        # Centralized Critic
        self.critic = CentralizedCritic(global_state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_actions(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[Dict[str, int], Dict[str, float]]:
        actions = {}
        log_probs = {}
        
        with torch.no_grad():
            for agent_id, obs_vec in obs_dict.items():
                role = agent_id.split("_")[0]
                obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                actor = self.actors[role]
                act_t, log_p_t, _ = actor.get_action(obs_t, deterministic=deterministic)
                
                actions[agent_id] = int(act_t.item())
                log_probs[agent_id] = float(log_p_t.item())

        return actions, log_probs

    def get_value(self, global_state_vec: np.ndarray) -> float:
        with torch.no_grad():
            gs_t = torch.tensor(global_state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            val_t = self.critic(gs_t)
            return float(val_t.item())

    def update(self, buffer: MAPPOBuffer) -> Dict[str, float]:
        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.epochs):
            for b_obs, b_global_state, b_actions, b_log_probs, b_advantages, b_returns, b_values in buffer.get_batches(self.batch_size):
                
                # 1. Update per-role actors
                for role, actor in self.actors.items():
                    role_agents = [a for a in self.possible_agents if a.startswith(role)]
                    
                    role_actor_loss = 0.0
                    role_entropy = 0.0

                    for agent_id in role_agents:
                        obs = b_obs[agent_id]
                        actions = b_actions[agent_id]
                        old_log_probs = b_log_probs[agent_id]
                        advantages = b_advantages[agent_id]

                        # Advantage normalization per agent
                        adv_mean = advantages.mean()
                        adv_std = advantages.std() + 1e-8
                        norm_advantages = (advantages - adv_mean) / adv_std

                        new_log_probs, entropy = actor.evaluate_actions(obs, actions)
                        
                        ratios = torch.exp(new_log_probs - old_log_probs)
                        surr1 = ratios * norm_advantages
                        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * norm_advantages
                        
                        policy_loss = -torch.min(surr1, surr2).mean()
                        entropy_loss = -entropy.mean()

                        role_actor_loss += policy_loss + self.entropy_coef * entropy_loss
                        role_entropy += entropy.mean().item()

                    role_actor_loss = role_actor_loss / len(role_agents)
                    
                    self.actor_optimizers[role].zero_grad()
                    role_actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    self.actor_optimizers[role].step()

                    actor_losses.append(role_actor_loss.item())
                    entropies.append(role_entropy / len(role_agents))

                # 2. Update Centralized Critic
                predicted_values = self.critic(b_global_state)
                # Mean target return across agents
                target_returns = torch.stack([b_returns[a] for a in self.possible_agents], dim=-1).mean(dim=-1)
                
                critic_loss = F.mse_loss(predicted_values, target_returns) * self.value_coef
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                critic_losses.append(critic_loss.item())

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies))
        }

    def save_checkpoint(self, path: str, extra_state: dict = None) -> str:
        checkpoint = {
            "config": self.config,
            "actors": {role: actor.state_dict() for role, actor in self.actors.items()},
            "actor_optimizers": {role: opt.state_dict() for role, opt in self.actor_optimizers.items()},
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "extra_state": extra_state or {}
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for role, actor in self.actors.items():
            if role in checkpoint["actors"]:
                actor.load_state_dict(checkpoint["actors"][role])
            if role in checkpoint["actor_optimizers"]:
                self.actor_optimizers[role].load_state_dict(checkpoint["actor_optimizers"][role])
        
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint.get("extra_state", {})
