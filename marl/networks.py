"""
Neural Network Architectures for MAPPO (CTDE) (FR4, NFR3).
Provides per-role decentralized actor networks and a centralized critic network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Tuple

class RoleActor(nn.Module):
    """
    Decentralized Actor Network for a specific agent role.
    Takes local observation vector (including comms messages) and outputs categorical action logits.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(obs)
        if action_mask is not None:
            # Apply action masking for invalid actions
            logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))
        return logits

    def get_action(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized Critic Network for CTDE.
    Takes global state (concatenated observations of all agents) and returns scalar value estimate V(s).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
