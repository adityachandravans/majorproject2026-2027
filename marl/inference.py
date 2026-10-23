"""
Decentralized Execution & Inference Module for MAPPO (FR4, FR5).
Provides lightweight policy loading and decentralized local action selection.
"""

import torch
import numpy as np
from typing import Dict, Any
from marl.networks import RoleActor

class MAPPOPolicy:
    """
    Decentralized MAPPO Policy Execution wrapper.
    Selects actions using only per-agent local observations (and comms messages).
    """
    def __init__(self, actors: Dict[str, RoleActor], device: torch.device = None):
        self.actors = actors
        self.device = device or torch.device("cpu")
        for actor in self.actors.values():
            actor.eval()

    def act(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = True) -> Dict[str, int]:
        actions = {}
        with torch.no_grad():
            for agent_id, obs_vec in obs_dict.items():
                role = agent_id.split("_")[0]
                if role in self.actors:
                    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_t, _, _ = self.actors[role].get_action(obs_t, deterministic=deterministic)
                    actions[agent_id] = int(act_t.item())
                else:
                    actions[agent_id] = 0
        return actions


def load_policy(checkpoint_path: str, device: torch.device = None) -> MAPPOPolicy:
    """
    Loads trained MAPPO actors from a checkpoint file for decentralized execution.
    
    Args:
        checkpoint_path (str): Path to .pt / .pth / .ckpt checkpoint file.
        device (torch.device, optional): Target PyTorch device.
        
    Returns:
        MAPPOPolicy: Ready-to-run decentralized policy object.
    """
    dev = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    
    config = checkpoint.get("config", {})
    env_cfg = config.get("environment", {})
    num_zones = env_cfg.get("num_zones", 4)
    num_depots = env_cfg.get("num_depots", 2)
    
    # Action dims per role
    role_action_dims = {
        "medical": num_zones + 2,
        "rescue": num_zones + 2,
        "logistics": num_zones + num_depots + 1
    }

    actors = {}
    for role, state_dict in checkpoint.get("actors", {}).items():
        # Deduce input obs_dim from first layer weight shape
        obs_dim = state_dict["net.0.weight"].shape[1]
        act_dim = role_action_dims.get(role, state_dict["net.4.weight"].shape[0])
        
        actor = RoleActor(obs_dim, act_dim).to(dev)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors[role] = actor

    return MAPPOPolicy(actors, device=dev)
