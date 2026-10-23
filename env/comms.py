"""
Communication Layer for Multi-Agent Disaster Response System (FR3).
Manages shared message passing, message dropout, target zone broadcasting, and collision tracking.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class CommunicationChannel:
    def __init__(self, config: dict, possible_agents: List[str], num_zones: int, seed: int = 42):
        self.config = config
        self.possible_agents = possible_agents
        self.num_zones = num_zones
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        comm_cfg = config.get("comms", {})
        self.enabled = comm_cfg.get("comm_enabled", True)
        self.dropout_prob = comm_cfg.get("comm_dropout_prob", 0.1)
        
        # Message dimension per agent: 
        # target_zone_one_hot (num_zones + 1) + role (3) + status (1) + resource_level (1)
        self.msg_dim_per_agent = (num_zones + 1) + 3 + 1 + 1
        self.messages: Dict[str, np.ndarray] = {
            agent_id: np.zeros(self.msg_dim_per_agent, dtype=np.float32)
            for agent_id in possible_agents
        }
        self.claimed_targets: Dict[str, int] = {agent_id: -1 for agent_id in possible_agents}

    def reset(self, seed: int = None):
        """Reset messages, target claims, and RNG seed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.messages = {
            agent_id: np.zeros(self.msg_dim_per_agent, dtype=np.float32)
            for agent_id in self.possible_agents
        }
        self.claimed_targets = {agent_id: -1 for agent_id in self.possible_agents}

    def encode_message(self, agent_id: str, role: str, target_zone: int, status: float, resource_level: float) -> np.ndarray:
        """Encodes an agent's current state into a message vector."""
        msg = np.zeros(self.msg_dim_per_agent, dtype=np.float32)
        
        tz_idx = target_zone if (0 <= target_zone < self.num_zones) else self.num_zones
        msg[tz_idx] = 1.0
        
        role_map = {"medical": 0, "rescue": 1, "logistics": 2}
        r_idx = role_map.get(role, 0)
        msg[self.num_zones + 1 + r_idx] = 1.0
        
        msg[self.num_zones + 4] = float(status)
        msg[self.num_zones + 5] = float(resource_level)
        
        return msg

    def update_messages(self, agents_data: Dict[str, dict], last_actions: Dict[str, int]):
        """Updates the shared channel with latest messages from all active agents."""
        for agent_id in self.possible_agents:
            if agent_id in agents_data:
                info = agents_data[agent_id]
                role = info.get("role", "medical")
                action = last_actions.get(agent_id, -1)
                
                target_zone = action if (0 <= action < self.num_zones) else info.get("target_zone", -1)
                self.claimed_targets[agent_id] = target_zone
                
                status = info.get("status", 0)
                resource_level = info.get("carrying", info.get("supply_payload", 0))
                
                self.messages[agent_id] = self.encode_message(
                    agent_id, role, target_zone, status, resource_level
                )
            else:
                self.messages[agent_id] = np.zeros(self.msg_dim_per_agent, dtype=np.float32)
                self.claimed_targets[agent_id] = -1

    def get_received_messages(self, querying_agent_id: str) -> np.ndarray:
        """
        Returns concatenated message vector from all OTHER agents using local RNG for dropout.
        """
        if not self.enabled:
            other_agents_count = len(self.possible_agents) - 1
            return np.zeros(other_agents_count * self.msg_dim_per_agent, dtype=np.float32)

        received = []
        for agent_id in self.possible_agents:
            if agent_id == querying_agent_id:
                continue
            
            if self.dropout_prob > 0.0:
                drop_roll = self.rng.random()
                if drop_roll < self.dropout_prob:
                    received.append(np.zeros(self.msg_dim_per_agent, dtype=np.float32))
                    continue
            
            received.append(self.messages[agent_id])
            
        return np.concatenate(received, axis=0) if received else np.zeros(0, dtype=np.float32)

    def compute_duplicate_target_count(self) -> int:
        """Computes how many agents selected the same target zone in the current step."""
        target_counts: Dict[int, int] = {}
        for agent_id, target in self.claimed_targets.items():
            if target >= 0 and target < self.num_zones:
                target_counts[target] = target_counts.get(target, 0) + 1
        
        duplicates = sum(count - 1 for count in target_counts.values() if count > 1)
        return duplicates

    def get_comms_dim(self) -> int:
        """Returns total length of received message vector for an agent."""
        return (len(self.possible_agents) - 1) * self.msg_dim_per_agent
