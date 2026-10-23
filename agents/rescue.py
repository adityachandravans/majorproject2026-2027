"""
Rescue Agent implementation for Multi-Agent Disaster Response Coordination.
Handles rescue team reservation, Manhattan movement, and rescuing trapped population.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Any
from agents.base_agent import BaseAgent

class RescueAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, role="rescue", config=config)
        self.rewards_cfg = config.get("rewards", {}).get("rescue", {
            "rescued_person": 15.0,
            "delay_penalty": 0.5,
            "unrescued_penalty": 10.0
        })

    def get_action_space(self, num_zones: int, num_depots: int) -> spaces.Space:
        # 0..num_zones-1: Move to zone
        # num_zones: Reserve team / Perform rescue operation
        # num_zones + 1: Request backup / Idle
        return spaces.Discrete(num_zones + 2)

    def get_observation(self, env_state: dict, comm_channel: Optional[Any] = None) -> np.ndarray:
        grid_w, grid_h = env_state["grid_size"]
        agent_info = env_state["agents"][self.agent_id]
        ax, ay = agent_info["pos"]

        obs = [
            ax / float(grid_w),
            ay / float(grid_h),
            float(agent_info.get("status", 0)),
            1.0 if agent_info.get("has_rescue_team", False) else 0.0
        ]

        # Zones features
        for zone in env_state["zones"]:
            zx, zy = zone["pos"]
            obs.extend([
                zx / float(grid_w),
                zy / float(grid_h),
                zone["trapped_count"] / 10.0,
                1.0 if zone["accessibility"] else 0.0
            ])

        # Depots features
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            max_rt = max(1, depot.get("max_rescue_teams", 10))
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["rescue_teams"] / float(max_rt)
            ])

        # Communication messages
        if comm_channel is not None:
            comm_msgs = comm_channel.get_received_messages(self.agent_id)
            obs.extend(comm_msgs.tolist())
        else:
            possible_agents = list(env_state["agents"].keys())
            msg_dim_per_agent = (len(env_state["zones"]) + 1) + 3 + 1 + 1
            other_agents_count = max(0, len(possible_agents) - 1)
            obs.extend([0.0] * (other_agents_count * msg_dim_per_agent))

        return np.array(obs, dtype=np.float32)

    def compute_reward(self, env_state: dict, action: int, next_state: dict, step_info: dict = None) -> float:
        reward = 0.0

        prev_trapped = sum(z["trapped_count"] for z in env_state["zones"])
        curr_trapped = sum(z["trapped_count"] for z in next_state["zones"])
        rescued_this_step = prev_trapped - curr_trapped

        if rescued_this_step > 0:
            reward += rescued_this_step * self.rewards_cfg["rescued_person"]

        # Delay penalty for remaining trapped population
        reward -= curr_trapped * self.rewards_cfg["delay_penalty"]

        # Unrescued penalty at episode done
        if next_state.get("done", False) and curr_trapped > 0:
            reward -= curr_trapped * self.rewards_cfg["unrescued_penalty"]

        return float(reward)
