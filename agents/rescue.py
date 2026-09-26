"""
Rescue Agent implementation for Disaster Response Coordination.

Role Specs (SRS Table 2 & Section 5.2):
- Observation: trapped population count, terrain accessibility, zone positions, rescue team availability.
- Action space (Discrete): move-to-zone(zone_id), perform-rescue, request-backup.
- Reward Formula:
  +15 per successful rescue performed
  -0.5 per step per trapped person remaining (delay penalty)
  -10 per unrescued person if episode ends with trapped people remaining
"""
import numpy as np
from gymnasium import spaces
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
        # Actions: 0..num_zones-1 -> move to zone
        # num_zones -> perform rescue at current zone
        # num_zones + 1 -> request backup from depot
        return spaces.Discrete(num_zones + 2)

    def get_observation(self, env_state: dict) -> np.ndarray:
        grid_w, grid_h = env_state["grid_size"]
        agent_info = env_state["agents"][self.agent_id]
        ax, ay = agent_info["pos"]

        obs = [
            ax / float(grid_w),
            ay / float(grid_h),
            float(agent_info.get("status", 0))
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

        # Depots stock
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["rescue_teams"] / 10.0
            ])

        return np.array(obs, dtype=np.float32)

    def compute_reward(self, env_state: dict, action: int, next_state: dict) -> float:
        """
        Calculates Rescue Agent reward:
        +15 for each person rescued from trapped pool
        -0.5 per step per trapped person remaining
        -10 penalty per unrescued person if episode ends
        """
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
