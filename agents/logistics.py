"""
Logistics Agent implementation for Disaster Response Coordination.

Role Specs (SRS Table 2 & Section 5.2):
- Observation: depot stock levels, zone supply demands, agent carrying payload and position.
- Action space (Discrete): route-supplies(zone_id), restock(depot_id), idle.
- Reward Formula:
  +8 for supply demand efficiently fulfilled at active disaster zone
  -2 per step for unfulfilled demand across zones (shortage penalty)
  -5 for supply waste (delivering supplies to a zone with no demand)
"""
import numpy as np
from gymnasium import spaces
from agents.base_agent import BaseAgent

class LogisticsAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, role="logistics", config=config)
        self.rewards_cfg = config.get("rewards", {}).get("logistics", {
            "demand_fulfilled": 8.0,
            "shortage_penalty": 2.0,
            "waste_penalty": 5.0
        })

    def get_action_space(self, num_zones: int, num_depots: int) -> spaces.Space:
        # Actions: 0..num_zones-1 -> route/deliver supplies to zone_id
        # num_zones..num_zones+num_depots-1 -> restock supplies at depot_id
        # num_zones + num_depots -> idle/hold
        return spaces.Discrete(num_zones + num_depots + 1)

    def get_observation(self, env_state: dict) -> np.ndarray:
        grid_w, grid_h = env_state["grid_size"]
        agent_info = env_state["agents"][self.agent_id]
        ax, ay = agent_info["pos"]

        obs = [
            ax / float(grid_w),
            ay / float(grid_h),
            agent_info.get("supply_payload", 0.0) / 20.0
        ]

        # Zones features
        for zone in env_state["zones"]:
            zx, zy = zone["pos"]
            demand = float(zone["casualty_count"] + zone["trapped_count"])
            obs.extend([
                zx / float(grid_w),
                zy / float(grid_h),
                demand / 10.0
            ])

        # Depots stock
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["supply_units"] / 20.0
            ])

        return np.array(obs, dtype=np.float32)

    def compute_reward(self, env_state: dict, action: int, next_state: dict) -> float:
        """
        Calculates Logistics Agent reward:
        +8 when supplies delivered to zone with active demand
        -2 per step for total unmet demand across zones
        -5 if supplies delivered to zone with zero demand (waste)
        """
        reward = 0.0

        # Check supply deliveries in this step
        prev_supplies = [z.get("supplies_delivered", 0) for z in env_state["zones"]]
        curr_supplies = [z.get("supplies_delivered", 0) for z in next_state["zones"]]

        num_zones = len(env_state["zones"])
        for z_idx in range(num_zones):
            delivered = curr_supplies[z_idx] - prev_supplies[z_idx]
            if delivered > 0:
                zone_demand = env_state["zones"][z_idx]["casualty_count"] + env_state["zones"][z_idx]["trapped_count"]
                if zone_demand > 0:
                    reward += self.rewards_cfg["demand_fulfilled"] * delivered
                else:
                    reward -= self.rewards_cfg["waste_penalty"] * delivered

        # Shortage penalty for unmet demand
        total_demand = sum(z["casualty_count"] + z["trapped_count"] for z in next_state["zones"])
        reward -= total_demand * self.rewards_cfg["shortage_penalty"] * 0.1

        return float(reward)
