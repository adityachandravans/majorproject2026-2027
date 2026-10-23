"""
Logistics Agent implementation for Multi-Agent Disaster Response Coordination.
Handles supply payload management, Manhattan movement, depot restocking, and supply delivery.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Any
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
        # 0..num_zones-1: Deliver supplies to zone_id
        # num_zones..num_zones+num_depots-1: Restock supplies at depot_id
        # num_zones + num_depots: Idle/Hold
        return spaces.Discrete(num_zones + num_depots + 1)

    def get_observation(self, env_state: dict, comm_channel: Optional[Any] = None) -> np.ndarray:
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
            demand = float(zone.get("supply_demand", zone["casualty_count"] + zone["trapped_count"]))
            obs.extend([
                zx / float(grid_w),
                zy / float(grid_h),
                demand / 20.0
            ])

        # Depots features
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            max_sup = max(1, depot.get("max_supply_units", 20))
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["supply_units"] / float(max_sup)
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

        # Deliveries made in this step
        prev_supplies = [z.get("supplies_delivered", 0) for z in env_state["zones"]]
        curr_supplies = [z.get("supplies_delivered", 0) for z in next_state["zones"]]

        num_zones = len(env_state["zones"])
        for z_idx in range(num_zones):
            delivered = curr_supplies[z_idx] - prev_supplies[z_idx]
            if delivered > 0:
                initial_demand = env_state["zones"][z_idx].get("supply_demand", 0)
                if initial_demand > 0:
                    fulfilled = min(delivered, initial_demand)
                    wasted = max(0, delivered - initial_demand)
                    reward += self.rewards_cfg["demand_fulfilled"] * fulfilled
                    if wasted > 0:
                        reward -= self.rewards_cfg["waste_penalty"] * wasted
                else:
                    reward -= self.rewards_cfg["waste_penalty"] * delivered

        # Shortage penalty for total remaining unmet supply demand across zones
        total_demand = sum(z.get("supply_demand", 0) for z in next_state["zones"])
        if total_demand > 0:
            reward -= total_demand * self.rewards_cfg["shortage_penalty"] * 0.1

        return float(reward)
