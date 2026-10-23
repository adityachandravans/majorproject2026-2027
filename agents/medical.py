"""
Medical Agent implementation for Multi-Agent Disaster Response Coordination.
Handles ambulance reservation, Manhattan movement, casualty treatment, and hospital transport.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Any
from agents.base_agent import BaseAgent

class MedicalAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, role="medical", config=config)
        env_cfg = config.get("environment", {})
        self.time_threshold = env_cfg.get("time_thresholds", {}).get("medical_treat_delay", 8)
        self.rewards_cfg = config.get("rewards", {}).get("medical", {
            "treated_on_time": 10.0,
            "delay_penalty": 1.0,
            "lost_casualty_penalty": 20.0
        })

    def get_action_space(self, num_zones: int, num_depots: int) -> spaces.Space:
        # 0..num_zones-1: Dispatch/Move to zone
        # num_zones: Reserve ambulance / Treat casualty at zone
        # num_zones + 1: Transport casualty to hospital/depot
        return spaces.Discrete(num_zones + 2)

    def get_observation(self, env_state: dict, comm_channel: Optional[Any] = None) -> np.ndarray:
        grid_w, grid_h = env_state["grid_size"]
        agent_info = env_state["agents"][self.agent_id]
        ax, ay = agent_info["pos"]

        obs = [
            ax / float(grid_w),
            ay / float(grid_h),
            float(agent_info.get("status", 0)),
            float(agent_info.get("carrying", 0)),
            1.0 if agent_info.get("has_ambulance", False) else 0.0
        ]

        # Zones features
        for zone in env_state["zones"]:
            zx, zy = zone["pos"]
            obs.extend([
                zx / float(grid_w),
                zy / float(grid_h),
                zone["severity"] / 3.0,
                zone["casualty_count"] / 10.0,
                zone["casualties_treated"] / 10.0,
                1.0 if zone["accessibility"] else 0.0
            ])

        # Depots features
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            max_amb = max(1, depot.get("max_ambulances", 10))
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["ambulances"] / float(max_amb)
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
        step_count = next_state.get("step_count", 0)

        # 1. On-time and delayed treatment rewards per casualty treated in step_info
        if step_info and "treatments_in_step" in step_info:
            for treat_event in step_info["treatments_in_step"]:
                if treat_event.get("agent_id") == self.agent_id:
                    resp_time = treat_event.get("response_time", step_count)
                    if resp_time <= self.time_threshold:
                        reward += self.rewards_cfg["treated_on_time"]
                    else:
                        delay = resp_time - self.time_threshold
                        r_val = max(1.0, self.rewards_cfg["treated_on_time"] - self.rewards_cfg["delay_penalty"] * delay)
                        reward += r_val
        else:
            # Fallback based on state difference
            prev_treated = sum(z["casualties_treated"] for z in env_state["zones"])
            curr_treated = sum(z["casualties_treated"] for z in next_state["zones"])
            delta_treated = curr_treated - prev_treated
            if delta_treated > 0:
                if step_count <= self.time_threshold:
                    reward += delta_treated * self.rewards_cfg["treated_on_time"]
                else:
                    delay = step_count - self.time_threshold
                    reward += delta_treated * max(1.0, self.rewards_cfg["treated_on_time"] - self.rewards_cfg["delay_penalty"] * delay)

        # 2. Penalty for lost casualties
        if step_info and "lost_in_step" in step_info:
            reward -= step_info["lost_in_step"] * self.rewards_cfg["lost_casualty_penalty"]

        # 3. Delay penalty for active untreated casualties
        active_casualties = sum(z["casualty_count"] for z in next_state["zones"])
        if active_casualties > 0 and step_count > self.time_threshold:
            reward -= 0.1 * active_casualties * self.rewards_cfg["delay_penalty"]

        # 4. Episode done penalty for unserved casualties
        if next_state.get("done", False) and active_casualties > 0:
            reward -= active_casualties * self.rewards_cfg["lost_casualty_penalty"]

        return float(reward)
