"""
Medical Agent implementation for Disaster Response Coordination.

Role Specs (SRS Table 2 & Section 5.2):
- Observation: casualty locations, casualty severity, ambulance availability, agent position/status.
- Action space (Discrete): dispatch-to-zone(zone_id), treat, transport-to-hospital.
- Reward Formula:
  +10 per casualty treated within time threshold
  -1 per step of delay past threshold (medical_treat_delay)
  -20 if casualty is lost / untreated at episode conclusion
"""
import numpy as np
from gymnasium import spaces
from agents.base_agent import BaseAgent

class MedicalAgent(BaseAgent):
    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, role="medical", config=config)
        self.time_threshold = config["environment"].get("time_thresholds", {}).get("medical_treat_delay", 8)
        self.rewards_cfg = config.get("rewards", {}).get("medical", {
            "treated_on_time": 10.0,
            "delay_penalty": 1.0,
            "lost_casualty_penalty": 20.0
        })

    def get_action_space(self, num_zones: int, num_depots: int) -> spaces.Space:
        # Actions: 0..num_zones-1 -> dispatch to zone
        # num_zones -> treat casualty at current zone
        # num_zones + 1 -> transport casualty to hospital/depot
        return spaces.Discrete(num_zones + 2)

    def get_observation(self, env_state: dict) -> np.ndarray:
        grid_w, grid_h = env_state["grid_size"]
        agent_info = env_state["agents"][self.agent_id]
        ax, ay = agent_info["pos"]

        obs = [
            ax / float(grid_w),
            ay / float(grid_h),
            float(agent_info.get("status", 0)), # 0: idle/moving, 1: treating, 2: transporting
            float(agent_info.get("carrying", 0)) # carrying casualty flag
        ]

        # Add zones features
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

        # Add depots stock & position
        for depot in env_state["depots"]:
            dx, dy = depot["pos"]
            obs.extend([
                dx / float(grid_w),
                dy / float(grid_h),
                depot["ambulances"] / 10.0
            ])

        return np.array(obs, dtype=np.float32)

    def compute_reward(self, env_state: dict, action: int, next_state: dict) -> float:
        """
        Calculates Medical Agent reward strictly according to formula:
        +10 for each casualty treated on time
        -1 per step of delay past threshold
        -20 if casualty is lost / untreated at episode end
        """
        reward = 0.0
        step_count = next_state.get("step_count", 0)

        # Check casualties treated in this step
        prev_treated = sum(z["casualties_treated"] for z in env_state["zones"])
        curr_treated = sum(z["casualties_treated"] for z in next_state["zones"])
        delta_treated = curr_treated - prev_treated

        if delta_treated > 0:
            if step_count <= self.time_threshold:
                reward += delta_treated * self.rewards_cfg["treated_on_time"]
            else:
                delay = step_count - self.time_threshold
                reward += delta_treated * (self.rewards_cfg["treated_on_time"] - self.rewards_cfg["delay_penalty"] * delay)

        # Step penalty for remaining untreated casualties
        total_remaining = sum(z["casualty_count"] for z in next_state["zones"])
        if total_remaining > 0 and step_count > self.time_threshold:
            reward -= self.rewards_cfg["delay_penalty"] * 0.1 * total_remaining

        # Episode termination penalty for lost casualties
        if next_state.get("done", False) and total_remaining > 0:
            reward -= total_remaining * self.rewards_cfg["lost_casualty_penalty"]

        return float(reward)
