"""
DisasterResponseEnv: PettingZoo ParallelEnv implementation for multi-agent disaster response coordination.
MARL-ready grid-world environment supporting Medical, Rescue, and Logistics agents.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces

from env.scenario_gen import generate_scenario
from agents.medical import MedicalAgent
from agents.rescue import RescueAgent
from agents.logistics import LogisticsAgent

class DisasterResponseEnv(ParallelEnv):
    metadata = {
        "render_modes": ["ansi", "matplotlib"],
        "name": "disaster_response_v0"
    }

    def __init__(self, config: dict, fixed_scenario: bool = False):
        super().__init__()
        self.config = config
        self.fixed_scenario = fixed_scenario

        env_cfg = config.get("environment", {})
        self.grid_size = tuple(env_cfg.get("grid_size", [10, 10]))
        self.max_steps = env_cfg.get("max_steps", 50)
        self.num_zones = env_cfg.get("num_zones", 4)
        self.num_depots = env_cfg.get("num_depots", 2)
        self.agent_counts = env_cfg.get("agent_counts", {"medical": 2, "rescue": 2, "logistics": 2})

        # Create agent instances & IDs
        self.possible_agents = []
        self.agent_objects: Dict[str, Any] = {}

        for role, count in self.agent_counts.items():
            for idx in range(count):
                agent_id = f"{role}_{idx}"
                self.possible_agents.append(agent_id)
                if role == "medical":
                    self.agent_objects[agent_id] = MedicalAgent(agent_id, config)
                elif role == "rescue":
                    self.agent_objects[agent_id] = RescueAgent(agent_id, config)
                elif role == "logistics":
                    self.agent_objects[agent_id] = LogisticsAgent(agent_id, config)

        self.agents = self.possible_agents[:]

        # Construct spaces per agent
        self.observation_spaces = {}
        self.action_spaces = {}

        # Dummy scenario to determine observation shape dynamically
        dummy_state = generate_scenario(self.config, seed=42, fixed=self.fixed_scenario)
        for agent_id in self.possible_agents:
            agent_obj = self.agent_objects[agent_id]
            obs_sample = agent_obj.get_observation(dummy_state)
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
            )
            self.action_spaces[agent_id] = agent_obj.get_action_space(self.num_zones, self.num_depots)

        self.state_data = None
        self.step_count = 0

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._seed = seed
        
        is_fixed = self.fixed_scenario or (options and options.get("fixed", False))
        self.state_data = generate_scenario(self.config, seed=seed, fixed=is_fixed)
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Reset total statistics tracking for coordination metric
        self.total_actions_count = 0
        self.duplicated_actions_count = 0

        observations = {
            agent_id: self.agent_objects[agent_id].get_observation(self.state_data)
            for agent_id in self.agents
        }
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict]
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        prev_state = copy.deepcopy(self.state_data)
        self.step_count += 1
        self.state_data["step_count"] = self.step_count

        # Track actions per zone for coordination efficiency proxy metric
        zone_action_counts = {z_id: 0 for z_id in range(len(self.state_data["zones"]))}

        # Resolve agent actions
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue
            
            agent_info = self.state_data["agents"][agent_id]
            role = agent_info["role"]
            self.total_actions_count += 1

            if action < len(self.state_data["zones"]):
                target_zone = self.state_data["zones"][action]
                agent_info["pos"] = target_zone["pos"]
                zone_action_counts[action] += 1
            else:
                if role == "medical":
                    if action == len(self.state_data["zones"]): # Treat
                        # Check if agent is at a zone with casualties
                        for zone in self.state_data["zones"]:
                            if agent_info["pos"] == zone["pos"] and zone["casualty_count"] > 0:
                                zone["casualty_count"] -= 1
                                zone["casualties_treated"] += 1
                                agent_info["status"] = 1 # Treating
                                break
                    elif action == len(self.state_data["zones"]) + 1: # Transport to hospital/depot
                        agent_info["pos"] = self.state_data["depots"][0]["pos"]
                        agent_info["status"] = 0

                elif role == "rescue":
                    if action == len(self.state_data["zones"]): # Rescue
                        for zone in self.state_data["zones"]:
                            if agent_info["pos"] == zone["pos"] and zone["trapped_count"] > 0:
                                zone["trapped_count"] -= 1
                                zone["casualty_count"] += 1 # Rescuing adds to casualty treat queue
                                break

                elif role == "logistics":
                    num_z = len(self.state_data["zones"])
                    if action < num_z + self.num_depots: # Restock at depot
                        depot_idx = action - num_z
                        if depot_idx < len(self.state_data["depots"]):
                            agent_info["pos"] = self.state_data["depots"][depot_idx]["pos"]
                            agent_info["supply_payload"] = 20

        # Calculate duplicated actions for coordination efficiency metric
        for z_id, count in zone_action_counts.items():
            if count > 1:
                self.duplicated_actions_count += (count - 1)

        # Check termination & truncation
        total_remaining_casualties = sum(z["casualty_count"] for z in self.state_data["zones"])
        total_remaining_trapped = sum(z["trapped_count"] for z in self.state_data["zones"])
        
        all_resolved = (total_remaining_casualties == 0 and total_remaining_trapped == 0)
        timed_out = (self.step_count >= self.max_steps)

        is_done = all_resolved or timed_out
        self.state_data["done"] = is_done

        # Compute rewards and next observations
        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent_id in self.agents:
            agent_obj = self.agent_objects[agent_id]
            rewards[agent_id] = agent_obj.compute_reward(prev_state, actions.get(agent_id, 0), self.state_data)
            observations[agent_id] = agent_obj.get_observation(self.state_data)
            terminations[agent_id] = all_resolved
            truncations[agent_id] = timed_out
            infos[agent_id] = {
                "step_count": self.step_count,
                "all_resolved": all_resolved,
                "duplicated_actions": self.duplicated_actions_count,
                "total_actions": self.total_actions_count
            }

        if is_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self, mode: str = "ansi"):
        """Render grid state as ASCII text or matplotlib visual plot."""
        grid_w, grid_h = self.grid_size
        grid_matrix = [["." for _ in range(grid_w)] for _ in range(grid_h)]

        for depot in self.state_data["depots"]:
            dx, dy = depot["pos"]
            grid_matrix[dy][dx] = "D"

        for zone in self.state_data["zones"]:
            zx, zy = zone["pos"]
            grid_matrix[zy][zx] = f"Z{zone['severity']}"

        for agent_id, a_data in self.state_data["agents"].items():
            ax, ay = a_data["pos"]
            role_char = a_data["role"][0].upper()
            grid_matrix[ay][ax] = f"{grid_matrix[ay][ax]}/{role_char}"

        ascii_str = "\n".join([" ".join(row) for row in grid_matrix])

        if mode == "ansi":
            print(f"--- Step {self.step_count} ---")
            print(ascii_str)
            return ascii_str
        elif mode == "matplotlib":
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_xticks(range(grid_w))
            ax.set_yticks(range(grid_h))
            ax.grid(True)
            ax.set_title(f"Disaster Response Grid (Step {self.step_count})")
            
            # Plot depots
            for d in self.state_data["depots"]:
                ax.plot(d["pos"][0], d["pos"][1], 'bs', markersize=14, label="Depot")
            
            # Plot zones
            colors = {1: 'yellow', 2: 'orange', 3: 'red'}
            for z in self.state_data["zones"]:
                ax.plot(z["pos"][0], z["pos"][1], 'o', color=colors.get(z["severity"], 'red'), markersize=16, label=f"Zone (Sev {z['severity']})")
                ax.text(z["pos"][0], z["pos"][1], f"C:{z['casualty_count']}", ha='center', va='center', color='black', fontsize=8, weight='bold')

            # Plot agents
            role_markers = {"medical": 'M', "rescue": 'R', "logistics": 'L'}
            for a_id, a_data in self.state_data["agents"].items():
                ax.text(a_data["pos"][0] + 0.1, a_data["pos"][1] + 0.1, role_markers.get(a_data["role"], 'A'), fontsize=12, color='purple', weight='bold')

            plt.close(fig)
            return fig
