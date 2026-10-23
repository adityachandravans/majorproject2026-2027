"""
DisasterResponseEnv: PettingZoo ParallelEnv implementation for multi-agent disaster response coordination.
MARL-ready grid-world environment supporting Medical, Rescue, and Logistics agents (FR1, FR2, FR8, NFR2, NFR7).
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
from pettingzoo import ParallelEnv
from gymnasium import spaces

from env.scenario_gen import generate_scenario
from env.comms import CommunicationChannel
from agents.medical import MedicalAgent
from agents.rescue import RescueAgent
from agents.logistics import LogisticsAgent

class DisasterResponseEnv(ParallelEnv):
    metadata = {
        "render_modes": ["ansi", "matplotlib"],
        "name": "disaster_response_v1"
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
        self.inaccessible_cost = env_cfg.get("inaccessible_cost", 2)
        self.rescue_duration = env_cfg.get("rescue_duration", 2)
        self.demand_growth_rate = env_cfg.get("demand_growth_rate", 0.2)
        self.time_threshold = env_cfg.get("time_thresholds", {}).get("medical_treat_delay", 8)

        # Create agent instances & IDs
        self.possible_agents: List[str] = []
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

        # Communication channel
        self.comm_channel = CommunicationChannel(config, self.possible_agents, self.num_zones)

        # Construct spaces per agent dynamically
        self.observation_spaces = {}
        self.action_spaces = {}

        # Dummy state to determine observation shape dynamically
        dummy_state = generate_scenario(self.config, seed=42, fixed=self.fixed_scenario)
        for agent_id in self.possible_agents:
            agent_obj = self.agent_objects[agent_id]
            obs_sample = agent_obj.get_observation(dummy_state, self.comm_channel)
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
            )
            self.action_spaces[agent_id] = agent_obj.get_action_space(self.num_zones, self.num_depots)

        self.state_data = None
        self.step_count = 0
        self.response_times: List[float] = []
        self.busy_resource_steps = 0
        self.total_resource_steps = 0

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._seed = seed
        else:
            self._seed = self.config.get("environment", {}).get("seed", 42)
            
        is_fixed = self.fixed_scenario or (options and options.get("fixed", False))
        self.state_data = generate_scenario(self.config, seed=self._seed, fixed=is_fixed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.response_times = []
        self.busy_resource_steps = 0
        self.total_resource_steps = 0

        self.total_actions_count = 0
        self.duplicated_actions_count = 0

        self.comm_channel.reset(self._seed)

        observations = {
            agent_id: self.agent_objects[agent_id].get_observation(self.state_data, self.comm_channel)
            for agent_id in self.agents
        }
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def _get_nearest_depot(self, pos: Tuple[int, int], resource_type: str = None) -> dict:
        """Finds nearest depot, optionally filtered by available resource stock > 0."""
        valid_depots = self.state_data["depots"]
        if resource_type:
            valid_depots = [d for d in valid_depots if d.get(resource_type, 0) > 0]
        if not valid_depots:
            valid_depots = self.state_data["depots"]
            
        def dist(d):
            return abs(pos[0] - d["pos"][0]) + abs(pos[1] - d["pos"][1])
            
        return min(valid_depots, key=dist)

    def _move_agent_step(self, agent_id: str, target_pos: Tuple[int, int]):
        """Moves agent one Manhattan step towards target_pos considering terrain accessibility."""
        agent_info = self.state_data["agents"][agent_id]
        ax, ay = agent_info["pos"]
        tx, ty = target_pos

        if (ax, ay) == (tx, ty):
            return

        # Check if agent is undergoing inaccessible terrain delay
        if agent_info.get("inaccessible_delay", 0) > 0:
            agent_info["inaccessible_delay"] -= 1
            return

        # Single step move
        dx = np.sign(tx - ax)
        dy = np.sign(ty - ay)

        if dx != 0:
            next_pos = (ax + dx, ay)
        else:
            next_pos = (ax, ay + dy)

        # Check if next_pos is an inaccessible zone
        for z in self.state_data["zones"]:
            if z["pos"] == next_pos and not z["accessibility"]:
                agent_info["inaccessible_delay"] = self.inaccessible_cost - 1
                break

        agent_info["pos"] = next_pos

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

        # 1. Update supply demands across disaster zones over time
        for zone in self.state_data["zones"]:
            if zone["casualty_count"] > 0 or zone["trapped_count"] > 0:
                zone["supply_demand"] += self.demand_growth_rate

        # 2. Track communication messages & target collisions
        self.comm_channel.update_messages(self.state_data["agents"], actions)
        dups_in_step = self.comm_channel.compute_duplicate_target_count()
        self.duplicated_actions_count += dups_in_step

        treatments_in_step = []
        lost_in_step = 0

        num_zones = len(self.state_data["zones"])

        # 3. Process actions per agent
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue

            self.total_actions_count += 1
            agent_info = self.state_data["agents"][agent_id]
            role = agent_info["role"]
            ax, ay = agent_info["pos"]

            # Resource utilization tracking
            self.total_resource_steps += 1
            if agent_info.get("has_ambulance") or agent_info.get("has_rescue_team") or agent_info.get("supply_payload", 0) > 0:
                self.busy_resource_steps += 1

            if role == "medical":
                if action < num_zones: # Dispatch to target zone
                    target_zone = self.state_data["zones"][action]
                    agent_info["target_zone"] = action
                    
                    # If medical agent has no ambulance, auto-reserve from nearest depot if available
                    if not agent_info.get("has_ambulance", False):
                        depot = self._get_nearest_depot((ax, ay), "ambulances")
                        if depot["ambulances"] > 0 and (ax, ay) == depot["pos"]:
                            depot["ambulances"] -= 1
                            agent_info["has_ambulance"] = True
                            agent_info["assigned_depot_id"] = depot["id"]

                    # Move towards target zone (or towards depot if needed)
                    if not agent_info.get("has_ambulance", False):
                        depot = self._get_nearest_depot((ax, ay), "ambulances")
                        self._move_agent_step(agent_id, depot["pos"])
                    else:
                        self._move_agent_step(agent_id, target_zone["pos"])

                elif action == num_zones: # Treat casualty at current zone
                    for zone in self.state_data["zones"]:
                        if agent_info["pos"] == zone["pos"] and zone["casualty_count"] > 0:
                            if not agent_info.get("has_ambulance", False):
                                # Must hold ambulance to treat
                                depot = self._get_nearest_depot((ax, ay), "ambulances")
                                if (ax, ay) == depot["pos"] and depot["ambulances"] > 0:
                                    depot["ambulances"] -= 1
                                    agent_info["has_ambulance"] = True
                                    agent_info["assigned_depot_id"] = depot["id"]

                            if agent_info.get("has_ambulance", False):
                                zone["casualty_count"] -= 1
                                zone["casualties_treated"] += 1
                                agent_info["status"] = 1 # Treating
                                agent_info["carrying"] = 1 # Carrying treated casualty for transport

                                # Record casualty treatment step & response time
                                if zone.get("casualty_queue"):
                                    untreated_cas = next((c for c in zone["casualty_queue"] if c["treated_step"] is None and not c["lost"]), None)
                                    if untreated_cas:
                                        untreated_cas["treated_step"] = self.step_count
                                        resp_t = self.step_count - untreated_cas["spawn_step"]
                                        self.response_times.append(resp_t)
                                        treatments_in_step.append({
                                            "agent_id": agent_id,
                                            "response_time": resp_t
                                        })
                                break

                elif action >= num_zones + 1: # Transport to hospital/depot
                    depot = self._get_nearest_depot((ax, ay))
                    self._move_agent_step(agent_id, depot["pos"])
                    if agent_info["pos"] == depot["pos"]:
                        if agent_info.get("has_ambulance", False):
                            depot["ambulances"] = min(depot["max_ambulances"], depot["ambulances"] + 1)
                            agent_info["has_ambulance"] = False
                        agent_info["carrying"] = 0
                        agent_info["status"] = 0

            elif role == "rescue":
                if action < num_zones: # Move to target zone
                    target_zone = self.state_data["zones"][action]
                    agent_info["target_zone"] = action
                    
                    if not agent_info.get("has_rescue_team", False):
                        depot = self._get_nearest_depot((ax, ay), "rescue_teams")
                        if depot["rescue_teams"] > 0 and (ax, ay) == depot["pos"]:
                            depot["rescue_teams"] -= 1
                            agent_info["has_rescue_team"] = True
                            agent_info["assigned_depot_id"] = depot["id"]

                    if not agent_info.get("has_rescue_team", False):
                        depot = self._get_nearest_depot((ax, ay), "rescue_teams")
                        self._move_agent_step(agent_id, depot["pos"])
                    else:
                        self._move_agent_step(agent_id, target_zone["pos"])

                elif action == num_zones: # Perform rescue operation
                    for zone in self.state_data["zones"]:
                        if agent_info["pos"] == zone["pos"] and zone["trapped_count"] > 0:
                            if not agent_info.get("has_rescue_team", False):
                                depot = self._get_nearest_depot((ax, ay), "rescue_teams")
                                if (ax, ay) == depot["pos"] and depot["rescue_teams"] > 0:
                                    depot["rescue_teams"] -= 1
                                    agent_info["has_rescue_team"] = True
                                    agent_info["assigned_depot_id"] = depot["id"]

                            if agent_info.get("has_rescue_team", False):
                                agent_info["progress_steps"] += 1
                                agent_info["status"] = 1 # Rescuing
                                if agent_info["progress_steps"] >= self.rescue_duration:
                                    zone["trapped_count"] -= 1
                                    zone["casualty_count"] += 1 # Rescued person joins casualty treat queue
                                    zone.get("casualty_queue", []).append({
                                        "id": f"freed_{self.step_count}",
                                        "spawn_step": self.step_count,
                                        "treated_step": None,
                                        "lost": False
                                    })
                                    agent_info["progress_steps"] = 0
                                    agent_info["status"] = 0
                                break

                elif action >= num_zones + 1: # Request backup / Return team to depot
                    depot = self._get_nearest_depot((ax, ay))
                    self._move_agent_step(agent_id, depot["pos"])
                    if agent_info["pos"] == depot["pos"]:
                        if agent_info.get("has_rescue_team", False):
                            depot["rescue_teams"] = min(depot["max_rescue_teams"], depot["rescue_teams"] + 1)
                            agent_info["has_rescue_team"] = False
                        agent_info["status"] = 0

            elif role == "logistics":
                if action < num_zones: # Deliver supplies to zone
                    target_zone = self.state_data["zones"][action]
                    agent_info["target_zone"] = action
                    self._move_agent_step(agent_id, target_zone["pos"])

                    if agent_info["pos"] == target_zone["pos"] and agent_info.get("supply_payload", 0) > 0:
                        payload = agent_info["supply_payload"]
                        demand = target_zone["supply_demand"]
                        delivered = min(payload, max(5, int(demand))) if demand > 0 else payload
                        
                        target_zone["supplies_delivered"] += delivered
                        target_zone["supply_demand"] = max(0.0, target_zone["supply_demand"] - delivered)
                        agent_info["supply_payload"] = max(0, agent_info["supply_payload"] - delivered)

                elif action >= num_zones and action < num_zones + len(self.state_data["depots"]): # Restock at depot
                    depot_idx = action - num_zones
                    if depot_idx < len(self.state_data["depots"]):
                        depot = self.state_data["depots"][depot_idx]
                        self._move_agent_step(agent_id, depot["pos"])

                        if agent_info["pos"] == depot["pos"]:
                            needed = 20 - agent_info.get("supply_payload", 0)
                            stock = depot["supply_units"]
                            transfer = min(needed, stock)
                            depot["supply_units"] -= transfer
                            agent_info["supply_payload"] += transfer

        # 4. Check casualty health expiration (lost casualties)
        for zone in self.state_data["zones"]:
            for cas in zone.get("casualty_queue", []):
                if cas["treated_step"] is None and not cas["lost"]:
                    if (self.step_count - cas["spawn_step"]) > (self.time_threshold * 3):
                        cas["lost"] = True
                        zone["casualty_count"] = max(0, zone["casualty_count"] - 1)
                        zone["casualties_lost"] += 1
                        lost_in_step += 1

        # 5. Check termination & truncation
        total_remaining_casualties = sum(z["casualty_count"] for z in self.state_data["zones"])
        total_remaining_trapped = sum(z["trapped_count"] for z in self.state_data["zones"])

        all_resolved = (total_remaining_casualties == 0 and total_remaining_trapped == 0)
        timed_out = (self.step_count >= self.max_steps)

        is_done = all_resolved or timed_out
        self.state_data["done"] = is_done

        # 6. Compute rewards and observations
        step_info_meta = {
            "treatments_in_step": treatments_in_step,
            "lost_in_step": lost_in_step
        }

        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        raw_rewards = {}
        for agent_id in self.agents:
            agent_obj = self.agent_objects[agent_id]
            r = agent_obj.compute_reward(prev_state, actions.get(agent_id, 0), self.state_data, step_info_meta)
            raw_rewards[agent_id] = r

        # Add optional shared team-reward term
        team_reward_weight = self.config.get("rewards", {}).get("team_reward_weight", 0.2)
        total_raw_reward = sum(raw_rewards.values())

        for agent_id in self.agents:
            rewards[agent_id] = raw_rewards[agent_id] + team_reward_weight * total_raw_reward
            observations[agent_id] = self.agent_objects[agent_id].get_observation(self.state_data, self.comm_channel)
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

    def render_state(self) -> dict:
        """Returns JSON-serializable dictionary representation of complete environment state & metrics."""
        total_served = sum(z["casualties_treated"] for z in self.state_data["zones"])
        total_lost = sum(z["casualties_lost"] for z in self.state_data["zones"])
        remaining_cas = sum(z["casualty_count"] for z in self.state_data["zones"])
        remaining_trap = sum(z["trapped_count"] for z in self.state_data["zones"])
        total_cas = total_served + total_lost + remaining_cas + remaining_trap

        avg_resp_t = float(np.mean(self.response_times)) if self.response_times else float(self.step_count)
        total_act = max(1, self.total_actions_count)
        coord_eff = max(0.0, 1.0 - (self.duplicated_actions_count / float(total_act)))
        res_util = min(1.0, float(self.busy_resource_steps) / float(max(1, self.total_resource_steps)))

        agents_dict = {}
        for a_id, a_data in self.state_data["agents"].items():
            agents_dict[a_id] = {
                "pos": [int(a_data["pos"][0]), int(a_data["pos"][1])],
                "role": str(a_data["role"]),
                "status": int(a_data["status"]),
                "target_zone": int(a_data.get("target_zone", -1)),
                "carrying": int(a_data.get("carrying", 0)),
                "supply_payload": int(a_data.get("supply_payload", 0)),
                "has_ambulance": bool(a_data.get("has_ambulance", False)),
                "has_rescue_team": bool(a_data.get("has_rescue_team", False))
            }

        zones_list = []
        for z in self.state_data["zones"]:
            zones_list.append({
                "id": int(z["id"]),
                "pos": [int(z["pos"][0]), int(z["pos"][1])],
                "severity": int(z["severity"]),
                "casualty_count": int(z["casualty_count"]),
                "trapped_count": int(z["trapped_count"]),
                "casualties_treated": int(z["casualties_treated"]),
                "casualties_lost": int(z["casualties_lost"]),
                "accessibility": bool(z["accessibility"]),
                "supply_demand": round(float(z["supply_demand"]), 2),
                "supplies_delivered": int(z["supplies_delivered"])
            })

        depots_list = []
        for d in self.state_data["depots"]:
            depots_list.append({
                "id": int(d["id"]),
                "pos": [int(d["pos"][0]), int(d["pos"][1])],
                "ambulances": int(d["ambulances"]),
                "rescue_teams": int(d["rescue_teams"]),
                "supply_units": int(d["supply_units"]),
                "max_ambulances": int(d.get("max_ambulances", 5)),
                "max_rescue_teams": int(d.get("max_rescue_teams", 5)),
                "max_supply_units": int(d.get("max_supply_units", 20))
            })

        return {
            "grid_size": list(self.grid_size),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.state_data.get("done", False),
            "zones": zones_list,
            "depots": depots_list,
            "agents": agents_dict,
            "metrics": {
                "avg_response_time": round(avg_resp_t, 2),
                "resource_utilization_rate": round(res_util, 3),
                "task_completion_rate": round(total_served / float(max(1, total_cas)), 3),
                "coordination_efficiency": round(coord_eff, 3),
                "casualties_served": total_served,
                "casualties_lost": total_lost,
                "casualties_total": total_cas
            }
        }

    def render(self, mode: str = "ansi"):
        """Render grid state as ASCII text or matplotlib plot."""
        if mode == "ansi":
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
            print(f"--- Step {self.step_count} ---")
            print(ascii_str)
            return ascii_str
        elif mode == "matplotlib":
            grid_w, grid_h = self.grid_size
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_xticks(range(grid_w))
            ax.set_yticks(range(grid_h))
            ax.grid(True)
            ax.set_title(f"Disaster Response Grid (Step {self.step_count})")
            for d in self.state_data["depots"]:
                ax.plot(d["pos"][0], d["pos"][1], 'bs', markersize=14, label="Depot")
            colors = {1: 'yellow', 2: 'orange', 3: 'red'}
            for z in self.state_data["zones"]:
                ax.plot(z["pos"][0], z["pos"][1], 'o', color=colors.get(z["severity"], 'red'), markersize=16)
                ax.text(z["pos"][0], z["pos"][1], f"C:{z['casualty_count']}", ha='center', va='center', fontsize=8, weight='bold')
            role_markers = {"medical": 'M', "rescue": 'R', "logistics": 'L'}
            for a_id, a_data in self.state_data["agents"].items():
                ax.text(a_data["pos"][0] + 0.1, a_data["pos"][1] + 0.1, role_markers.get(a_data["role"], 'A'), fontsize=12, color='purple', weight='bold')
            plt.close(fig)
            return fig
