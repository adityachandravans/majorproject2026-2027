"""
Greedy Rule-Based Baseline Policy (FR9).
Provides heuristic baseline actions for Medical, Rescue, and Logistics agents.
"""

import random
import numpy as np
from typing import Dict, Any

class GreedyPolicy:
    def __init__(self, config: dict):
        self.config = config

    def select_action(self, agent_id: str, agent_role: str, state_data: dict) -> int:
        """
        Selects a greedy rule-based discrete action based on role specs:
        - Medical: Reserve ambulance if missing, move to zone with highest severity casualties, treat, transport.
        - Rescue: Reserve team if missing, move to nearest zone with trapped population, perform rescue.
        - Logistics: Move to zone with highest unmet demand and deliver; restock at depot when empty.
        """
        grid_w, grid_h = state_data["grid_size"]
        agent_info = state_data["agents"][agent_id]
        ax, ay = agent_info["pos"]
        zones = state_data["zones"]
        depots = state_data["depots"]
        num_zones = len(zones)
        num_depots = len(depots)

        def distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        if agent_role == "medical":
            # 1. If carrying treated casualty -> transport to nearest depot/hospital (action = num_zones + 1)
            if agent_info.get("carrying", 0) > 0:
                return num_zones + 1

            # 2. If at zone with casualties > 0 -> Treat action (num_zones)
            for z in zones:
                if (ax, ay) == z["pos"] and z["casualty_count"] > 0:
                    return num_zones

            # 3. If missing ambulance -> move to depot or zone 0
            if not agent_info.get("has_ambulance", False):
                # find nearest depot with ambulances
                depots_with_stock = [d for d in depots if d["ambulances"] > 0]
                if depots_with_stock:
                    best_depot = min(depots_with_stock, key=lambda d: distance((ax, ay), d["pos"]))
                    # find action heading to depot (zone action closest or default 0)
                    return 0

            # 4. Target zone with casualties, prioritized by severity desc, then distance asc
            eligible_zones = [z for z in zones if z["casualty_count"] > 0]
            if eligible_zones:
                best_zone = max(
                    eligible_zones,
                    key=lambda z: (z["severity"], -distance((ax, ay), z["pos"]))
                )
                return best_zone["id"]

            return 0

        elif agent_role == "rescue":
            # 1. If at zone with trapped > 0 -> Rescue action (num_zones)
            for z in zones:
                if (ax, ay) == z["pos"] and z["trapped_count"] > 0:
                    return num_zones

            # 2. Target nearest zone with trapped > 0
            eligible_zones = [z for z in zones if z["trapped_count"] > 0]
            if eligible_zones:
                best_zone = min(
                    eligible_zones,
                    key=lambda z: distance((ax, ay), z["pos"])
                )
                return best_zone["id"]

            return 0

        elif agent_role == "logistics":
            # 1. If payload is 0 -> Restock at depot 0 (action = num_zones)
            if agent_info.get("supply_payload", 0) <= 0:
                return num_zones # depot 0

            # 2. Target zone with largest unmet supply demand
            zones_with_demand = [z for z in zones if z.get("supply_demand", 0) > 0]
            if zones_with_demand:
                best_zone = max(
                    zones_with_demand,
                    key=lambda z: (z.get("supply_demand", 0), -distance((ax, ay), z["pos"]))
                )
                return best_zone["id"]

            return num_zones # depot 0

        return 0


class RandomPolicy:
    """Random baseline policy for benchmarking."""
    def __init__(self, config: dict):
        self.config = config

    def select_action(self, agent_id: str, agent_role: str, state_data: dict) -> int:
        num_zones = len(state_data["zones"])
        num_depots = len(state_data["depots"])
        if agent_role == "logistics":
            return random.randint(0, num_zones + num_depots)
        return random.randint(0, num_zones + 1)


def run_baseline_episode(env, policy) -> dict:
    """Runs a full episode using a baseline policy object and returns metric summaries."""
    obs, infos = env.reset()
    done = False
    accumulated_reward = 0.0

    while not done and env.agents:
        actions = {}
        for agent_id in env.agents:
            role = env.agent_objects[agent_id].role
            actions[agent_id] = policy.select_action(agent_id, role, env.state_data)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        accumulated_reward += sum(rewards.values())
        done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0

    render_meta = env.render_state()
    res_metrics = render_meta["metrics"]
    res_metrics["step_count"] = render_meta["step_count"]
    res_metrics["total_reward"] = accumulated_reward
    return res_metrics
