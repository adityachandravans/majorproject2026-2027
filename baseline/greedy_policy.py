"""
Greedy Rule-Based Baseline Policy (FR9).
Provides rule-based heuristics for Medical, Rescue, and Logistics agents for benchmark comparison.
"""
import numpy as np
from typing import Dict, Any

class GreedyPolicy:
    def __init__(self, config: dict):
        self.config = config

    def select_action(self, agent_id: str, agent_role: str, state_data: dict) -> int:
        """
        Selects a greedy rule-based discrete action based on role specs:
        - Medical: Always dispatch to nearest untreated casualty zone with highest severity; treat if present.
        - Rescue: Always move to nearest zone with trapped population; rescue if present.
        - Logistics: Route supplies to zone with largest unmet demand.
        """
        grid_w, grid_h = state_data["grid_size"]
        agent_info = state_data["agents"][agent_id]
        ax, ay = agent_info["pos"]
        zones = state_data["zones"]
        num_zones = len(zones)
        num_depots = len(state_data["depots"])

        def distance(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        if agent_role == "medical":
            # 1. If currently at a zone with untreated casualties -> TREAT (action = num_zones)
            for z in zones:
                if (ax, ay) == z["pos"] and z["casualty_count"] > 0:
                    return num_zones # treat action

            # 2. Find zone with casualties > 0, prioritized by severity desc, then distance asc
            eligible_zones = [z for z in zones if z["casualty_count"] > 0]
            if eligible_zones:
                best_zone = max(
                    eligible_zones,
                    key=lambda z: (z["severity"], -distance((ax, ay), z["pos"]))
                )
                return best_zone["id"]
            
            # Default: move to zone 0 or idle
            return 0

        elif agent_role == "rescue":
            # 1. If at a zone with trapped population -> RESCUE (action = num_zones)
            for z in zones:
                if (ax, ay) == z["pos"] and z["trapped_count"] > 0:
                    return num_zones # rescue action

            # 2. Find nearest zone with trapped population > 0
            eligible_zones = [z for z in zones if z["trapped_count"] > 0]
            if eligible_zones:
                best_zone = min(
                    eligible_zones,
                    key=lambda z: distance((ax, ay), z["pos"])
                )
                return best_zone["id"]

            return 0

        elif agent_role == "logistics":
            # 1. Target zone with largest unmet demand (casualty_count + trapped_count)
            zones_with_demand = [
                (z, z["casualty_count"] + z["trapped_count"])
                for z in zones
            ]
            zones_with_demand.sort(key=lambda item: (item[1], -distance((ax, ay), item[0]["pos"])), reverse=True)
            
            if zones_with_demand and zones_with_demand[0][1] > 0:
                target_z = zones_with_demand[0][0]
                return target_z["id"]

            # Default: restock at depot 0 if no demand
            return num_zones # depot 0 action

        return 0


def run_baseline_episode(env, policy: GreedyPolicy) -> dict:
    """
    Runs a full episode using the Greedy Baseline Policy and gathers episode trajectory & metrics.
    
    Returns:
        dict: Episode summary metrics including total rewards, response times, completion rates.
    """
    obs, infos = env.reset()
    done = False
    
    total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
    role_rewards = {"medical": 0.0, "rescue": 0.0, "logistics": 0.0}
    
    initial_total_casualties = sum(z["casualty_count"] + z["trapped_count"] for z in env.state_data["zones"])
    initial_depot_resources = sum(
        d["ambulances"] + d["rescue_teams"] + d["supply_units"]
        for d in env.state_data["depots"]
    )
    
    step_history = []

    while not done and env.agents:
        state_snapshot = copy_state(env.state_data)
        actions = {}
        for agent_id in env.agents:
            role = env.agent_objects[agent_id].role
            actions[agent_id] = policy.select_action(agent_id, role, state_snapshot)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        step_history.append({
            "step": env.step_count,
            "state": copy_state(env.state_data),
            "actions": actions,
            "rewards": rewards
        })

        for agent_id, r in rewards.items():
            total_rewards[agent_id] += r
            role = env.agent_objects[agent_id].role
            role_rewards[role] += r

        done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0

    # Compute final metrics
    final_state = env.state_data
    total_casualties_served = sum(z["casualties_treated"] for z in final_state["zones"])
    remaining_casualties = sum(z["casualty_count"] for z in final_state["zones"])
    remaining_trapped = sum(z["trapped_count"] for z in final_state["zones"])

    task_completion_rate = (
        total_casualties_served / float(initial_total_casualties)
        if initial_total_casualties > 0 else 1.0
    )

    avg_response_time = float(env.step_count) # proxy metric average time taken per episode step

    # Resource utilization rate: percentage of resources/agent actions actively deployed vs idle
    total_actions = max(1, env.total_actions_count)
    coordination_efficiency = max(0.0, 1.0 - (env.duplicated_actions_count / float(total_actions)))
    resource_utilization_rate = min(1.0, float(total_actions) / (float(env.step_count * len(env.possible_agents)) + 1e-5))

    return {
        "step_count": env.step_count,
        "total_reward": sum(total_rewards.values()),
        "role_rewards": role_rewards,
        "avg_response_time": avg_response_time,
        "resource_utilization_rate": resource_utilization_rate,
        "casualties_served": total_casualties_served,
        "casualties_total": initial_total_casualties,
        "task_completion_rate": task_completion_rate,
        "coordination_efficiency": coordination_efficiency,
        "step_history": step_history
    }


def copy_state(state_dict: dict) -> dict:
    """Utility helper to produce a clean dict copy of environment state data."""
    import copy
    return copy.deepcopy(state_dict)
