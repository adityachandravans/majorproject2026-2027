"""
Scenario Generator for Disaster Response Coordination.
Generates deterministic initial states using local Random/NumPy Generators (FR8, NFR2).
"""

import random
import numpy as np
from typing import Dict, Any, List

def generate_scenario(config: dict, seed: int = None, fixed: bool = False) -> dict:
    """
    Generates initial environment scenario using strictly local RNG instances.
    
    Args:
        config (dict): Configuration dictionary.
        seed (int, optional): Seed for exact reproducibility.
        fixed (bool): If True, generates a small deterministic smoke scenario.
        
    Returns:
        dict: Complete environment initial state structure.
    """
    env_cfg = config.get("environment", {})
    actual_seed = seed if seed is not None else env_cfg.get("seed", 42)
    
    # Strictly local random generators (NFR2 / FR8 requirement 7)
    np_rng = np.random.default_rng(actual_seed)
    py_rng = random.Random(actual_seed)

    grid_w, grid_h = env_cfg.get("grid_size", [10, 10])
    num_zones = env_cfg.get("num_zones", 4)
    num_depots = env_cfg.get("num_depots", 2)
    agent_counts = env_cfg.get("agent_counts", {"medical": 2, "rescue": 2, "logistics": 2})
    inaccessible_prob = env_cfg.get("inaccessible_prob", 0.2)
    depot_stock_cfg = env_cfg.get("depot_stock", {"ambulances": 5, "rescue_teams": 5, "supply_units": 20})

    if fixed:
        # Deterministic fixed scenario for smoke testing
        zones = [
            {
                "id": 0,
                "pos": (1, 1),
                "severity": 2,
                "casualty_count": 3,
                "trapped_count": 2,
                "casualties_treated": 0,
                "casualties_lost": 0,
                "accessibility": True,
                "supply_demand": 5.0,
                "supplies_delivered": 0,
                "casualty_queue": [
                    {"id": f"cas_0_{c}", "spawn_step": 0, "treated_step": None, "lost": False}
                    for c in range(3)
                ]
            }
        ]
        depots = [
            {
                "id": 0,
                "pos": (0, 0),
                "ambulances": 5,
                "rescue_teams": 5,
                "supply_units": 20,
                "max_ambulances": 5,
                "max_rescue_teams": 5,
                "max_supply_units": 20
            }
        ]
        agents = {
            "medical_0": {
                "pos": (0, 0), "role": "medical", "status": 0, "target_zone": -1,
                "has_ambulance": True, "assigned_depot_id": 0, "carrying": 0, "progress_steps": 0
            },
            "rescue_0": {
                "pos": (0, 0), "role": "rescue", "status": 0, "target_zone": -1,
                "has_rescue_team": True, "assigned_depot_id": 0, "progress_steps": 0
            },
            "logistics_0": {
                "pos": (0, 0), "role": "logistics", "status": 0, "target_zone": -1,
                "supply_payload": 10, "assigned_depot_id": 0, "progress_steps": 0
            }
        }
        return {
            "grid_size": (grid_w, grid_h),
            "zones": zones,
            "depots": depots,
            "agents": agents,
            "step_count": 0,
            "seed": actual_seed
        }

    # Generate Depots at corners/edges
    depot_positions = [(0, 0), (grid_w - 1, grid_h - 1), (0, grid_h - 1), (grid_w - 1, 0)]
    depots = []
    for i in range(num_depots):
        pos = depot_positions[i % len(depot_positions)]
        depots.append({
            "id": i,
            "pos": pos,
            "ambulances": depot_stock_cfg["ambulances"],
            "rescue_teams": depot_stock_cfg["rescue_teams"],
            "supply_units": depot_stock_cfg["supply_units"],
            "max_ambulances": depot_stock_cfg["ambulances"],
            "max_rescue_teams": depot_stock_cfg["rescue_teams"],
            "max_supply_units": depot_stock_cfg["supply_units"]
        })

    # Generate Zones randomly on grid avoiding depot locations
    occupied_positions = set(d["pos"] for d in depots)
    zones = []
    zone_severities = env_cfg.get("zone_severities", [1, 2, 3])
    casualty_ranges = env_cfg.get("casualty_ranges", {1: [1, 3], 2: [4, 6], 3: [7, 10]})
    trapped_ranges = env_cfg.get("trapped_ranges", {1: [1, 2], 2: [2, 4], 3: [4, 7]})

    cas_global_idx = 0
    for i in range(num_zones):
        while True:
            zx = py_rng.randint(0, grid_w - 1)
            zy = py_rng.randint(0, grid_h - 1)
            if (zx, zy) not in occupied_positions:
                occupied_positions.add((zx, zy))
                break

        severity = py_rng.choice(zone_severities)
        c_min, c_max = casualty_ranges.get(severity, casualty_ranges.get(str(severity), [2, 5]))
        t_min, t_max = trapped_ranges.get(severity, trapped_ranges.get(str(severity), [1, 3]))

        cas_count = py_rng.randint(c_min, c_max)
        trap_count = py_rng.randint(t_min, t_max)
        accessible = py_rng.random() >= inaccessible_prob

        cas_queue = []
        for _ in range(cas_count):
            cas_queue.append({
                "id": f"cas_{cas_global_idx}",
                "spawn_step": 0,
                "treated_step": None,
                "lost": False
            })
            cas_global_idx += 1

        zones.append({
            "id": i,
            "pos": (zx, zy),
            "severity": severity,
            "casualty_count": cas_count,
            "trapped_count": trap_count,
            "casualties_treated": 0,
            "casualties_lost": 0,
            "accessibility": accessible,
            "supply_demand": float(cas_count + trap_count),
            "supplies_delivered": 0,
            "casualty_queue": cas_queue
        })

    # Generate Agents
    agents = {}
    for role, count in agent_counts.items():
        for idx in range(count):
            agent_id = f"{role}_{idx}"
            start_depot = depots[idx % len(depots)]
            agent_dict = {
                "pos": start_depot["pos"],
                "role": role,
                "status": 0,
                "target_zone": -1,
                "assigned_depot_id": start_depot["id"],
                "progress_steps": 0
            }
            
            if role == "medical":
                # Medical agents start without ambulance reserved until requested or at depot
                agent_dict["has_ambulance"] = False
                agent_dict["carrying"] = 0
            elif role == "rescue":
                agent_dict["has_rescue_team"] = False
            elif role == "logistics":
                agent_dict["supply_payload"] = 0
                
            agents[agent_id] = agent_dict

    return {
        "grid_size": (grid_w, grid_h),
        "zones": zones,
        "depots": depots,
        "agents": agents,
        "step_count": 0,
        "seed": actual_seed
    }
