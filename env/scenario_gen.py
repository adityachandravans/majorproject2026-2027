"""
Scenario Generator for Disaster Response Coordination.
Kept separate from the PettingZoo environment class to allow independent tuning of scenario randomization.
"""
import random
import numpy as np

def generate_scenario(config: dict, seed: int = None, fixed: bool = False) -> dict:
    """
    Generates scenario initial state including grid, zones, depots, and agent positions.
    
    Args:
        config (dict): Configuration dictionary containing environment parameters.
        seed (int, optional): Random seed for reproducibility.
        fixed (bool): If True, generates a deterministic scenario (ideal for smoke testing).
        
    Returns:
        dict: Initial scenario state dictionary.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env_cfg = config.get("environment", {})
    grid_w, grid_h = env_cfg.get("grid_size", [10, 10])
    num_zones = env_cfg.get("num_zones", 4)
    num_depots = env_cfg.get("num_depots", 2)
    agent_counts = env_cfg.get("agent_counts", {"medical": 2, "rescue": 2, "logistics": 2})

    if fixed:
        # Deterministic 3x3 or grid scenario for smoke test
        zones = [
            {
                "id": 0,
                "pos": (1, 1),
                "severity": 2,
                "casualty_count": 3,
                "trapped_count": 2,
                "casualties_treated": 0,
                "accessibility": True,
                "supplies_delivered": 0
            }
        ]
        depots = [
            {
                "id": 0,
                "pos": (0, 0),
                "ambulances": 5,
                "rescue_teams": 5,
                "supply_units": 20
            }
        ]
        agents = {
            "medical_0": {"pos": (0, 0), "role": "medical", "status": 0, "carrying": 0},
            "rescue_0": {"pos": (0, 0), "role": "rescue", "status": 0},
            "logistics_0": {"pos": (0, 0), "role": "logistics", "supply_payload": 10}
        }
        return {
            "grid_size": (grid_w, grid_h),
            "zones": zones,
            "depots": depots,
            "agents": agents,
            "step_count": 0
        }

    # Dynamic / randomized scenario generation
    # Place depots at corners/edges
    depots = []
    depot_positions = [(0, 0), (grid_w - 1, grid_h - 1), (0, grid_h - 1), (grid_w - 1, 0)]
    for i in range(num_depots):
        pos = depot_positions[i % len(depot_positions)]
        depot_stock = env_cfg.get("depot_stock", {"ambulances": 5, "rescue_teams": 5, "supply_units": 20})
        depots.append({
            "id": i,
            "pos": pos,
            "ambulances": depot_stock["ambulances"],
            "rescue_teams": depot_stock["rescue_teams"],
            "supply_units": depot_stock["supply_units"]
        })

    # Place disaster zones randomly on grid (avoiding depots)
    occupied_positions = set(d["pos"] for d in depots)
    zones = []
    zone_severities = env_cfg.get("zone_severities", [1, 2, 3])
    casualty_ranges = env_cfg.get("casualty_ranges", {1: [1, 3], 2: [4, 6], 3: [7, 10]})
    trapped_ranges = env_cfg.get("trapped_ranges", {1: [1, 2], 2: [2, 4], 3: [4, 7]})

    for i in range(num_zones):
        while True:
            zx = random.randint(0, grid_w - 1)
            zy = random.randint(0, grid_h - 1)
            if (zx, zy) not in occupied_positions:
                occupied_positions.add((zx, zy))
                break

        severity = random.choice(zone_severities)
        c_min, c_max = casualty_ranges.get(severity, [2, 5])
        t_min, t_max = trapped_ranges.get(severity, [1, 3])
        
        cas_count = random.randint(c_min, c_max)
        trap_count = random.randint(t_min, t_max)
        accessible = random.random() > 0.2  # 80% accessible, 20% harder

        zones.append({
            "id": i,
            "pos": (zx, zy),
            "severity": severity,
            "casualty_count": cas_count,
            "trapped_count": trap_count,
            "casualties_treated": 0,
            "accessibility": accessible,
            "supplies_delivered": 0
        })

    # Initialize agents starting at depots
    agents = {}
    for role, count in agent_counts.items():
        for idx in range(count):
            agent_id = f"{role}_{idx}"
            start_depot = depots[idx % len(depots)]
            agent_dict = {
                "pos": start_depot["pos"],
                "role": role,
                "status": 0
            }
            if role == "medical":
                agent_dict["carrying"] = 0
            elif role == "logistics":
                agent_dict["supply_payload"] = 10
            agents[agent_id] = agent_dict

    return {
        "grid_size": (grid_w, grid_h),
        "zones": zones,
        "depots": depots,
        "agents": agents,
        "step_count": 0
    }
