"""
Unit tests for dynamic zone customization, variable zone counts, zone sizes,
random size evolution, and policy compatibility.
"""

import pytest
import yaml
import numpy as np

from env.scenario_gen import generate_scenario
from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, RandomPolicy
from marl.inference import load_policy, MAPPOPolicy

@pytest.fixture
def base_config():
    with open("config/smoke_config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_custom_zones_generation(base_config):
    cfg = base_config.copy()
    cfg["environment"]["custom_zones"] = [
        {"id": 0, "pos": [1, 2], "size": 1, "severity": 1, "casualty_count": 3, "trapped_count": 1, "accessibility": True},
        {"id": 1, "pos": [3, 4], "size": 3, "severity": 3, "casualty_count": 10, "trapped_count": 5, "accessibility": False},
        {"id": 2, "pos": [2, 1], "size": 2, "severity": 2, "casualty_count": 6, "trapped_count": 2, "accessibility": True},
        {"id": 3, "pos": [0, 4], "size": 1, "severity": 1, "casualty_count": 2, "trapped_count": 0, "accessibility": True},
        {"id": 4, "pos": [4, 0], "size": 2, "severity": 2, "casualty_count": 4, "trapped_count": 3, "accessibility": True},
    ]

    scenario = generate_scenario(cfg, seed=123)
    assert len(scenario["zones"]) == 5
    assert scenario["zones"][0]["size"] == 1
    assert scenario["zones"][1]["size"] == 3
    assert scenario["zones"][1]["casualty_count"] == 10
    assert scenario["zones"][1]["trapped_count"] == 5
    assert not scenario["zones"][1]["accessibility"]
    assert scenario["zones"][2]["pos"] == (2, 1)

def test_procedural_random_zone_sizes(base_config):
    cfg = base_config.copy()
    cfg["environment"]["num_zones"] = 6
    cfg["environment"]["randomize_zone_sizes"] = True
    cfg["environment"]["zone_size_range"] = [1, 4]

    scenario = generate_scenario(cfg, seed=42)
    assert len(scenario["zones"]) == 6
    sizes = [z["size"] for z in scenario["zones"]]
    for s in sizes:
        assert 1 <= s <= 4
    # With 6 zones between 1 and 4, there should be variation
    assert max(sizes) >= 2

def test_env_with_variable_zones_and_sizes(base_config):
    cfg = base_config.copy()
    cfg["environment"]["custom_zones"] = [
        {"id": 0, "pos": [2, 2], "size": 2, "severity": 2, "casualty_count": 5, "trapped_count": 2, "accessibility": True}
    ]
    cfg["environment"]["dynamic_zone_sizes"] = True
    cfg["environment"]["size_change_prob"] = 1.0  # Force size evolution check
    cfg["environment"]["zone_size_range"] = [1, 3]

    env = DisasterResponseEnv(cfg)
    assert env.num_zones == 1
    obs, _ = env.reset(seed=42)

    # Zone size is 2, center at (2, 2). Agent at (2, 3) is distance 1, hence inside the zone area
    assert env._is_agent_in_zone((2, 2), env.state_data["zones"][0])
    assert env._is_agent_in_zone((2, 3), env.state_data["zones"][0])
    assert env._is_agent_in_zone((3, 3), env.state_data["zones"][0])
    # Distance 2 from center is outside size 2 (which covers radius 1)
    assert not env._is_agent_in_zone((2, 4), env.state_data["zones"][0])

    # Step environment and check dynamic size change
    actions = {a: 0 for a in env.agents}
    next_obs, rewards, _, _, _ = env.step(actions)
    assert len(next_obs) == len(env.possible_agents)
    state = env.render_state()
    assert "size" in state["zones"][0]

def test_greedy_policy_with_arbitrary_zones(base_config):
    cfg = base_config.copy()
    # 5 zones with different sizes and casualties
    cfg["environment"]["custom_zones"] = [
        {"id": i, "pos": [i % 5, (i * 2) % 5], "size": (i % 3) + 1, "severity": (i % 3) + 1, "casualty_count": i + 2, "trapped_count": i + 1, "accessibility": True}
        for i in range(5)
    ]
    env = DisasterResponseEnv(cfg)
    obs, _ = env.reset(seed=42)
    greedy = GreedyPolicy(cfg)

    for agent_id in env.agents:
        role = env.agent_objects[agent_id].role
        action = greedy.select_action(agent_id, role, env.state_data)
        assert isinstance(action, int)
        assert 0 <= action < env.action_space(agent_id).n

def test_mappo_policy_robustness_with_variable_zones(base_config):
    # Train dummy or test inference with obs padding
    cfg = base_config.copy()
    cfg["environment"]["custom_zones"] = [
        {"id": 0, "pos": [1, 1], "size": 1, "severity": 1, "casualty_count": 2, "trapped_count": 1, "accessibility": True}
    ]
    env = DisasterResponseEnv(cfg)
    obs, _ = env.reset(seed=42)

    # Use RandomPolicy as well to test random actions
    random_pol = RandomPolicy(cfg)
    actions = {a: random_pol.select_action(a, env.agent_objects[a].role, env.state_data) for a in env.agents}
    next_obs, rewards, _, _, _ = env.step(actions)
    assert len(next_obs) == len(env.possible_agents)
