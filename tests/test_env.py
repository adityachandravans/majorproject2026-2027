"""
Unit tests for DisasterResponseEnv environment contracts and mechanics (Stage 1).
"""

import pytest
import yaml
import numpy as np
from env.disaster_env import DisasterResponseEnv

@pytest.fixture
def smoke_config():
    with open("config/smoke_config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def default_config():
    with open("config/default_config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_env_reset_step_contracts(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    obs, infos = env.reset(seed=42)
    
    assert len(obs) == len(env.possible_agents)
    assert len(infos) == len(env.possible_agents)
    for agent_id, agent_obs in obs.items():
        assert isinstance(agent_obs, np.ndarray)
        assert agent_obs.shape == env.observation_space(agent_id).shape

    # Sample step
    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

    assert len(rewards) == len(env.possible_agents)
    for agent_id in env.possible_agents:
        assert isinstance(rewards[agent_id], (float, int, np.floating))

def test_seed_determinism(default_config):
    env1 = DisasterResponseEnv(default_config)
    obs1, _ = env1.reset(seed=123)
    actions1 = {agent_id: 0 for agent_id in env1.possible_agents}
    obs1_next, r1, _, _, _ = env1.step(actions1)

    env2 = DisasterResponseEnv(default_config)
    obs2, _ = env2.reset(seed=123)
    actions2 = {agent_id: 0 for agent_id in env2.possible_agents}
    obs2_next, r2, _, _, _ = env2.step(actions2)

    for agent_id in env1.possible_agents:
        np.testing.assert_array_equal(obs1[agent_id], obs2[agent_id])
        np.testing.assert_array_equal(obs1_next[agent_id], obs2_next[agent_id])
        assert r1[agent_id] == r2[agent_id]

def test_resource_conservation(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    env.reset(seed=42)

    for _ in range(30):
        if not env.agents:
            break
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        env.step(actions)

        # Verify stock is never negative
        for depot in env.state_data["depots"]:
            assert depot["ambulances"] >= 0
            assert depot["rescue_teams"] >= 0
            assert depot["supply_units"] >= 0

def test_render_state_format(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    env.reset(seed=42)
    state_dict = env.render_state()

    assert "grid_size" in state_dict
    assert "zones" in state_dict
    assert "depots" in state_dict
    assert "agents" in state_dict
    assert "metrics" in state_dict
    assert "avg_response_time" in state_dict["metrics"]
    assert "coordination_efficiency" in state_dict["metrics"]


def test_lazy_initialization_before_reset(smoke_config):
    env = DisasterResponseEnv(smoke_config)

    state = env.render_state()
    assert state["step_count"] == 0
    assert len(env.agents) == len(env.possible_agents)

    actions = {agent_id: 0 for agent_id in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    assert len(next_obs) == len(env.possible_agents)
    assert len(rewards) == len(env.possible_agents)
