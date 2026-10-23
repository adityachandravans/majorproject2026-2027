"""
Unit tests for Medical, Rescue, Logistics agents and Comms channel (Stage 1).
"""

import pytest
import yaml
import numpy as np
from env.disaster_env import DisasterResponseEnv
from env.comms import CommunicationChannel
from agents.medical import MedicalAgent
from agents.rescue import RescueAgent
from agents.logistics import LogisticsAgent

@pytest.fixture
def smoke_config():
    with open("config/smoke_config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_agent_action_spaces(smoke_config):
    num_zones = smoke_config["environment"]["num_zones"]
    num_depots = smoke_config["environment"]["num_depots"]

    med = MedicalAgent("medical_0", smoke_config)
    res = RescueAgent("rescue_0", smoke_config)
    log = LogisticsAgent("logistics_0", smoke_config)

    assert med.get_action_space(num_zones, num_depots).n == num_zones + 2
    assert res.get_action_space(num_zones, num_depots).n == num_zones + 2
    assert log.get_action_space(num_zones, num_depots).n == num_zones + num_depots + 1

def test_comms_layer(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    obs, _ = env.reset(seed=42)

    comm = env.comm_channel
    assert comm.get_comms_dim() > 0

    agents_data = env.state_data["agents"]
    actions = {"medical_0": 0, "rescue_0": 0, "logistics_0": 0}
    comm.update_messages(agents_data, actions)

    # All 3 agents selecting target zone 0 -> duplicate target count = 2
    dups = comm.compute_duplicate_target_count()
    assert dups == 2

    # Check comms toggle OFF
    cfg_off = smoke_config.copy()
    cfg_off["comms"] = {"comm_enabled": False, "comm_dropout_prob": 0.0}
    env_off = DisasterResponseEnv(cfg_off)
    obs_off, _ = env_off.reset(seed=42)
    for agent_id, obs_vec in obs_off.items():
        assert obs_vec is not None
