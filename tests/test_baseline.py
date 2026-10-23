"""
Unit tests for Greedy Baseline Policy, Random Policy, and DB Logger schema (Stage 3).
"""

import os
import pytest
import yaml
import numpy as np

from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, RandomPolicy, run_baseline_episode
from metrics_logging.db_logger import DBLogger

@pytest.fixture
def smoke_config():
    with open("config/smoke_config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_greedy_policy_execution(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    policy = GreedyPolicy(smoke_config)
    
    metrics = run_baseline_episode(env, policy)
    
    assert "task_completion_rate" in metrics
    assert "coordination_efficiency" in metrics
    assert "resource_utilization_rate" in metrics
    assert 0.0 <= metrics["task_completion_rate"] <= 1.0
    assert 0.0 <= metrics["coordination_efficiency"] <= 1.0

def test_random_policy_execution(smoke_config):
    env = DisasterResponseEnv(smoke_config)
    policy = RandomPolicy(smoke_config)
    
    metrics = run_baseline_episode(env, policy)
    assert "task_completion_rate" in metrics

def test_db_logger_schema_and_methods(smoke_config, tmp_path):
    db_file = str(tmp_path / "test_db.db")
    db = DBLogger(db_file)
    db.clear_database()

    run_id = db.start_run(run_type="test_run", config=smoke_config, policy="baseline", seed=42)
    assert run_id.startswith("test_run_")

    # Log episode
    metrics = {
        "total_reward": 100.5,
        "avg_response_time": 4.2,
        "resource_utilization_rate": 0.75,
        "casualties_served": 5,
        "casualties_lost": 1,
        "casualties_total": 6,
        "task_completion_rate": 0.833,
        "coordination_efficiency": 0.90,
        "role_rewards": {"medical": 40.0, "rescue": 35.0, "logistics": 25.5}
    }
    db.log_episode(run_id, 1, metrics)
    df_ep = db.get_run_episodes(run_id)
    assert len(df_ep) == 1
    assert df_ep.iloc[0]["total_reward"] == 100.5

    # Log training update
    db.log_training_update(run_id, 0, 50.0, 0.12, 0.45, 0.05)
    df_curve = db.get_training_curve(run_id)
    assert len(df_curve) == 1
    assert df_curve.iloc[0]["mean_reward"] == 50.0
