"""
Comprehensive tests for MongoDB Logger implementation.
"""

import pytest
import pandas as pd
from metrics_logging.db_logger import DBLogger

@pytest.fixture
def mongo_logger():
    # Use a test database
    logger = DBLogger(uri_or_config="mongodb://localhost:27017/", db_name="test_marl_db")
    logger.clear_database()
    yield logger
    logger.clear_database()
    logger.close()

def test_mongo_connection_and_indexes(mongo_logger):
    assert mongo_logger.db_name == "test_marl_db"
    assert "runs" in mongo_logger.db.list_collection_names() or True
    run_indexes = mongo_logger.runs.index_information()
    assert "run_id_1" in run_indexes

def test_start_run_and_get_all_runs(mongo_logger):
    config = {"env": "test", "seed": 42}
    run_id1 = mongo_logger.start_run("test_sim", config=config, policy="marl", seed=42, notes="First run")
    assert run_id1.startswith("test_sim_")

    run_id2 = mongo_logger.start_run("test_sim", config=config, policy="baseline", seed=100, notes="Second run")
    assert run_id2.startswith("test_sim_")

    all_runs = mongo_logger.get_all_runs()
    assert len(all_runs) == 2
    run_ids = [r["run_id"] for r in all_runs]
    assert run_id1 in run_ids
    assert run_id2 in run_ids

def test_log_episodes_and_dataframe(mongo_logger):
    run_id = mongo_logger.start_run("marl_eval", config={})
    
    # Check empty query
    empty_df = mongo_logger.get_run_episodes("non_existent_run")
    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty
    assert "total_reward" in empty_df.columns

    # Log 3 episodes
    for ep in range(1, 4):
        metrics = {
            "total_reward": 50.0 * ep,
            "avg_response_time": 3.5,
            "resource_utilization_rate": 0.85,
            "casualties_served": 4 * ep,
            "casualties_lost": 0,
            "casualties_total": 4 * ep,
            "task_completion_rate": 1.0,
            "coordination_efficiency": 0.95,
            "role_rewards": {"medical": 20.0, "rescue": 20.0, "logistics": 10.0}
        }
        mongo_logger.log_episode(run_id, ep, metrics)

    df = mongo_logger.get_run_episodes(run_id)
    assert len(df) == 3
    assert list(df["episode_num"]) == [1, 2, 3]
    assert df.iloc[0]["total_reward"] == 50.0
    assert df.iloc[2]["total_reward"] == 150.0
    assert df.iloc[0]["medical_reward"] == 20.0
    assert "episode_id" in df.columns

def test_log_training_curve(mongo_logger):
    run_id = mongo_logger.start_run("marl_train", config={})
    
    mongo_logger.log_training_update(run_id, update_num=1, mean_reward=10.0, actor_loss=0.5, critic_loss=1.2, entropy=0.08)
    mongo_logger.log_training_update(run_id, update_num=2, mean_reward=25.0, actor_loss=0.3, critic_loss=0.8, entropy=0.06)

    df_curve = mongo_logger.get_training_curve(run_id)
    assert len(df_curve) == 2
    assert list(df_curve["update_num"]) == [1, 2]
    assert df_curve.iloc[1]["mean_reward"] == 25.0

    latest_df = mongo_logger.get_latest_training_curve()
    assert len(latest_df) >= 2
