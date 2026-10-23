"""
Smoke Test for Multi-Agent Disaster Response Coordination System (Section 6).
Executes a fixed 3x3 scenario, 1 zone, 1 agent per role, 5 max steps end-to-end with the baseline policy.
Asserts no crashes, valid environment stepping, and successful database metric logging.
"""
import os
import yaml
import pytest
from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, run_baseline_episode
from metrics_logging.db_logger import DBLogger

def test_smoke_pipeline():
    config_path = os.path.join("config", "smoke_config.yaml")
    assert os.path.exists(config_path), "Smoke config file missing!"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    db_cfg = config.get("db", {})
    db = DBLogger(db_cfg)
    db.clear_database()

    # 1. Initialize environment with fixed scenario mode
    env = DisasterResponseEnv(config, fixed_scenario=True)
    obs, infos = env.reset(seed=config["environment"]["seed"], options={"fixed": True})

    assert len(obs) == len(env.possible_agents), "Observation dict size mismatch!"
    for agent_id, agent_obs in obs.items():
        assert agent_obs is not None and len(agent_obs) > 0, f"Empty observation for {agent_id}"

    # 2. Instantiate greedy policy
    policy = GreedyPolicy(config)

    # 3. Step environment for 1 step manually to verify space and step output contracts
    actions = {
        agent_id: policy.select_action(agent_id, env.agent_objects[agent_id].role, env.state_data)
        for agent_id in env.agents
    }
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

    assert len(rewards) == len(env.possible_agents), "Rewards dict size mismatch!"
    assert len(terminations) == len(env.possible_agents), "Terminations dict size mismatch!"

    # 4. Run full episode using helper
    env_full = DisasterResponseEnv(config, fixed_scenario=True)
    metrics = run_baseline_episode(env_full, policy)

    assert metrics["step_count"] > 0, "Episode step count should be positive"
    assert "total_reward" in metrics, "Missing total_reward in metrics"
    assert "task_completion_rate" in metrics, "Missing task_completion_rate in metrics"

    # 5. Verify database logging
    db = DBLogger(db_cfg)
    run_id = db.start_run(run_type="smoke_baseline", config=config, notes="Smoke test run")
    db.log_episode(run_id, episode_num=1, metrics=metrics)

    df_episodes = db.get_run_episodes(run_id)
    assert not df_episodes.empty, "Failed to fetch logged episode from database"
    assert len(df_episodes) == 1, "Expected 1 episode record"
    assert df_episodes.iloc[0]["run_id"] == run_id

    print("\n[SUCCESS] SMOKE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_smoke_pipeline()
