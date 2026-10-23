"""
Evaluation and Benchmark Comparison Framework (FR9, Objectives 5 & 6).
Evaluates MAPPO, Greedy Baseline, and Random Policy across identical seeds and generates summary reports & plots.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any

from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, RandomPolicy
from marl.inference import load_policy, MAPPOPolicy
from metrics_logging.db_logger import DBLogger
from metrics_logging.plotter import plot_learning_curve, plot_metric_bars, plot_response_time_boxplot

def run_policy_episodes(env: DisasterResponseEnv, policy: Any, num_episodes: int, start_seed: int = 42, is_marl: bool = False) -> Tuple[List[dict], List[float]]:
    ep_metrics = []
    response_times = []

    for ep in range(num_episodes):
        seed = start_seed + ep
        obs_dict, _ = env.reset(seed=seed)
        done = False

        while not done and env.agents:
            if is_marl:
                actions = policy.act(obs_dict, deterministic=True)
            else:
                actions = {
                    agent_id: policy.select_action(agent_id, env.agent_objects[agent_id].role, env.state_data)
                    for agent_id in env.agents
                }

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0

        render_meta = env.render_state()
        m = render_meta["metrics"]
        m["seed"] = seed
        ep_metrics.append(m)
        if env.response_times:
            response_times.extend(env.response_times)
        else:
            response_times.append(env.step_count)

    return ep_metrics, response_times

def evaluate_all(
    config_path: str = "config/default_config.yaml",
    checkpoint_path: str = None,
    num_episodes: int = 100,
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    start_seed = config["environment"].get("seed", 42)
    env = DisasterResponseEnv(config)

    # 1. Greedy Baseline
    print(f"Evaluating Greedy Baseline Policy ({num_episodes} episodes)...")
    greedy_pol = GreedyPolicy(config)
    greedy_metrics, greedy_resp_times = run_policy_episodes(env, greedy_pol, num_episodes, start_seed=start_seed)

    # 2. Random Policy
    print(f"Evaluating Random Policy ({num_episodes} episodes)...")
    random_pol = RandomPolicy(config)
    random_metrics, random_resp_times = run_policy_episodes(env, random_pol, num_episodes, start_seed=start_seed)

    # 3. MARL Policy
    marl_metrics = []
    marl_resp_times = []
    
    if checkpoint_path is None:
        best_ckpt = os.path.join(config.get("training", {}).get("checkpoint_dir", "checkpoints"), "mappo_best.pt")
        latest_ckpt = os.path.join(config.get("training", {}).get("checkpoint_dir", "checkpoints"), "mappo_latest.pt")
        if os.path.exists(best_ckpt):
            checkpoint_path = best_ckpt
        elif os.path.exists(latest_ckpt):
            checkpoint_path = latest_ckpt

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Evaluating Trained MAPPO Policy from {checkpoint_path} ({num_episodes} episodes)...")
        marl_policy = load_policy(checkpoint_path)
        marl_metrics, marl_resp_times = run_policy_episodes(env, marl_policy, num_episodes, start_seed=start_seed, is_marl=True)
    else:
        print("No MAPPO checkpoint found! Running evaluation with initialized policy...")
        from marl.mappo import MAPPOTrainer
        possible_agents = env.possible_agents
        obs_dims = {a: env.observation_space(a).shape[0] for a in possible_agents}
        action_dims = {a: env.action_space(a).n for a in possible_agents}
        global_state_dim = sum(obs_dims.values())
        trainer = MAPPOTrainer(config, possible_agents, obs_dims, global_state_dim, action_dims)
        marl_policy = MAPPOPolicy(trainer.actors)
        marl_metrics, marl_resp_times = run_policy_episodes(env, marl_policy, num_episodes, start_seed=start_seed, is_marl=True)

    # Build DataFrames
    df_greedy = pd.DataFrame(greedy_metrics)
    df_greedy["policy"] = "baseline"

    df_random = pd.DataFrame(random_metrics)
    df_random["policy"] = "random"

    df_marl = pd.DataFrame(marl_metrics)
    df_marl["policy"] = "marl"

    all_df = pd.concat([df_marl, df_greedy, df_random], ignore_index=True)
    csv_path = os.path.join(output_dir, "comparison.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"Saved evaluation raw metrics to {csv_path}")

    # Compute Statistical Metrics & Significance
    summary_rows = []
    metrics_list = ["task_completion_rate", "coordination_efficiency", "resource_utilization_rate", "avg_response_time"]

    for m in metrics_list:
        marl_vals = df_marl[m].values
        greedy_vals = df_greedy[m].values

        marl_mean, marl_std = np.mean(marl_vals), np.std(marl_vals)
        greedy_mean, greedy_std = np.mean(greedy_vals), np.std(greedy_vals)
        
        pct_imp = ((marl_mean - greedy_mean) / abs(greedy_mean + 1e-8)) * 100.0
        
        try:
            stat, p_val = stats.wilcoxon(marl_vals, greedy_vals)
        except Exception:
            stat, p_val = 0.0, 1.0

        summary_rows.append({
            "metric": m,
            "marl_mean_std": f"{marl_mean:.3f} ± {marl_std:.3f}",
            "baseline_mean_std": f"{greedy_mean:.3f} ± {greedy_std:.3f}",
            "pct_improvement": f"{pct_imp:+.2f}%",
            "p_value": f"{p_val:.4e}"
        })

    summary_df = pd.DataFrame(summary_rows)

    # Markdown Summary Report
    summary_md_path = os.path.join(output_dir, "summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Evaluation Summary Report\n\n")
        f.write(f"Evaluated over **{num_episodes} identical seed episodes**.\n\n")
        f.write("### 📊 Metric Performance Matrix (Mean ± Std)\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("### 📈 Key Insights\n")
        f.write(f"- **MARL Task Completion Rate:** {df_marl['task_completion_rate'].mean():.3f} vs Baseline {df_greedy['task_completion_rate'].mean():.3f}\n")
        f.write(f"- **MARL Coordination Efficiency:** {df_marl['coordination_efficiency'].mean():.3f} vs Baseline {df_greedy['coordination_efficiency'].mean():.3f}\n")

    print(f"Saved evaluation markdown summary to {summary_md_path}")

    # Plot figures
    plot_metric_bars(all_df, os.path.join(output_dir, "policy_comparison_bars.png"))
    plot_response_time_boxplot({
        "marl": marl_resp_times,
        "baseline": greedy_resp_times,
        "random": random_resp_times
    }, os.path.join(output_dir, "response_time_boxplot.png"))

    # Plot learning curve from DB if available
    db_cfg = config.get("db", {})
    try:
        db = DBLogger(db_cfg)
        curve_df = db.get_latest_training_curve()
        if not curve_df.empty:
            plot_learning_curve(curve_df, os.path.join(output_dir, "learning_curve.png"))
    except Exception as e:
        print(f"Note: Could not retrieve training curve from MongoDB: {e}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL Benchmark Evaluation")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MAPPO checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output", type=str, default="results", help="Output results directory")

    args = parser.parse_args()
    evaluate_all(config_path=args.config, checkpoint_path=args.checkpoint, num_episodes=args.episodes, output_dir=args.output)
