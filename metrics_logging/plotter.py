"""
Matplotlib Plotting Helpers for Evaluation and Training Results (FR9, Section 8).
Generates learning curves, metric comparison bar charts, and response-time box plots.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

def plot_learning_curve(df: pd.DataFrame, output_path: str = "results/learning_curve.png"):
    """Plots training learning curve (reward and losses vs update) with moving average."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    updates = pd.to_numeric(df["update_num"], errors="coerce").values
    rewards = pd.to_numeric(df["mean_reward"], errors="coerce").values

    ax1.plot(updates, rewards, alpha=0.3, color="blue", label="Raw Mean Reward")
    if len(rewards) >= 5:
        window = min(20, len(rewards))
        smooth = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
        ax1.plot(updates, smooth, color="navy", linewidth=2, label=f"{window}-Update Moving Avg")

    ax1.set_ylabel("Episode Reward")
    ax1.set_title("MAPPO CTDE Training Learning Curve")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    if "actor_loss" in df and "critic_loss" in df:
        a_loss = pd.to_numeric(df["actor_loss"], errors="coerce").values
        c_loss = pd.to_numeric(df["critic_loss"], errors="coerce").values
        ax2.plot(updates, a_loss, color="red", label="Actor Loss", alpha=0.8)
        ax2.plot(updates, c_loss, color="orange", label="Critic Loss", alpha=0.8)
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Training Updates")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved learning curve plot to {output_path}")

def plot_metric_bars(df: pd.DataFrame, output_path: str = "results/policy_comparison_bars.png"):
    """Bar chart comparing policies across key coordination metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if df.empty:
        return

    metrics_to_plot = ["task_completion_rate", "coordination_efficiency", "resource_utilization_rate"]
    policies = df["policy"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for idx, policy in enumerate(policies):
        p_df = df[df["policy"] == policy]
        means = [p_df[m].mean() for m in metrics_to_plot]
        stds = [p_df[m].std() for m in metrics_to_plot]
        ax.bar(x + idx * width, means, width, yerr=stds, capsize=5, label=policy.upper())

    ax.set_ylabel("Score / Rate")
    ax.set_title("Policy Performance Comparison (Mean ± Std)")
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(["Task Completion", "Coordination Eff.", "Resource Util."])
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved policy comparison bar chart to {output_path}")

def plot_response_time_boxplot(data_dict: Dict[str, List[float]], output_path: str = "results/response_time_boxplot.png"):
    """Boxplot comparing casualty response times across policies."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not data_dict:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(data_dict.keys())
    values = [data_dict[k] for k in labels]

    ax.boxplot(values, labels=[l.upper() for l in labels], patch_artist=True)
    ax.set_ylabel("Response Time (Steps)")
    ax.set_title("Per-Casualty Response Time Distribution")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved response time boxplot to {output_path}")

def plot_run_metrics(df: pd.DataFrame, output_path: str = "results/run_metrics.png"):
    """Alias helper for plotting run metric bar charts."""
    if "policy" in df:
        plot_metric_bars(df, output_path)
    else:
        plot_learning_curve(df, output_path)

