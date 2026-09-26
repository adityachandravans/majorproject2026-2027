"""
Matplotlib Plotter helper to visualize training and evaluation metrics from SQLite DB logs.
"""
import matplotlib.pyplot as plt
import pandas as pd
from metrics_logging.db_logger import DBLogger

def plot_run_metrics(db_path: str, run_id: str, save_path: str = None):
    """
    Queries SQLite database for episode metrics of run_id and generates a 4-panel matplotlib chart.
    """
    db = DBLogger(db_path)
    df = db.get_run_episodes(run_id)

    if df.empty:
        print(f"No episode data found for run_id: {run_id}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Episode Metrics Progression (Run: {run_id})", fontsize=14, weight='bold')

    # Panel 1: Total Reward
    axes[0, 0].plot(df["episode_num"], df["total_reward"], marker='o', color='b', label='Total Reward')
    axes[0, 0].plot(df["episode_num"], df["medical_reward"], linestyle='--', color='g', label='Medical')
    axes[0, 0].plot(df["episode_num"], df["rescue_reward"], linestyle='--', color='orange', label='Rescue')
    axes[0, 0].plot(df["episode_num"], df["logistics_reward"], linestyle='--', color='purple', label='Logistics')
    axes[0, 0].set_title("Reward per Episode & Role")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Panel 2: Task Completion Rate
    axes[0, 1].plot(df["episode_num"], df["task_completion_rate"] * 100.0, marker='s', color='g')
    axes[0, 1].set_title("Task Completion Rate (%)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Completion Rate (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(True)

    # Panel 3: Response Time
    axes[1, 0].plot(df["episode_num"], df["avg_response_time"], marker='^', color='r')
    axes[1, 0].set_title("Average Response Time (Steps)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(True)

    # Panel 4: Coordination Efficiency & Resource Utilization
    axes[1, 1].plot(df["episode_num"], df["coordination_efficiency"] * 100.0, marker='d', color='purple', label='Coordination Efficiency (%)')
    axes[1, 1].plot(df["episode_num"], df["resource_utilization_rate"] * 100.0, marker='v', color='teal', label='Resource Utilization (%)')
    axes[1, 1].set_title("Efficiency & Utilization Metrics")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Metrics plot saved to {save_path}")

    return fig
