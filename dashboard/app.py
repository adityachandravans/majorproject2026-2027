"""
Streamlit Dashboard for Multi-Agent Disaster Response Coordination System (Section 5.4).
Provides interactive grid visualization, live episode stepping, and training metric progress line charts.
"""

import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, run_baseline_episode
from metrics_logging.db_logger import DBLogger
from metrics_logging.plotter import plot_run_metrics

st.set_page_config(
    page_title="Multi-Agent Disaster Response System",
    page_icon="🚨",
    layout="wide"
)

# Initialize Session State
if "current_episode_metrics" not in st.session_state:
    st.session_state.current_episode_metrics = None
if "current_step_history" not in st.session_state:
    st.session_state.current_step_history = []

def render_grid_plot(state_data: dict, step_num: int):
    """Renders a color-coded 2D grid plot representing disaster zones, depots, and agents."""
    grid_w, grid_h = state_data["grid_size"]
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.set_xlim(-0.5, grid_w - 0.5)
    ax.set_ylim(-0.5, grid_h - 0.5)
    ax.set_xticks(range(grid_w))
    ax.set_yticks(range(grid_h))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"Simulation Grid State - Step {step_num}", fontsize=13, weight="bold")

    # Render Depots
    for depot in state_data["depots"]:
        dx, dy = depot["pos"]
        ax.plot(dx, dy, 's', color='navy', markersize=20, label="Depot")
        ax.text(dx, dy, f"🏥 D{depot['id']}", ha='center', va='center', color='white', fontsize=8, weight='bold')

    # Render Disaster Zones (Severity 1: Yellow, 2: Orange, 3: Red)
    sev_colors = {1: '#f39c12', 2: '#e67e22', 3: '#e74c3c'}
    for zone in state_data["zones"]:
        zx, zy = zone["pos"]
        c_color = sev_colors.get(zone["severity"], 'red')
        ax.plot(zx, zy, 'o', color=c_color, markersize=26, alpha=0.8)
        
        info_str = f"Z{zone['id']} (S{zone['severity']})\nC:{zone['casualty_count']}|T:{zone['trapped_count']}"
        ax.text(zx, zy, info_str, ha='center', va='center', color='black', fontsize=7, weight='bold')

    # Render Agents
    role_icons = {"medical": "🚑", "rescue": "🛠️", "logistics": "🚚"}
    role_offsets = {"medical": (-0.2, 0.2), "rescue": (0.2, 0.2), "logistics": (0.0, -0.2)}

    for agent_id, a_data in state_data["agents"].items():
        ax_pos, ay_pos = a_data["pos"]
        role = a_data["role"]
        icon = role_icons.get(role, "🤖")
        off_x, off_y = role_offsets.get(role, (0, 0))
        
        ax.text(
            ax_pos + off_x, ay_pos + off_y,
            f"{icon} {agent_id}",
            fontsize=9, weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9)
        )

    plt.tight_layout()
    return fig


def main():
    st.title("🚨 Multi-Agent AI System for Disaster Response Coordination")
    st.caption("B.Tech Final Year Project Demo - Phase 1+2 & Phase 3 Milestone")

    # Sidebar Controls
    st.sidebar.header("⚙️ System Controls & Config")
    config_path = st.sidebar.text_input("Config Path", "config/default_config.yaml")
    
    if not os.path.exists(config_path):
        st.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    db_path = config.get("db", {}).get("path", "disaster_metrics.db")
    db = DBLogger(db_path)

    st.sidebar.subheader("Database & Execution")
    runs_list = db.get_all_runs()
    run_options = {f"{r['run_id']} ({r['run_type']})": r['run_id'] for r in runs_list}
    
    selected_run_label = st.sidebar.selectbox("Select Logged Run", list(run_options.keys()) if run_options else ["No Runs Logged Yet"])
    
    if st.sidebar.button("⚡ Run Live Baseline Episode"):
        env = DisasterResponseEnv(config)
        policy = GreedyPolicy(config)
        with st.spinner("Executing Greedy Baseline Policy Episode..."):
            metrics = run_baseline_episode(env, policy)
            run_id = db.start_run(run_type="baseline", config=config, notes="Interactive Streamlit Run")
            db.log_episode(run_id, 1, metrics)
            st.session_state.current_episode_metrics = metrics
            st.session_state.current_step_history = metrics["step_history"]
            st.success(f"Executed & Logged Live Episode! Run ID: {run_id}")
            st.rerun()

    # Dashboard Tabs
    tab1, tab2 = st.tabs(["🗺️ Live Grid & Episode Stepping", "📊 Logged Metrics & Training Curves"])

    with tab1:
        st.subheader("Interactive Environment Grid & Agent Playback")
        
        if st.session_state.current_step_history:
            history = st.session_state.current_step_history
            max_step = len(history) - 1
            
            step_idx = st.slider("Step Playback Slider", 0, max_step, 0)
            step_data = history[step_idx]

            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = render_grid_plot(step_data["state"], step_data["step"])
                st.pyplot(fig)

            with col2:
                st.markdown(f"### Step {step_data['step']} Actions & Status")
                st.json(step_data["actions"])
                
                st.markdown("### Zone Severities & Casualty Status")
                zones_df = pd.DataFrame([
                    {
                        "Zone": f"Z{z['id']}",
                        "Severity": z["severity"],
                        "Casualties Left": z["casualty_count"],
                        "Trapped Left": z["trapped_count"],
                        "Treated": z["casualties_treated"]
                    }
                    for z in step_data["state"]["zones"]
                ])
                st.dataframe(zones_df, use_container_width=True)

        else:
            st.info("Click '⚡ Run Live Baseline Episode' in the sidebar to simulate an interactive episode step-by-step!")

            # Display initial grid reset preview
            env_preview = DisasterResponseEnv(config)
            obs, _ = env_preview.reset(seed=config["environment"]["seed"])
            fig = render_grid_plot(env_preview.state_data, 0)
            st.pyplot(fig)

    with tab2:
        st.subheader("Episode Metrics Progression & Multi-Agent Analytics")

        if run_options and selected_run_label in run_options:
            selected_run_id = run_options[selected_run_label]
            df_episodes = db.get_run_episodes(selected_run_id)

            if not df_episodes.empty:
                # Top Row Metrics Cards
                latest = df_episodes.iloc[-1]
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Task Completion", f"{latest['task_completion_rate']*100:.1f}%")
                m2.metric("Avg Response Time", f"{latest['avg_response_time']:.1f} steps")
                m3.metric("Resource Util.", f"{latest['resource_utilization_rate']*100:.1f}%")
                m4.metric("Coordination Eff.", f"{latest['coordination_efficiency']*100:.1f}%")
                m5.metric("Total Reward", f"{latest['total_reward']:.1f}")

                st.markdown("---")

                # Metrics Charts
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Reward per Episode")
                    st.line_chart(df_episodes.set_index("episode_num")[["total_reward", "medical_reward", "rescue_reward", "logistics_reward"]])
                with c2:
                    st.markdown("#### Task Completion & Coordination Efficiency (%)")
                    eff_df = df_episodes.copy()
                    eff_df["Task Completion (%)"] = eff_df["task_completion_rate"] * 100
                    eff_df["Coordination Efficiency (%)"] = eff_df["coordination_efficiency"] * 100
                    st.line_chart(eff_df.set_index("episode_num")[["Task Completion (%)", "Coordination Efficiency (%)"]])

                st.markdown("#### Logged Database Records")
                st.dataframe(df_episodes, use_container_width=True)
            else:
                st.warning("No episode records found for selected run.")
        else:
            st.info("No database runs available. Run training or baseline episodes to log metrics.")

if __name__ == "__main__":
    main()
