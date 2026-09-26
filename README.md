# Multi-Agent AI System for Disaster Response Coordination

A MARL-ready (Multi-Agent Reinforcement Learning) system for disaster response coordination built with **PettingZoo (ParallelEnv)**, **Stable-Baselines3 (PPO)**, **SQLite**, and **Streamlit**.

This codebase represents **Phase 1, Phase 2, and early Phase 3** of the 12-week SRS timeline (40% B.Tech Final Year Project milestone).

---

## 🏗️ Architecture & Tech Stack

| Component | Framework / Library | Justification |
| :--- | :--- | :--- |
| **Environment** | PettingZoo (`ParallelEnv` API) | Native multi-agent parallel execution environment API. |
| **RL Framework** | Stable-Baselines3 + Custom Wrapper | **Choice Rationale:** SB3 PPO with a centralized multi-agent joint-action wrapper was chosen over RLlib to avoid heavy Ray dependencies, ensure clean cross-platform installation on Windows, and provide an instant, robust training loop. |
| **Dashboard** | Streamlit | Minimal, fast interactive web UI for live grid playback and metric visualization. |
| **Logging** | SQLite (`sqlite3`) | Persistent storage of runs and per-episode coordination metrics. |
| **Config** | PyYAML (`config/default_config.yaml`) | Centralized scenario and hyperparameter configuration. |

---

## 📁 Repository Structure

```
/config/           - Default and smoke test YAML configurations
/env/              - DisasterResponseEnv (PettingZoo ParallelEnv) & Scenario Generator
/agents/           - MedicalAgent, RescueAgent, LogisticsAgent role implementations
/baseline/         - Greedy rule-based policy implementation & runner
/metrics_logging/  - SQLite DBLogger schema & matplotlib plotter helper
/training/         - MARL PPO training script & custom Gymnasium joint wrapper
/dashboard/        - Streamlit interactive application
/tests/            - End-to-end smoke test script
requirements.txt   - Pinned dependencies
README.md          - System documentation & execution guide
```

---

## 🤖 Agent Specifications & Reward Formulas (Section 5.2, Table 2)

### 1. Medical Agent (`agents/medical.py`)
* **Observation:** Casualty locations, casualty severities (1-3), ambulance availability, agent position, status.
* **Action Space (`Discrete(num_zones + 2)`):**
  * `0 .. num_zones-1`: Dispatch/move to target zone.
  * `num_zones`: Treat casualty at current zone.
  * `num_zones + 1`: Transport casualty to hospital/depot.
* **Reward Formula:**
  $$\text{Reward} = +10.0 \times \text{Casualties Treated (on time)} - 1.0 \times \text{Delay Steps} - 20.0 \times \text{Lost Casualties}$$

### 2. Rescue Agent (`agents/rescue.py`)
* **Observation:** Trapped population count, terrain accessibility, agent position, status.
* **Action Space (`Discrete(num_zones + 2)`):**
  * `0 .. num_zones-1`: Move to target zone.
  * `num_zones`: Perform rescue operation (frees trapped population into casualty pool).
  * `num_zones + 1`: Request backup from depot.
* **Reward Formula:**
  $$\text{Reward} = +15.0 \times \text{Rescued Persons} - 0.5 \times \text{Trapped Population Remaining} - 10.0 \times \text{Unrescued Penalty}$$

### 3. Logistics Agent (`agents/logistics.py`)
* **Observation:** Depot stock levels (ambulances, rescue teams, supplies), zone supply demand, payload position.
* **Action Space (`Discrete(num_zones + num_depots + 1)`):**
  * `0 .. num_zones-1`: Deliver supplies to zone.
  * `num_zones .. num_zones + num_depots - 1`: Restock at depot.
  * `num_zones + num_depots`: Idle.
* **Reward Formula:**
  $$\text{Reward} = +8.0 \times \text{Supplies Delivered (Active Demand)} - 2.0 \times \text{Shortage Penalty} - 5.0 \times \text{Wasted Delivery}$$

---

## ⚡ Quickstart & Execution Guide

### 1. Installation
Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Smoke Test (Section 6)
Executes a fixed 3x3 scenario (1 zone, 1 agent per role, 5 steps max) end-to-end to verify environment, baseline policy, and SQLite logging:
```bash
python tests/test_smoke.py
```
*(Or via pytest: `python -m pytest tests/test_smoke.py`)*

### 3. Run Baseline Policy Demo
Run a full episode using the rule-based greedy baseline:
```bash
python -c "import yaml; from env.disaster_env import DisasterResponseEnv; from baseline.greedy_policy import GreedyPolicy, run_baseline_episode; from metrics_logging.db_logger import DBLogger; config = yaml.safe_load(open('config/default_config.yaml')); env = DisasterResponseEnv(config); policy = GreedyPolicy(config); db = DBLogger(config['db']['path']); metrics = run_baseline_episode(env, policy); run_id = db.start_run('baseline', config, 'Baseline Demo'); db.log_episode(run_id, 1, metrics); print('Baseline execution completed! Total reward:', metrics['total_reward'])"
```

### 4. Run MARL PPO Training Pipeline (Section 7)
Train the multi-agent system for a short smoke training run (500 timesteps) or full training:
```bash
python training/train_marl.py
```
This saves model checkpoints in `/checkpoints/` and logs episode metrics to `disaster_metrics.db`.

### 5. Launch Interactive Streamlit Dashboard (Section 5.4)
Launch the interactive web visualization dashboard:
```bash
streamlit run dashboard/app.py
```
* View color-coded disaster zones (Severity 1-3), agent positions, and depot stock levels.
* Step through episodes using the interactive step playback slider.
* Trigger live baseline episodes with one click.
* Inspect line charts of Total Reward, Task Completion Rate, Response Time, and Coordination Efficiency over episodes.

---

## 📊 SQLite Metrics Schema (Section 8, FR7)
Database file: `disaster_metrics.db`
* **`runs` Table:** `run_id`, `run_type` (`baseline` | `marl_training`), `config_used`, `timestamp`, `notes`
* **`episodes` Table:** `episode_id`, `run_id`, `episode_num`, `total_reward`, `avg_response_time`, `resource_utilization_rate`, `casualties_served`, `task_completion_rate`, `coordination_efficiency`, `medical_reward`, `rescue_reward`, `logistics_reward`

---

## 🎓 Evaluator Defense Notes
* **Separation of Concerns:** Scenario generation is isolated in `env/scenario_gen.py`, allowing independent tuning of casualty ranges and zone severities.
* **Coordination Efficiency Metric:** Quantified as $1.0 - \frac{\text{duplicated actions}}{\text{total agent actions}}$, penalizing redundant agent actions at the same location.
* **MARL Multi-Agent Wrapper:** Implements joint observation vector concatenation and multi-discrete action mapping to interface PettingZoo directly with SB3 PPO.
