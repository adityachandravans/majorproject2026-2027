# Multi-Agent AI System for Disaster Response Coordination

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-ParallelEnv-green.svg)](https://pettingzoo.farama.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/Frontend-React_18_%2B_Vite-61dafb.svg)](https://reactjs.org/)

An end-to-end Multi-Agent Reinforcement Learning (MARL) framework for disaster response coordination built with **PettingZoo ParallelEnv**, **PyTorch MAPPO (Centralized Training with Decentralized Execution)**, **FastAPI + WebSockets**, and an interactive **React 18 + Vite + TypeScript SVG Dashboard**.

---

## 🏗️ Architecture & Stack Overview

```
                        +---------------------------------------+
                        |   React 18 + Vite SVG Frontend        |
                        |   (Live Grid, Charts, Controls)       |
                        +-------------------+-------------------+
                                            |  WebSocket / REST
                                            v
                        +---------------------------------------+
                        |      FastAPI + Uvicorn Backend        |
                        | (Simulation Manager, CORS, Latency)   |
                        +-------------------+-------------------+
                                            |
                +---------------------------+---------------------------+
                |                                                       |
                v                                                       v
+-------------------------------+                       +-------------------------------+
|  MARL Engine (MAPPO CTDE)     |                       | PettingZoo Disaster Env       |
|  - Role Actors (Medical/      |                       | - Depots & Stock Conservation |
|    Rescue/Logistics)          |                       | - Step-by-Step Manhattan Move |
|  - Centralized Critic V(s)    |                       | - Inter-Agent Comms Layer     |
+---------------+---------------+                       +---------------+---------------+
                |                                                       |
                +---------------------------+---------------------------+
                                            |
                                            v
                        +---------------------------------------+
                        |   MongoDB Logger (disaster_metrics)   |
                        |   (Runs, Episodes, Training Curve)    |
                        +---------------------------------------+
```

---

## 📁 Project Directory Structure

```
multiagentmodels/
├── config/            # YAML configurations (default_config.yaml, smoke_config.yaml)
├── env/               # DisasterResponseEnv, scenario_gen.py, comms.py
├── agents/            # MedicalAgent, RescueAgent, LogisticsAgent role implementations
├── baseline/          # Greedy rule-based policy & Random policy implementations
├── marl/              # MAPPO CTDE (networks.py, mappo.py, buffer.py, train.py, evaluate.py, inference.py)
├── metrics_logging/   # MongoDB DBLogger schema & Matplotlib plotter helpers
├── backend/           # FastAPI main.py, simulation_manager.py, schemas.py (Pydantic)
├── frontend/          # React 18 + Vite + TypeScript + SVG Grid + Recharts App
├── tests/             # Comprehensive pytest & vitest test suite
├── checkpoints/       # Saved PyTorch MAPPO model checkpoints (.pt)
├── results/           # Benchmark comparison CSVs, markdown summaries, and plot PNGs
├── docs/              # REPORT_RESULTS.md (Traceability Matrix) & DEMO_SCRIPT.md
├── Makefile           # Multi-platform build & run automation
├── requirements.txt   # Pinned Python dependencies
└── README.md          # System documentation & execution guide
```

---

## 🤖 Agent Roles & Action Spaces

| Role | Actions | Reward Rules |
| :--- | :--- | :--- |
| **Medical Agent** | `0..N-1`: Move to Zone<br>`N`: Treat Casualty<br>`N+1`: Transport to Hospital | $+10.0$ on-time treatment<br>$-1.0 \times \text{delay}$ step penalty<br>$-20.0$ lost casualty penalty |
| **Rescue Agent** | `0..N-1`: Move to Zone<br>`N`: Perform Rescue<br>`N+1`: Return Team to Depot | $+15.0$ per rescued person<br>$-0.5$ per remaining trapped person/step<br>$-10.0$ unrescued penalty at end |
| **Logistics Agent** | `0..N-1`: Deliver Supplies to Zone<br>`N..N+D-1`: Restock at Depot<br>`N+D`: Idle | $+8.0$ demand fulfilled<br>$-2.0 \times \text{shortage}$ penalty<br>$-5.0$ wasted delivery penalty |

---

## ⚡ Quickstart & Execution Guide

### 1. Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install React frontend dependencies:
```bash
cd frontend && npm install && npm run build && cd ..
```

### 2. Run Test Suite
Run end-to-end Python pytest and frontend Vitest suites:
```bash
python -m pytest -o pythonpath=.
cd frontend && npm test && cd ..
```
*(Or via Makefile: `make test`)*

### 3. One-Command Demo Launch
Launch the complete system (FastAPI backend + embedded React frontend at `http://localhost:8000`):
```bash
python scripts/demo.py
```
*(Or via Makefile: `make demo`)*

### 4. Train MAPPO (CTDE)
Train multi-agent PPO models from scratch or resume:
```bash
# Smoke test training (2,000 steps)
python marl/train.py --smoke

# Full training (200,000 steps)
python marl/train.py --timesteps 200000

# Resume training from latest checkpoint
python marl/train.py --resume
```

### 5. Benchmark Evaluation
Evaluate MAPPO vs Greedy Baseline vs Random Policy across identical seeds:
```bash
python marl/evaluate.py --episodes 100
```
Outputs `results/comparison.csv`, `results/summary.md`, and plot PNGs.

---

## 🔌 API Reference (FastAPI REST & WebSocket)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/config/default` | `GET` | Returns default YAML scenario configuration. |
| `/api/simulations` | `POST` | Creates a new simulation instance (`policy`, `seed`, `speed`). |
| `/api/simulations/{id}/state` | `GET` | Returns current state dictionary (`render_state()`). |
| `/api/simulations/{id}/start` | `POST` | Starts background asyncio step loop. |
| `/api/simulations/{id}/pause` | `POST` | Pauses background simulation loop. |
| `/api/simulations/{id}/step` | `POST` | Executes a single step synchronously. |
| `/api/simulations/{id}/policy` | `POST` | Switches policy mid-run (`marl`, `baseline`, `random`). |
| `/api/checkpoints` | `GET` | Lists available PyTorch model checkpoints. |
| `/api/runs` | `GET` | Queries registered runs from MongoDB. |
| `/api/compare` | `POST` | Runs side-by-side MARL vs Baseline comparison on identical seed. |
| `/ws/simulations/{id}` | `WS` | Real-time WebSocket streaming of simulation frames & metrics. |

---

## 🛠️ Design Decisions & Deviations from Legacy SRS

1. **Frontend Architecture:** Replaced legacy Streamlit (`dashboard/app.py`) with a modern **React 18 + Vite + TypeScript** web application with SVG grid rendering and real-time WebSocket streaming.
2. **MARL Framework:** Replaced legacy SB3 joint-action wrapper with a custom PyTorch **MAPPO CTDE** implementation supporting parameter-sharing per role, action masking, and decentralized execution.
3. **Resource Management:** Enforced explicit depot stock decrement/replenishment without negative values, requiring medical agents to hold ambulances and rescue agents to hold rescue teams.
4. **Local RNG Seeding:** Used `np.random.default_rng(seed)` and `random.Random(seed)` strictly local instances to guarantee exact scenario determinism across runs.
