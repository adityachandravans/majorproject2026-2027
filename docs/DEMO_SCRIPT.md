# 5-Minute Evaluator Demo Script

**Project Title:** Multi-Agent AI System for Disaster Response Coordination  
**Target Audience:** Final Year B.Tech Evaluation Committee / Evaluators

---

## 🎯 Step 1: System Startup (0:00 - 0:30)
1. Open a terminal in the project root (`multiagentmodels/`).
2. Launch the single-command demo launcher:
   ```bash
   python scripts/demo.py
   ```
   *(Or via Make: `make demo`)*
3. The launcher automatically builds the React frontend (if needed), starts the FastAPI backend server on `http://localhost:8000`, and opens the browser.

---

## ⚡ Step 2: Live Multi-Agent Simulation Playback (0:30 - 2:00)
1. Point out the interactive SVG Grid Map on the **Live Simulation** tab:
   - **Disaster Zones:** Color-coded by severity (Yellow: Low, Orange: Medium, Red: High) with live casualty (`C`) and trapped (`T`) count badges. Point out inaccessible zones marked with dashed borders.
   - **Depots:** Blue squares showing live resource stock bars (Ambulances, Rescue Teams, Supply Units).
   - **Agent Markers:** Color-coded circles with role letters: **Medical (`M`, Red)**, **Rescue (`R`, Yellow)**, **Logistics (`L`, Green)**, with dashed target vector lines pointing toward their destination.
2. Demonstrate real-time controls:
   - Click **▶ Start** to run the simulation loop streaming at 60 FPS over WebSocket.
   - Adjust the **Speed Slider** (0.5x to 5.0x).
   - Show the **Mid-Run Policy Toggle**: Switch from **MARL (MAPPO)** to **Greedy Baseline** live while agents are executing!
   - Show live Recharts metrics updating real-time: Task Completion Rate, Coordination Efficiency, Resource Utilization, Avg Response Time.

---

## ⚙️ Step 3: Scenario Configuration & Presets (2:00 - 3:00)
1. Navigate to the **Scenario Config** tab.
2. Click the preset buttons:
   - **🌋 Earthquake:** 12x12 grid, high rubble blockage (40% inaccessible), 3 Medical / 4 Rescue / 2 Logistics agents.
   - **🌊 Flood:** 10x10 grid, 30% inaccessible terrain.
   - **🌀 Cyclone:** 15x15 large-scale grid.
3. Click **🚀 Launch Simulation** to dynamically spin up a custom PettingZoo instance.

---

## ⚔️ Step 4: Head-to-Head Policy Comparison (3:00 - 4:00)
1. Navigate to the **Compare Policies** tab.
2. Click **⚔️ Run MARL vs Baseline Comparison**.
3. Point out:
   - **Synchronized Dual Grid Maps:** Showing exact side-by-side agent behavior on the identical seed.
   - **Metric Comparison Table:** Highlights the winning policy for Task Completion, Coordination Efficiency, and Casualty Response Time.
   - **Bar Chart:** Visualizing relative performance gains.

---

## 📊 Step 5: Training Curves & MongoDB Metrics (4:00 - 5:00)
1. Navigate to the **Training & Results** tab.
2. Inspect:
   - **MongoDB Training Curve:** Episode reward, Actor loss, and Critic loss over training updates.
   - **Saved Model Checkpoints:** List of model checkpoints in `checkpoints/`.
   - **Evaluation Benchmark Matrix:** Quantitative mean ± std metrics across 100 evaluation seeds with Wilcoxon significance test p-values.
