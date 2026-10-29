"""
FastAPI Backend Application with WebSocket Streaming and REST Endpoints (FR5, FR6, FR8, NFR1).
"""

import os
import sys
import yaml
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import CreateSimRequest, SwitchPolicyRequest, CompareRequest, SimStateResponse, CheckpointInfo
from backend.simulation_manager import SimulationManager
from metrics_logging.db_logger import DBLogger
from baseline.greedy_policy import GreedyPolicy
from marl.inference import load_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackendAPI")

app = FastAPI(
    title="Multi-Agent Disaster Response Coordination API",
    version="2.0.0",
    description="FastAPI + WebSockets Backend for MARL Disaster Response"
)

_frontend_origins = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sim_manager = SimulationManager()
_default_cfg_path = "config/default_config.yaml"
_init_cfg = {}
if os.path.exists(_default_cfg_path):
    with open(_default_cfg_path, "r") as _f:
        _init_cfg = yaml.safe_load(_f) or {}
db_logger = DBLogger(_init_cfg.get("db", {}))


@app.get("/api/config/default")
def get_default_config():
    config_path = "config/default_config.yaml"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Default config file not found.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@app.post("/api/simulations", response_model=SimStateResponse)
def create_simulation(req: CreateSimRequest):
    default_cfg = get_default_config()
    cfg = req.config if req.config else default_cfg
    
    sim = sim_manager.create_simulation(
        config=cfg,
        policy=req.policy,
        checkpoint=req.checkpoint,
        seed=req.seed,
        speed=req.speed
    )
    return {"sim_id": sim.sim_id, "state": sim.get_state()}


@app.get("/api/simulations/{sim_id}/state")
def get_simulation_state(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    return sim.get_state()


@app.post("/api/simulations/{sim_id}/start")
async def start_simulation(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    sim.start()
    return sim.get_state()


@app.post("/api/simulations/{sim_id}/pause")
def pause_simulation(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    sim.pause()
    return sim.get_state()


@app.post("/api/simulations/{sim_id}/resume")
async def resume_simulation(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    sim.start()
    return sim.get_state()


@app.post("/api/simulations/{sim_id}/reset")
def reset_simulation(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    sim.pause()
    return sim.reset()


@app.post("/api/simulations/{sim_id}/step")
def step_simulation(sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    return sim.step_sync()


@app.post("/api/simulations/{sim_id}/policy")
def switch_simulation_policy(sim_id: str, req: SwitchPolicyRequest):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    sim.switch_policy(req.policy, req.checkpoint)
    return sim.get_state()


@app.get("/api/checkpoints", response_model=List[CheckpointInfo])
def list_checkpoints():
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return []

    checkpoints = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".zip"):
            full_p = os.path.join(ckpt_dir, f)
            stat = os.stat(full_p)
            from datetime import datetime
            checkpoints.append({
                "filename": f,
                "path": full_p,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return checkpoints


@app.get("/api/runs")
def get_all_runs():
    return db_logger.get_all_runs()


@app.get("/api/runs/{run_id}/episodes")
def get_run_episodes(run_id: str):
    df = db_logger.get_run_episodes(run_id)
    return df.to_dict(orient="records")


@app.get("/api/training/{run_id}/curve")
def get_training_curve(run_id: str):
    df = db_logger.get_training_curve(run_id)
    return df.to_dict(orient="records")


@app.post("/api/compare")
def compare_policies_side_by_side(req: CompareRequest):
    """Runs same scenario and seed under MARL and baseline and returns both step histories & metrics."""
    default_cfg = get_default_config()
    cfg = req.config if req.config else default_cfg
    if req.max_steps:
        cfg["environment"]["max_steps"] = req.max_steps

    seed = req.seed

    # 1. MARL Run
    sim_marl = sim_manager.create_simulation(cfg, policy="marl", seed=seed)
    marl_history = []
    while sim_marl.status != "completed" and sim_marl.env.agents:
        st = sim_marl.step_sync()
        marl_history.append(st)

    # 2. Baseline Run
    sim_base = sim_manager.create_simulation(cfg, policy="baseline", seed=seed)
    base_history = []
    while sim_base.status != "completed" and sim_base.env.agents:
        st = sim_base.step_sync()
        base_history.append(st)

    return {
        "seed": seed,
        "marl_metrics": sim_marl.get_state()["metrics"],
        "baseline_metrics": sim_base.get_state()["metrics"],
        "marl_history": marl_history,
        "baseline_history": base_history
    }


@app.get("/api/results/summary")
def get_evaluation_summary():
    summary_md_path = "results/summary.md"
    csv_path = "results/comparison.csv"
    
    summary_text = ""
    if os.path.exists(summary_md_path):
        with open(summary_md_path, "r") as f:
            summary_text = f.read()

    data = []
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        data = df.to_dict(orient="records")

    return {
        "summary_markdown": summary_text,
        "raw_data": data
    }


@app.websocket("/ws/simulations/{sim_id}")
async def websocket_simulation(websocket: WebSocket, sim_id: str):
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        await websocket.close(code=4004, reason="Simulation not found.")
        return

    await websocket.accept()
    sim.ws_clients.add(websocket)
    
    # Send initial state frame
    await websocket.send_json(sim.get_state())

    try:
        while True:
            # Listen for client control commands (e.g. step, start, pause, policy switch)
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "start":
                sim.start()
            elif action == "pause":
                sim.pause()
            elif action == "step":
                st = sim.step_sync()
                await websocket.send_json(st)
            elif action == "reset":
                sim.pause()
                st = sim.reset()
                await websocket.send_json(st)
            elif action == "switch_policy":
                pol = cmd.get("policy", "baseline")
                sim.switch_policy(pol)
                await websocket.send_json(sim.get_state())
    except WebSocketDisconnect:
        sim.ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in sim.ws_clients:
            sim.ws_clients.remove(websocket)


# Serve React static frontend bundle if built
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
