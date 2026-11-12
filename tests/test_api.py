"""
Unit and integration tests for FastAPI REST API and WebSocket endpoints (Stage 4).
"""

import pytest
import json
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_get_default_config():
    response = client.get("/api/config/default")
    assert response.status_code == 200
    cfg = response.json()
    assert "environment" in cfg
    assert "rewards" in cfg
    assert "db" not in cfg

def test_health_check_includes_database_backend():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] in {"mongo", "sqlite"}

def test_production_cors_preflight():
    response = client.options(
        "/api/simulations",
        headers={
            "Origin": "https://majorproject2026-2027.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://majorproject2026-2027.vercel.app"

def test_cors_header_is_present_on_server_errors(monkeypatch):
    def fail_create_simulation(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("backend.main.sim_manager.create_simulation", fail_create_simulation)
    test_client = TestClient(app, raise_server_exceptions=False)
    response = test_client.post(
        "/api/simulations",
        headers={"Origin": "https://majorproject2026-2027.vercel.app"},
        json={"policy": "baseline"},
    )
    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "https://majorproject2026-2027.vercel.app"

def test_create_simulation_and_step():
    payload = {
        "policy": "baseline",
        "seed": 42,
        "speed": 2.0
    }
    res = client.post("/api/simulations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "sim_id" in data
    sim_id = data["sim_id"]

    # Step simulation
    step_res = client.post(f"/api/simulations/{sim_id}/step")
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["step_count"] == 1
    assert "last_step_latency_ms" in step_data

def test_start_and_resume_simulation():
    res = client.post("/api/simulations", json={"policy": "baseline", "seed": 42})
    sim_id = res.json()["sim_id"]

    start_res = client.post(f"/api/simulations/{sim_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "running"

    pause_res = client.post(f"/api/simulations/{sim_id}/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "paused"

    resume_res = client.post(f"/api/simulations/{sim_id}/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "running"

    client.post(f"/api/simulations/{sim_id}/pause")

def test_switch_policy_mid_run():
    res = client.post("/api/simulations", json={"policy": "baseline", "seed": 42})
    sim_id = res.json()["sim_id"]

    switch_res = client.post(f"/api/simulations/{sim_id}/policy", json={"policy": "random"})
    assert switch_res.status_code == 200
    assert switch_res.json()["policy"] == "random"

def test_compare_api():
    res = client.post("/api/compare", json={"seed": 42, "max_steps": 5})
    assert res.status_code == 200
    data = res.json()
    assert "marl_metrics" in data
    assert "baseline_metrics" in data
    assert "marl_history" in data
    assert "baseline_history" in data

def test_checkpoints_api():
    res = client.get("/api/checkpoints")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_runs_api():
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_websocket_stream():
    res = client.post("/api/simulations", json={"policy": "baseline", "seed": 42})
    sim_id = res.json()["sim_id"]

    with client.websocket_connect(f"/ws/simulations/{sim_id}") as websocket:
        initial_frame = websocket.receive_json()
        assert initial_frame["sim_id"] == sim_id

        # Send step command over WS
        websocket.send_json({"action": "step"})
        step_frame = websocket.receive_json()
        assert step_frame["step_count"] >= 1
