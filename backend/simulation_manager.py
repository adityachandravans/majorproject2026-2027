"""
Simulation Manager for Multi-Agent Disaster Response API (FR5, FR6, NFR1).
Manages simulation lifecycles, mid-run policy switching, background asyncio step loops, and WebSocket broadcasting.
"""

import os
import time
import asyncio
import uuid
import yaml
import logging
from typing import Dict, Any, Set, Optional
from fastapi import WebSocket

from env.disaster_env import DisasterResponseEnv
from baseline.greedy_policy import GreedyPolicy, RandomPolicy
from marl.inference import load_policy, MAPPOPolicy
from marl.mappo import MAPPOTrainer
from metrics_logging.db_logger import DBLogger

logger = logging.getLogger("SimulationManager")

class SimulationInstance:
    def __init__(self, sim_id: str, config: dict, policy_type: str = "marl", checkpoint: str = None, seed: int = 42, speed: float = 1.0):
        self.sim_id = sim_id
        self.config = config
        self.seed = seed
        self.speed = speed
        self.status = "created"  # created, running, paused, completed
        self.last_latency_ms = 0.0

        # Initialize PettingZoo Environment
        self.env = DisasterResponseEnv(config)
        self.obs_dict, _ = self.env.reset(seed=self.seed)

        # Policy setup
        self.policy_type = policy_type
        self.checkpoint = checkpoint
        self.policy_obj = self._create_policy(policy_type, checkpoint)

        # MongoDB Logger
        db_cfg = config.get("db", {})
        self.db = DBLogger(db_cfg)
        self.run_id = self.db.start_run(run_type="api_simulation", config=config, policy=policy_type, seed=seed)

        # WebSockets & Background Task
        self.ws_clients: Set[WebSocket] = set()
        self.step_task: Optional[asyncio.Task] = None

    def _create_policy(self, policy_type: str, checkpoint_path: Optional[str] = None):
        if policy_type == "marl":
            if checkpoint_path and os.path.exists(checkpoint_path):
                return load_policy(checkpoint_path)
            
            # Check default checkpoints
            best_ckpt = os.path.join(self.config.get("training", {}).get("checkpoint_dir", "checkpoints"), "mappo_best.pt")
            latest_ckpt = os.path.join(self.config.get("training", {}).get("checkpoint_dir", "checkpoints"), "mappo_latest.pt")
            
            if os.path.exists(best_ckpt):
                return load_policy(best_ckpt)
            elif os.path.exists(latest_ckpt):
                return load_policy(latest_ckpt)
            else:
                # Fallback to initialized MAPPO policy
                possible_agents = self.env.possible_agents
                obs_dims = {a: self.env.observation_space(a).shape[0] for a in possible_agents}
                action_dims = {a: self.env.action_space(a).n for a in possible_agents}
                global_state_dim = sum(obs_dims.values())
                trainer = MAPPOTrainer(self.config, possible_agents, obs_dims, global_state_dim, action_dims)
                return MAPPOPolicy(trainer.actors)

        elif policy_type == "baseline":
            return GreedyPolicy(self.config)
        elif policy_type == "random":
            return RandomPolicy(self.config)
        else:
            return GreedyPolicy(self.config)

    def switch_policy(self, new_policy_type: str, checkpoint_path: Optional[str] = None):
        """Switches policy mid-run seamlessly (FR5)."""
        self.policy_type = new_policy_type
        self.checkpoint = checkpoint_path
        self.policy_obj = self._create_policy(new_policy_type, checkpoint_path)
        logger.info(f"Simulation {self.sim_id} switched policy mid-run to {new_policy_type}.")

    def step_sync(self) -> dict:
        """Executes a single environment step synchronously and logs per-step latency (NFR1)."""
        if not self.env.agents or self.env.state_data.get("done", False):
            self.status = "completed"
            return self.get_state()

        start_time = time.perf_counter()

        if self.policy_type == "marl":
            actions = self.policy_obj.act(self.obs_dict, deterministic=True)
        else:
            actions = {
                agent_id: self.policy_obj.select_action(agent_id, self.env.agent_objects[agent_id].role, self.env.state_data)
                for agent_id in self.env.agents
            }

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.obs_dict = next_obs if next_obs else self.obs_dict

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_latency_ms = elapsed_ms
        logger.debug(f"Sim {self.sim_id} Step {self.env.step_count} latency: {elapsed_ms:.2f} ms")

        state = self.get_state()
        state["last_step_latency_ms"] = round(elapsed_ms, 2)
        state["actions"] = {str(k): int(v) for k, v in actions.items()}
        state["rewards"] = {str(k): float(v) for k, v in rewards.items()}

        if state["done"] or not self.env.agents:
            self.status = "completed"
            self.db.log_episode(self.run_id, 1, state["metrics"])

        return state

    def get_state(self) -> dict:
        state = self.env.render_state()
        state["sim_id"] = self.sim_id
        state["policy"] = self.policy_type
        state["status"] = self.status
        state["last_step_latency_ms"] = round(self.last_latency_ms, 2)
        return state

    def reset(self):
        self.obs_dict, _ = self.env.reset(seed=self.seed)
        self.status = "created"
        self.last_latency_ms = 0.0
        return self.get_state()

    async def broadcast_state(self, state: dict):
        if not self.ws_clients:
            return
        
        disconnected = set()
        for ws in self.ws_clients:
            try:
                await ws.send_json(state)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.ws_clients.remove(ws)

    async def _run_loop(self):
        self.status = "running"
        try:
            while self.status == "running":
                state = self.step_sync()
                await self.broadcast_state(state)

                if state["done"] or self.status == "completed":
                    self.status = "completed"
                    break

                delay = max(0.05, 1.0 / max(0.1, self.speed))
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.status = "paused"

    def start(self):
        if self.status != "running":
            self.status = "running"
            self.step_task = asyncio.create_task(self._run_loop())

    def pause(self):
        if self.step_task and not self.step_task.done():
            self.step_task.cancel()
        self.status = "paused"


class SimulationManager:
    def __init__(self):
        self.simulations: Dict[str, SimulationInstance] = {}

    def create_simulation(self, config: dict, policy: str = "marl", checkpoint: str = None, seed: int = 42, speed: float = 1.0) -> SimulationInstance:
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        instance = SimulationInstance(sim_id, config, policy_type=policy, checkpoint=checkpoint, seed=seed, speed=speed)
        self.simulations[sim_id] = instance
        return instance

    def get_simulation(self, sim_id: str) -> Optional[SimulationInstance]:
        return self.simulations.get(sim_id)

    def delete_simulation(self, sim_id: str):
        if sim_id in self.simulations:
            self.simulations[sim_id].pause()
            del self.simulations[sim_id]
