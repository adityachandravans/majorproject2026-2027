"""
MongoDB Database Logger for Disaster Response Coordination System.
Provides collection management, run tracking, episode metrics, and training curve logging via PyMongo.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pymongo
import pandas as pd


class DBLogger:
    def __init__(
        self,
        uri_or_config: Any = None,
        db_name: Optional[str] = None,
        client: Optional[pymongo.MongoClient] = None
    ):
        """
        Initializes MongoDB connection and sets up collections & indexes.
        Accepts:
            uri_or_config: MongoDB URI string, a config dictionary (e.g. config.get('db')),
                           or a legacy database name/file string.
            db_name: Optional explicit database name.
            client: Optional pre-existing MongoClient instance.
        """
        uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        resolved_db_name = db_name or os.environ.get("MONGODB_DATABASE")

        if isinstance(uri_or_config, dict):
            uri = uri_or_config.get("uri", uri)
            resolved_db_name = uri_or_config.get("database") or uri_or_config.get("db_name") or resolved_db_name
        elif isinstance(uri_or_config, str):
            if uri_or_config.startswith("mongodb://") or uri_or_config.startswith("mongodb+srv://"):
                uri = uri_or_config
            else:
                # Legacy file path or database name (e.g., 'disaster_metrics.db' -> 'disaster_metrics')
                base_name = os.path.basename(uri_or_config)
                clean_name = os.path.splitext(base_name)[0]
                resolved_db_name = resolved_db_name or clean_name

        self.uri = uri
        self.db_name = resolved_db_name or "disaster_metrics"

        self.client = client or pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.db_name]

        self.runs = self.db["runs"]
        self.episodes = self.db["episodes"]
        self.training_curve = self.db["training_curve"]
        self.counters = self.db["counters"]

        self._init_db()

    def _init_db(self):
        """Creates collection indexes and ensures counter collections are initialized."""
        # Indexes on runs
        self.runs.create_index("run_id", unique=True)
        self.runs.create_index([("timestamp", pymongo.DESCENDING)])

        # Indexes on episodes
        self.episodes.create_index([("run_id", pymongo.ASCENDING), ("episode_num", pymongo.ASCENDING)])
        self.episodes.create_index("episode_id")

        # Indexes on training_curve
        self.training_curve.create_index([("run_id", pymongo.ASCENDING), ("update_num", pymongo.ASCENDING)])
        self.training_curve.create_index("curve_id")

        # Initialize sequence counters from existing max IDs if not set
        if self.counters.find_one({"_id": "episode_id"}) is None:
            max_ep = self.episodes.find_one(sort=[("episode_id", pymongo.DESCENDING)])
            max_ep_val = int(max_ep["episode_id"]) if (max_ep and "episode_id" in max_ep) else 0
            self.counters.update_one({"_id": "episode_id"}, {"$set": {"seq": max_ep_val}}, upsert=True)

        if self.counters.find_one({"_id": "curve_id"}) is None:
            max_cv = self.training_curve.find_one(sort=[("curve_id", pymongo.DESCENDING)])
            max_cv_val = int(max_cv["curve_id"]) if (max_cv and "curve_id" in max_cv) else 0
            self.counters.update_one({"_id": "curve_id"}, {"$set": {"seq": max_cv_val}}, upsert=True)

    def _next_id(self, counter_name: str) -> int:
        """Atomically increments and returns the next integer ID for a sequence."""
        res = self.counters.find_one_and_update(
            {"_id": counter_name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=pymongo.ReturnDocument.AFTER
        )
        return int(res["seq"])

    def start_run(self, run_type: str, config: dict, policy: str = "marl", seed: int = 42, notes: str = "") -> str:
        """Registers a new run in MongoDB and returns a unique run_id."""
        run_id = f"{run_type}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        config_json = json.dumps(config) if isinstance(config, (dict, list)) else str(config)

        doc = {
            "run_id": run_id,
            "run_type": run_type,
            "policy": policy,
            "config_used": config_json,
            "seed": int(seed) if seed is not None else None,
            "timestamp": timestamp,
            "notes": notes or ""
        }
        self.runs.insert_one(doc)
        return run_id

    def log_episode(self, run_id: str, episode_num: int, metrics: dict):
        """Logs episode evaluation metrics to MongoDB."""
        role_rewards = metrics.get("role_rewards", {})
        ep_id = self._next_id("episode_id")

        doc = {
            "episode_id": ep_id,
            "run_id": run_id,
            "episode_num": int(episode_num),
            "total_reward": float(metrics.get("total_reward", 0.0)),
            "avg_response_time": float(metrics.get("avg_response_time", 0.0)),
            "resource_utilization_rate": float(metrics.get("resource_utilization_rate", 0.0)),
            "casualties_served": int(metrics.get("casualties_served", 0)),
            "casualties_lost": int(metrics.get("casualties_lost", 0)),
            "casualties_total": int(metrics.get("casualties_total", 0)),
            "task_completion_rate": float(metrics.get("task_completion_rate", 0.0)),
            "coordination_efficiency": float(metrics.get("coordination_efficiency", 0.0)),
            "medical_reward": float(role_rewards.get("medical", 0.0)),
            "rescue_reward": float(role_rewards.get("rescue", 0.0)),
            "logistics_reward": float(role_rewards.get("logistics", 0.0))
        }
        self.episodes.insert_one(doc)

    def log_training_update(self, run_id: str, update_num: int, mean_reward: float, actor_loss: float, critic_loss: float, entropy: float):
        """Logs MARL training update metrics to MongoDB."""
        cv_id = self._next_id("curve_id")
        doc = {
            "curve_id": cv_id,
            "run_id": run_id,
            "update_num": int(update_num),
            "mean_reward": float(mean_reward),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "entropy": float(entropy)
        }
        self.training_curve.insert_one(doc)

    def get_run_episodes(self, run_id: str) -> pd.DataFrame:
        """Fetches episodes for a specific run as a pandas DataFrame."""
        cursor = self.episodes.find({"run_id": run_id}, {"_id": 0}).sort("episode_num", pymongo.ASCENDING)
        records = list(cursor)
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame(columns=[
            "episode_id", "run_id", "episode_num", "total_reward",
            "avg_response_time", "resource_utilization_rate",
            "casualties_served", "casualties_lost", "casualties_total",
            "task_completion_rate", "coordination_efficiency",
            "medical_reward", "rescue_reward", "logistics_reward"
        ])

    def get_training_curve(self, run_id: str) -> pd.DataFrame:
        """Fetches training curve for a specific run as a pandas DataFrame."""
        cursor = self.training_curve.find({"run_id": run_id}, {"_id": 0}).sort("update_num", pymongo.ASCENDING)
        records = list(cursor)
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame(columns=[
            "curve_id", "run_id", "update_num", "mean_reward",
            "actor_loss", "critic_loss", "entropy"
        ])

    def get_latest_training_curve(self) -> pd.DataFrame:
        """Fetches latest training curve from MongoDB."""
        cursor = self.training_curve.find({}, {"_id": 0}).sort("curve_id", pymongo.ASCENDING)
        records = list(cursor)
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame(columns=[
            "curve_id", "run_id", "update_num", "mean_reward",
            "actor_loss", "critic_loss", "entropy"
        ])

    def get_all_runs(self) -> List[dict]:
        """Fetches all registered runs ordered by timestamp descending."""
        cursor = self.runs.find(
            {},
            {"_id": 0, "run_id": 1, "run_type": 1, "policy": 1, "timestamp": 1, "notes": 1}
        ).sort("timestamp", pymongo.DESCENDING)
        return [
            {
                "run_id": r.get("run_id", ""),
                "run_type": r.get("run_type", ""),
                "policy": r.get("policy", "marl"),
                "timestamp": r.get("timestamp", ""),
                "notes": r.get("notes", "")
            }
            for r in cursor
        ]

    def clear_database(self):
        """Removes all documents from collections and resets counters for clean testing."""
        self.runs.delete_many({})
        self.episodes.delete_many({})
        self.training_curve.delete_many({})
        self.counters.delete_many({})

    def drop_database(self):
        """Drops the entire database."""
        self.client.drop_database(self.db_name)

    def close(self):
        """Closes the MongoDB client connection."""
        self.client.close()
