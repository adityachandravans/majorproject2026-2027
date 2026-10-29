"""
MongoDB Database Logger for Disaster Response Coordination System.
Provides collection management, run tracking, episode metrics, and training curve logging via PyMongo.
"""

import os
import json
import uuid
import sqlite3
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
        self._backend = "mongo"
        self._sqlite_path = None
        self._sqlite_conn = None

        try:
            self.client = client or pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            self.runs = self.db["runs"]
            self.episodes = self.db["episodes"]
            self.training_curve = self.db["training_curve"]
            self.counters = self.db["counters"]
            self._init_db()
        except Exception:
            self._backend = "sqlite"
            sqlite_name = self.db_name if self.db_name.endswith(".db") else f"{self.db_name}.db"
            self._sqlite_path = os.path.join(os.getcwd(), sqlite_name)
            self._sqlite_conn = sqlite3.connect(self._sqlite_path)
            self._sqlite_conn.row_factory = sqlite3.Row
            self.runs = None
            self.episodes = None
            self.training_curve = None
            self.counters = None
            self._init_sqlite_db()

    def _init_sqlite_db(self):
        """Creates SQLite tables used by the same public API when MongoDB is unavailable."""
        conn = self._sqlite_conn
        if conn is None:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_type TEXT,
                policy TEXT,
                config_used TEXT,
                seed INTEGER,
                timestamp TEXT,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY,
                run_id TEXT,
                episode_num INTEGER,
                total_reward REAL,
                avg_response_time REAL,
                resource_utilization_rate REAL,
                casualties_served INTEGER,
                casualties_lost INTEGER,
                casualties_total INTEGER,
                task_completion_rate REAL,
                coordination_efficiency REAL,
                medical_reward REAL,
                rescue_reward REAL,
                logistics_reward REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_curve (
                curve_id INTEGER PRIMARY KEY,
                run_id TEXT,
                update_num INTEGER,
                mean_reward REAL,
                actor_loss REAL,
                critic_loss REAL,
                entropy REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                seq INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

    def _next_sqlite_id(self, counter_name: str) -> int:
        conn = self._sqlite_conn
        if conn is None:
            return 0

        current = conn.execute(
            "SELECT seq FROM counters WHERE name = ?",
            (counter_name,)
        ).fetchone()
        seq = int(current[0]) if current else 0
        seq += 1
        conn.execute(
            "INSERT INTO counters(name, seq) VALUES(?, ?) ON CONFLICT(name) DO UPDATE SET seq = excluded.seq",
            (counter_name, seq)
        )
        conn.commit()
        return seq

    def _sqlite_execute(self, query: str, params: tuple = ()):
        if self._sqlite_conn is None:
            return None
        return self._sqlite_conn.execute(query, params)

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
        """Registers a new run and returns a unique run_id."""
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

        if self._backend == "sqlite":
            conn = self._sqlite_conn
            if conn is not None:
                conn.execute(
                    "INSERT INTO runs(run_id, run_type, policy, config_used, seed, timestamp, notes) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (run_id, run_type, policy, config_json, int(seed) if seed is not None else None, timestamp, notes or "")
                )
                conn.commit()
            return run_id

        self.runs.insert_one(doc)
        return run_id

    def log_episode(self, run_id: str, episode_num: int, metrics: dict):
        """Logs episode evaluation metrics."""
        role_rewards = metrics.get("role_rewards", {})
        ep_id = self._next_sqlite_id("episode_id") if self._backend == "sqlite" else self._next_id("episode_id")

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

        if self._backend == "sqlite":
            conn = self._sqlite_conn
            if conn is not None:
                conn.execute(
                    "INSERT INTO episodes(episode_id, run_id, episode_num, total_reward, avg_response_time, resource_utilization_rate, casualties_served, casualties_lost, casualties_total, task_completion_rate, coordination_efficiency, medical_reward, rescue_reward, logistics_reward) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ep_id, run_id, int(episode_num), float(metrics.get("total_reward", 0.0)),
                        float(metrics.get("avg_response_time", 0.0)), float(metrics.get("resource_utilization_rate", 0.0)),
                        int(metrics.get("casualties_served", 0)), int(metrics.get("casualties_lost", 0)),
                        int(metrics.get("casualties_total", 0)), float(metrics.get("task_completion_rate", 0.0)),
                        float(metrics.get("coordination_efficiency", 0.0)), float(role_rewards.get("medical", 0.0)),
                        float(role_rewards.get("rescue", 0.0)), float(role_rewards.get("logistics", 0.0))
                    )
                )
                conn.commit()
            return

        self.episodes.insert_one(doc)

    def log_training_update(self, run_id: str, update_num: int, mean_reward: float, actor_loss: float, critic_loss: float, entropy: float):
        """Logs MARL training update metrics."""
        cv_id = self._next_sqlite_id("curve_id") if self._backend == "sqlite" else self._next_id("curve_id")
        doc = {
            "curve_id": cv_id,
            "run_id": run_id,
            "update_num": int(update_num),
            "mean_reward": float(mean_reward),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "entropy": float(entropy)
        }

        if self._backend == "sqlite":
            conn = self._sqlite_conn
            if conn is not None:
                conn.execute(
                    "INSERT INTO training_curve(curve_id, run_id, update_num, mean_reward, actor_loss, critic_loss, entropy) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (cv_id, run_id, int(update_num), float(mean_reward), float(actor_loss), float(critic_loss), float(entropy))
                )
                conn.commit()
            return

        self.training_curve.insert_one(doc)

    def get_run_episodes(self, run_id: str) -> pd.DataFrame:
        """Fetches episodes for a specific run as a pandas DataFrame."""
        if self._backend == "sqlite":
            rows = self._sqlite_conn.execute(
                "SELECT * FROM episodes WHERE run_id = ? ORDER BY episode_num ASC",
                (run_id,)
            ).fetchall()
            records = [dict(r) for r in rows]
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame(columns=[
                "episode_id", "run_id", "episode_num", "total_reward",
                "avg_response_time", "resource_utilization_rate",
                "casualties_served", "casualties_lost", "casualties_total",
                "task_completion_rate", "coordination_efficiency",
                "medical_reward", "rescue_reward", "logistics_reward"
            ])

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
        if self._backend == "sqlite":
            rows = self._sqlite_conn.execute(
                "SELECT * FROM training_curve WHERE run_id = ? ORDER BY update_num ASC",
                (run_id,)
            ).fetchall()
            records = [dict(r) for r in rows]
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame(columns=[
                "curve_id", "run_id", "update_num", "mean_reward",
                "actor_loss", "critic_loss", "entropy"
            ])

        cursor = self.training_curve.find({"run_id": run_id}, {"_id": 0}).sort("update_num", pymongo.ASCENDING)
        records = list(cursor)
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame(columns=[
            "curve_id", "run_id", "update_num", "mean_reward",
            "actor_loss", "critic_loss", "entropy"
        ])

    def get_latest_training_curve(self) -> pd.DataFrame:
        """Fetches latest training curve from the active backend."""
        if self._backend == "sqlite":
            rows = self._sqlite_conn.execute(
                "SELECT * FROM training_curve ORDER BY curve_id ASC"
            ).fetchall()
            records = [dict(r) for r in rows]
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame(columns=[
                "curve_id", "run_id", "update_num", "mean_reward",
                "actor_loss", "critic_loss", "entropy"
            ])

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
        if self._backend == "sqlite":
            rows = self._sqlite_conn.execute(
                "SELECT run_id, run_type, policy, timestamp, notes FROM runs ORDER BY timestamp DESC"
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "run_type": r["run_type"],
                    "policy": r["policy"],
                    "timestamp": r["timestamp"],
                    "notes": r["notes"]
                }
                for r in rows
            ]

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
        if self._backend == "sqlite":
            conn = self._sqlite_conn
            if conn is not None:
                for table in ("runs", "episodes", "training_curve", "counters"):
                    conn.execute(f"DELETE FROM {table}")
                conn.commit()
            return

        self.runs.delete_many({})
        self.episodes.delete_many({})
        self.training_curve.delete_many({})
        self.counters.delete_many({})

    def drop_database(self):
        """Drops the entire database."""
        if self._backend == "sqlite":
            if self._sqlite_path and os.path.exists(self._sqlite_path):
                self._sqlite_conn.close()
                os.remove(self._sqlite_path)
            return

        self.client.drop_database(self.db_name)

    def close(self):
        """Closes the active client connection."""
        if self._backend == "sqlite":
            if self._sqlite_conn is not None:
                self._sqlite_conn.close()
            return

        self.client.close()
