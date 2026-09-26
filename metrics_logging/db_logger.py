"""
SQLite Database Logger for Disaster Response Coordination System (Section 8, FR7).
Provides persistent schema and methods to store and query run configurations and episode metrics.
"""
import sqlite3
import json
import uuid
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List

class DBLogger:
    def __init__(self, db_path: str = "disaster_metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates runs and episodes tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table: runs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_type TEXT NOT NULL,
                config_used TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                notes TEXT
            );
            """)

            # Table: episodes
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode_num INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                avg_response_time REAL NOT NULL,
                resource_utilization_rate REAL NOT NULL,
                casualties_served INTEGER NOT NULL,
                casualties_total INTEGER NOT NULL,
                task_completion_rate REAL NOT NULL,
                coordination_efficiency REAL NOT NULL,
                medical_reward REAL NOT NULL,
                rescue_reward REAL NOT NULL,
                logistics_reward REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            """)
            conn.commit()

    def start_run(self, run_type: str, config: dict, notes: str = "") -> str:
        """Registers a new run in the database and returns a unique run_id."""
        run_id = f"{run_type}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        config_json = json.dumps(config)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO runs (run_id, run_type, config_used, timestamp, notes) VALUES (?, ?, ?, ?, ?)",
                (run_id, run_type, config_json, timestamp, notes)
            )
            conn.commit()
        return run_id

    def log_episode(self, run_id: str, episode_num: int, metrics: dict):
        """Logs episode metrics to SQLite DB."""
        role_rewards = metrics.get("role_rewards", {})
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodes (
                    run_id, episode_num, total_reward, avg_response_time,
                    resource_utilization_rate, casualties_served, casualties_total,
                    task_completion_rate, coordination_efficiency,
                    medical_reward, rescue_reward, logistics_reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                episode_num,
                float(metrics.get("total_reward", 0.0)),
                float(metrics.get("avg_response_time", 0.0)),
                float(metrics.get("resource_utilization_rate", 0.0)),
                int(metrics.get("casualties_served", 0)),
                int(metrics.get("casualties_total", 0)),
                float(metrics.get("task_completion_rate", 0.0)),
                float(metrics.get("coordination_efficiency", 0.0)),
                float(role_rewards.get("medical", 0.0)),
                float(role_rewards.get("rescue", 0.0)),
                float(role_rewards.get("logistics", 0.0))
            ))
            conn.commit()

    def get_run_episodes(self, run_id: str) -> pd.DataFrame:
        """Fetches episode records for a given run_id as a pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM episodes WHERE run_id = ? ORDER BY episode_num ASC",
                conn,
                params=(run_id,)
            )
        return df

    def get_all_runs(self) -> List[dict]:
        """Fetches all registered run summaries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id, run_type, timestamp, notes FROM runs ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [
                {"run_id": r[0], "run_type": r[1], "timestamp": r[2], "notes": r[3]}
                for r in rows
            ]
