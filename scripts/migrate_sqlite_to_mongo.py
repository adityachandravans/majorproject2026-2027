"""
Migration script to transfer historical data from SQLite databases (e.g. disaster_metrics.db)
to MongoDB.
"""

import os
import sys
import sqlite3
import pymongo

def migrate_sqlite_to_mongodb(
    sqlite_path: str,
    mongo_uri: str = "mongodb://localhost:27017/",
    database_name: str = "disaster_metrics"
):
    if not os.path.exists(sqlite_path):
        print(f"SQLite file {sqlite_path} does not exist. Skipping.")
        return

    print(f"Migrating {sqlite_path} -> MongoDB database '{database_name}' on {mongo_uri}...")
    client = pymongo.MongoClient(mongo_uri)
    db = client[database_name]

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    cur = sqlite_conn.cursor()

    # 1. Migrate runs
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs';")
    if cur.fetchone():
        cur.execute("SELECT * FROM runs;")
        rows = cur.fetchall()
        print(f"Found {len(rows)} runs in SQLite.")
        runs_col = db["runs"]
        runs_col.create_index("run_id", unique=True)
        for row in rows:
            doc = dict(row)
            # Upsert into runs
            runs_col.update_one({"run_id": doc["run_id"]}, {"$set": doc}, upsert=True)

    # 2. Migrate episodes
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes';")
    if cur.fetchone():
        cur.execute("SELECT * FROM episodes;")
        rows = cur.fetchall()
        print(f"Found {len(rows)} episodes in SQLite.")
        ep_col = db["episodes"]
        ep_col.create_index([("run_id", pymongo.ASCENDING), ("episode_num", pymongo.ASCENDING)])
        ep_col.create_index("episode_id")
        
        max_ep_id = 0
        docs = []
        for row in rows:
            doc = dict(row)
            if "episode_id" in doc and doc["episode_id"] > max_ep_id:
                max_ep_id = doc["episode_id"]
            docs.append(doc)

        if docs:
            # Clear or update
            for doc in docs:
                ep_col.update_one(
                    {"run_id": doc["run_id"], "episode_num": doc["episode_num"]},
                    {"$set": doc},
                    upsert=True
                )
            # Sync counters
            db["counters"].update_one(
                {"_id": "episode_id"},
                {"$max": {"seq": max_ep_id}},
                upsert=True
            )

    # 3. Migrate training_curve
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_curve';")
    if cur.fetchone():
        cur.execute("SELECT * FROM training_curve;")
        rows = cur.fetchall()
        print(f"Found {len(rows)} training curve records in SQLite.")
        tc_col = db["training_curve"]
        tc_col.create_index([("run_id", pymongo.ASCENDING), ("update_num", pymongo.ASCENDING)])
        tc_col.create_index("curve_id")

        max_curve_id = 0
        docs = []
        for row in rows:
            doc = dict(row)
            if "curve_id" in doc and doc["curve_id"] > max_curve_id:
                max_curve_id = doc["curve_id"]
            docs.append(doc)

        if docs:
            for doc in docs:
                tc_col.update_one(
                    {"run_id": doc["run_id"], "update_num": doc["update_num"]},
                    {"$set": doc},
                    upsert=True
                )
            # Sync counters
            db["counters"].update_one(
                {"_id": "curve_id"},
                {"$max": {"seq": max_curve_id}},
                upsert=True
            )

    sqlite_conn.close()
    print(f"Successfully migrated {sqlite_path} to MongoDB database '{database_name}'!\n")

if __name__ == "__main__":
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    migrate_sqlite_to_mongodb("disaster_metrics.db", uri, "disaster_metrics")
    if os.path.exists("smoke_metrics.db"):
        migrate_sqlite_to_mongodb("smoke_metrics.db", uri, "smoke_metrics")
