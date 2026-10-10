"""MongoDB persistence repository for SatNet simulations and results."""
from __future__ import annotations

from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.collection import Collection

from satnet.core.config import get_settings
from satnet.domain.models import SimulationSummary


class SimulationRepository:
    """MongoDB-backed persistence repository for simulation summaries."""

    def __init__(self, uri: str | None = None, db_name: str = "satnet") -> None:
        settings = get_settings()
        # If uri was not provided or was an old sqlite url, use mongodb_uri from settings
        target_uri = uri if (uri and not uri.startswith("sqlite")) else settings.mongodb_uri
        self.uri = target_uri
        self.db_name = db_name
        self._in_memory: dict[str, SimulationSummary] = {}
        self._mongo_available = False

        try:
            self.client: MongoClient = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.collection: Collection = self.db["simulations"]
            self.collection.create_index("simulation_id", unique=True)
            self.collection.create_index("created_at")
            self._mongo_available = True
        except Exception:
            # Fall back to in-memory dictionary if MongoDB is temporarily unreachable
            self._mongo_available = False

    def save_summary(self, summary: SimulationSummary, user_email: str | None = None) -> None:
        """Save a simulation summary to MongoDB."""
        self._in_memory[summary.simulation_id] = summary
        if self._mongo_available:
            try:
                doc = {
                    "_id": summary.simulation_id,
                    "simulation_id": summary.simulation_id,
                    "summary": summary.model_dump(mode="json"),
                    "created_at": datetime.now(timezone.utc),
                }
                if user_email:
                    doc["user_email"] = user_email
                self.collection.replace_one(
                    {"_id": summary.simulation_id},
                    doc,
                    upsert=True,
                )
            except Exception:
                pass

    def get_summary(self, simulation_id: str) -> SimulationSummary | None:
        """Fetch a simulation summary by ID from MongoDB."""
        if self._mongo_available:
            try:
                doc = self.collection.find_one({"_id": simulation_id})
                if doc and "summary" in doc:
                    return SimulationSummary.model_validate(doc["summary"])
            except Exception:
                pass
        return self._in_memory.get(simulation_id)

    def list_summaries(self, limit: int = 50, user_email: str | None = None) -> list[dict]:
        """List past simulation summaries from MongoDB."""
        if self._mongo_available:
            try:
                query = {}
                if user_email:
                    query["user_email"] = user_email
                cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
                return [doc["summary"] for doc in cursor if "summary" in doc]
            except Exception:
                pass
        return [s.model_dump(mode="json") for s in self._in_memory.values()][:limit]