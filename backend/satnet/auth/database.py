"""MongoDB connection manager for authentication.

Uses pymongo to connect to the MongoDB instance specified by MONGODB_URI.
The connection is lazily initialised and cached so every import shares
the same client.
"""
from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from satnet.core.config import get_settings


@lru_cache
def _get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)


def get_db() -> Database:
    """Return the ``satnet`` database handle."""
    return _get_client()["satnet"]


def get_users_collection():
    """Return the ``users`` collection, creating a unique index on *email*."""
    db = get_db()
    col = db["users"]
    col.create_index("email", unique=True)
    return col
