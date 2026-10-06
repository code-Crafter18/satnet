"""User management — registration, login, and JWT token handling.

Passwords are hashed with bcrypt via *passlib*.  Tokens are signed
with HS256 via *python-jose*.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from pymongo.errors import DuplicateKeyError, PyMongoError

from satnet.auth.database import get_users_collection
from satnet.core.config import get_settings

# ── Pydantic schemas ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

# ── Password helpers ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────

ACCESS_TOKEN_EXPIRE_HOURS = 24


def create_access_token(data: dict) -> str:
    settings = get_settings()
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict | None:
    """Return the payload dict if the token is valid, else *None*."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError:
        return None


import re
from satnet.auth.database import _get_client, get_users_collection

# ── Core operations ───────────────────────────────────────────────

def register_user(req: RegisterRequest) -> TokenResponse:
    """Create a new user and return a JWT token.

    Raises ``ValueError`` if the email is already taken.
    Raises ``ConnectionError`` if MongoDB is unreachable.
    """
    clean_email = str(req.email).strip().lower()
    clean_name = req.name.strip()

    try:
        users = get_users_collection()
        doc = {
            "name": clean_name,
            "email": clean_email,
            "password_hash": hash_password(req.password),
            "created_at": datetime.now(timezone.utc),
        }
        result = users.insert_one(doc)
    except DuplicateKeyError:
        raise ValueError("An account with this email already exists.")
    except PyMongoError as exc:
        raise ConnectionError(
            f"Database connection failed: {exc}. Please verify MONGODB_URI in your .env file."
        ) from exc

    user_data = {"id": str(result.inserted_id), "name": clean_name, "email": clean_email}
    token = create_access_token({"sub": clean_email, "name": clean_name})
    return TokenResponse(access_token=token, user=user_data)


def login_user(req: LoginRequest) -> TokenResponse:
    """Authenticate an existing user and return a JWT token.

    Raises ``ValueError`` on bad credentials.
    Raises ``ConnectionError`` if MongoDB is unreachable.
    """
    clean_email = str(req.email).strip().lower()

    try:
        users = get_users_collection()
        # 1. Exact match (fast index lookup)
        doc = users.find_one({"email": clean_email})

        # 2. Case-insensitive regex fallback in satnet.users
        if doc is None:
            doc = users.find_one({"email": {"$regex": f"^{re.escape(clean_email)}$", "$options": "i"}})

        # 3. Check legacy satnet_auth database if user was created before database rename
        if doc is None:
            try:
                legacy_col = _get_client()["satnet_auth"]["users"]
                doc = legacy_col.find_one({"email": {"$regex": f"^{re.escape(clean_email)}$", "$options": "i"}})
                if doc:
                    # Automatically migrate user into satnet.users
                    users.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            except Exception:
                pass

    except PyMongoError as exc:
        raise ConnectionError(
            f"Database connection failed: {exc}. Please verify MONGODB_URI in your .env file."
        ) from exc

    if doc is None or not verify_password(req.password, doc.get("password_hash", "")):
        raise ValueError("Invalid email or password.")

    # Record login timestamp in MongoDB
    try:
        users.update_one(
            {"_id": doc["_id"]},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )
    except Exception:
        pass

    user_data = {"id": str(doc["_id"]), "name": doc.get("name", "User"), "email": doc.get("email", clean_email)}
    token = create_access_token({"sub": user_data["email"], "name": user_data["name"]})
    return TokenResponse(access_token=token, user=user_data)
