"""Unit and integration tests for SatNet authentication module.

Tests password hashing, JWT creation & decoding, user registration & login flows,
and FastAPI endpoint protection via the require_auth dependency.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from satnet.api.main import app
from satnet.auth.dependencies import require_auth
from satnet.auth.service import (
    LoginRequest,
    RegisterRequest,
    create_access_token,
    decode_access_token,
    hash_password,
    login_user,
    register_user,
    verify_password,
)


class MockUsersCollection:
    """Lightweight in-memory MongoDB users collection mock for tests."""

    def __init__(self):
        self.docs = []
        self._next_id = 1

    def create_index(self, *args, **kwargs):
        pass

    def insert_one(self, doc):
        email = doc.get("email")
        if any(d.get("email") == email for d in self.docs):
            raise DuplicateKeyError(f"E11000 duplicate key error collection: satnet_auth.users index: email_1 dup key: {{ email: '{email}' }}")
        
        stored = dict(doc)
        stored["_id"] = str(self._next_id)
        self._next_id += 1
        self.docs.append(stored)
        res = MagicMock()
        res.inserted_id = stored["_id"]
        return res

    def find_one(self, query):
        for doc in self.docs:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return dict(doc)
        return None

    def update_one(self, filter, update):
        doc = self.find_one(filter)
        if doc and "$set" in update:
            for d in self.docs:
                if d.get("_id") == doc.get("_id"):
                    d.update(update["$set"])
                    break
        return MagicMock()


# ── Password Hashing Tests ────────────────────────────────────────

def test_password_hash_and_verification():
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


def test_password_hash_is_unique_each_time():
    password = "MySecurePassword"
    h1 = hash_password(password)
    h2 = hash_password(password)
    assert h1 != h2
    assert verify_password(password, h1) is True
    assert verify_password(password, h2) is True


# ── JWT Token Tests ───────────────────────────────────────────────

def test_jwt_token_lifecycle():
    data = {"sub": "commander@satnet.space", "name": "Commander Shepard"}
    token = create_access_token(data)

    assert isinstance(token, str) and len(token) > 20

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "commander@satnet.space"
    assert payload["name"] == "Commander Shepard"
    assert "exp" in payload


def test_jwt_token_tampering_rejected():
    token = create_access_token({"sub": "astronaut@satnet.space", "name": "Astronaut"})
    tampered = token[:-4] + "xxxx"
    assert decode_access_token(tampered) is None
    assert decode_access_token("completely-invalid-token") is None


# ── User Registration & Login Service Tests ───────────────────────

def test_register_and_login_service():
    mock_col = MockUsersCollection()
    with patch("satnet.auth.service.get_users_collection", return_value=mock_col):
        # 1. Register new user
        reg_req = RegisterRequest(
            name="Flight Director",
            email="flight.director@satnet.space",
            password="MissionControl2026!",
        )
        res = register_user(reg_req)

        assert res.access_token is not None
        assert res.user["email"] == "flight.director@satnet.space"
        assert res.user["name"] == "Flight Director"
        assert res.token_type == "bearer"

        # 2. Duplicate registration fails
        with pytest.raises(ValueError, match="already exists"):
            register_user(reg_req)

        # 3. Successful login
        login_req = LoginRequest(
            email="flight.director@satnet.space",
            password="MissionControl2026!",
        )
        login_res = login_user(login_req)
        assert login_res.access_token is not None
        assert login_res.user["email"] == "flight.director@satnet.space"

        # 4. Login with incorrect password
        with pytest.raises(ValueError, match="Invalid email or password"):
            login_user(LoginRequest(
                email="flight.director@satnet.space",
                password="WrongPassword999!",
            ))

        # 5. Login with non-existent email
        with pytest.raises(ValueError, match="Invalid email or password"):
            login_user(LoginRequest(
                email="nobody@satnet.space",
                password="SomePassword123!",
            ))


# ── FastAPI Endpoint Protection & Auth Routes Tests ───────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_unauthenticated_request_rejected(client):
    # Protected endpoint without Authorization header must return 401
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401
    assert "detail" in resp.json()

    # Protected simulation endpoint without token must return 401
    sim_resp = client.post("/api/simulations", json={
        "tle_text": "dummy",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T01:00:00Z",
    })
    assert sim_resp.status_code == 401


def test_authenticated_user_profile_endpoint(client):
    token = create_access_token({"sub": "orbit@satnet.org", "name": "Orbit Specialist"})
    resp = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "orbit@satnet.org"
    assert data["name"] == "Orbit Specialist"


def test_api_register_and_login_flow(client):
    mock_col = MockUsersCollection()
    with patch("satnet.auth.service.get_users_collection", return_value=mock_col):
        # Register via API
        reg_resp = client.post("/api/auth/register", json={
            "name": "Alex Vance",
            "email": "alex@blackmesa.org",
            "password": "Resistance2026!",
        })
        assert reg_resp.status_code == 200
        data = reg_resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "alex@blackmesa.org"

        # Duplicate register via API gives 409
        dup_resp = client.post("/api/auth/register", json={
            "name": "Alex Vance Duplicate",
            "email": "alex@blackmesa.org",
            "password": "DifferentPassword123!",
        })
        assert dup_resp.status_code == 409

        # Login via API
        login_resp = client.post("/api/auth/login", json={
            "email": "alex@blackmesa.org",
            "password": "Resistance2026!",
        })
        assert login_resp.status_code == 200
        login_data = login_resp.json()
        assert "access_token" in login_data

        # Use token to call protected route
        token = login_data["access_token"]
        me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == "alex@blackmesa.org"
        assert me_resp.json()["name"] == "Alex Vance"
