"""FastAPI dependency that extracts and validates the JWT bearer token.

Usage in any route::

    @app.get("/api/protected")
    def protected(user: dict = Depends(require_auth)):
        return {"hello": user["name"]}
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from satnet.auth.service import decode_access_token

_scheme = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_scheme),
) -> dict:
    """Validate the ``Authorization: Bearer <token>`` header.

    Returns the decoded JWT payload on success.
    Raises *401 Unauthorized* otherwise.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
