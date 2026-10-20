"""Single shared API key via the X-API-Key header -- not a user/account system.
Beyond the spec's stated minimum (auth is explicitly excluded there); added on request.
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    if not settings.API_KEY:
        return  # no key configured -- auth disabled, dev convenience
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "Missing or invalid X-API-Key header."},
        )
