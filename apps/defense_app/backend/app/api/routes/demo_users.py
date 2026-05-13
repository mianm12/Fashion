from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.database import get_database, not_found
from app.repositories.demo_user_repository import DemoUserRepository
from app.schemas.demo_users import (
    DemoUserItem,
    DemoUserListResponse,
    UserProfileResponse,
)

router = APIRouter(prefix="/demo-users", tags=["demo-users"])


@router.get("", response_model=DemoUserListResponse)
def list_demo_users(
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
    q: str | None = None,
    tag: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 50,
) -> dict[str, object]:
    repository = DemoUserRepository(connection)
    return {"items": repository.list_users(q, tag, limit)}


@router.get("/{case_id}", response_model=DemoUserItem)
def get_demo_user(
    case_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = DemoUserRepository(connection)
    user = repository.get(case_id)
    if user is None:
        raise not_found("未找到指定演示用户")
    return user


@router.get("/{case_id}/profile", response_model=UserProfileResponse)
def get_demo_user_profile(
    case_id: str,
    connection: Annotated[sqlite3.Connection, Depends(get_database)],
) -> dict[str, object]:
    repository = DemoUserRepository(connection)
    if repository.get(case_id) is None:
        raise not_found("未找到指定演示用户")
    return {"case_id": case_id, "items": repository.profile(case_id)}
