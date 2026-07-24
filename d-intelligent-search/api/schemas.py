"""API 스키마 — 디자인 doc 섹션 10."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=0, max_length=2000)
    options: dict | None = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    vector_index_ready: bool = False
    lora_adapter_loaded: bool = False
    lora_adapter_name: str | None = None
    lora_config_enabled: bool = False
    version: str
