from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core.agent_registry import AgentRegistry, AgentSource
from utils.letta_gateway import (
    LettaGatewayError,
    extract_assistant_message,
    resolve_base_url,
    resolve_letta_agent_id,
    send_message,
)


REGISTRY_PATH = Path(__file__).resolve().parent / "agents" / "registry.json"
registry = AgentRegistry(str(REGISTRY_PATH))

app = FastAPI(title="agent-forge", version="0.1.0")


class AgentMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to send")
    letta_agent_id: Optional[str] = Field(
        None, description="Override Letta agent ID for this request"
    )
    base_url: Optional[str] = Field(None, description="Override Letta base URL")
    token: Optional[str] = Field(
        None, description="Override Letta API token (avoid in shared deployments)"
    )
    include_raw: bool = Field(
        False, description="Include raw Letta response payload"
    )


class AgentMessageResponse(BaseModel):
    agent_id: str
    letta_agent_id: str
    assistant: str
    raw: Optional[Dict[str, Any]] = None


class AgentMetadataUpdate(BaseModel):
    letta_agent_id: Optional[str] = None
    letta_base_url: Optional[str] = None
    tags: Optional[List[str]] = None
    domain: Optional[str] = None
    role: Optional[str] = None
    source_metadata: Optional[Dict[str, Any]] = None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "service": "agent-forge"}


@app.get("/agents")
def list_agents(
    source: Optional[str] = Query(None, description="Filter by source type"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
) -> List[Dict[str, Any]]:
    try:
        source_filter = AgentSource(source) if source else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid source: {source}") from exc
    agents = registry.find_agents(source=source_filter, domain=domain)
    return [agent.model_dump() for agent in agents]


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str) -> Dict[str, Any]:
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.model_dump()


@app.post("/agents/{agent_id}/messages", response_model=AgentMessageResponse)
async def send_agent_message(agent_id: str, payload: AgentMessageRequest) -> AgentMessageResponse:
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    source_value = agent.source.value if hasattr(agent.source, "value") else str(agent.source)
    if source_value != AgentSource.LETTA_IMPORT.value and not payload.letta_agent_id:
        raise HTTPException(
            status_code=400,
            detail="Agent is not a Letta import. Provide letta_agent_id to override.",
        )

    try:
        letta_agent_id = resolve_letta_agent_id(
            agent.source_metadata, payload.letta_agent_id
        )
        base_url = resolve_base_url(payload.base_url, agent.source_metadata)
        response = await send_message(
            agent_id=letta_agent_id,
            message=payload.message,
            base_url=base_url,
            token=payload.token,
        )
    except LettaGatewayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    assistant = extract_assistant_message(response)
    registry.update_usage(agent_id)

    return AgentMessageResponse(
        agent_id=agent_id,
        letta_agent_id=letta_agent_id,
        assistant=assistant,
        raw=response if payload.include_raw else None,
    )


@app.patch("/agents/{agent_id}/metadata")
def update_agent_metadata(agent_id: str, payload: AgentMetadataUpdate) -> Dict[str, Any]:
    agent = registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if payload.tags is not None:
        agent.tags = payload.tags
    if payload.domain is not None:
        agent.domain = payload.domain
    if payload.role is not None:
        agent.role = payload.role

    if agent.source_metadata is None:
        agent.source_metadata = {}

    if payload.source_metadata:
        agent.source_metadata.update(payload.source_metadata)
    if payload.letta_agent_id is not None:
        agent.source_metadata["letta_agent_id"] = payload.letta_agent_id
    if payload.letta_base_url is not None:
        agent.source_metadata["letta_base_url"] = payload.letta_base_url

    registry.register_agent(agent)
    return agent.model_dump()
