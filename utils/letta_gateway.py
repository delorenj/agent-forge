from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    httpx = None


DEFAULT_LETTA_BASE_URL = "https://api.letta.com"


class LettaGatewayError(RuntimeError):
    """Raised when Letta API configuration or calls fail."""


def resolve_letta_agent_id(
    source_metadata: Optional[Dict[str, Any]] = None,
    override_agent_id: Optional[str] = None,
) -> str:
    """Resolve the Letta agent ID from override or registry metadata."""
    if override_agent_id:
        return override_agent_id

    metadata = source_metadata or {}
    agent_id = metadata.get("letta_agent_id") or metadata.get("letta_id")
    if not agent_id:
        raise LettaGatewayError(
            "Missing Letta agent ID. Provide --letta-agent-id (CLI) or "
            "letta_agent_id (API), or store it in registry source_metadata."
        )
    return agent_id


def resolve_base_url(
    override_base_url: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve Letta base URL from override, metadata, or env."""
    if override_base_url:
        return override_base_url.rstrip("/")

    metadata = source_metadata or {}
    base_url = metadata.get("letta_base_url") or os.getenv("LETTA_BASE_URL")
    if not base_url:
        base_url = DEFAULT_LETTA_BASE_URL
    return base_url.rstrip("/")


def resolve_token(override_token: Optional[str] = None) -> str:
    """Resolve Letta API token from override or env."""
    if override_token:
        return override_token

    token = os.getenv("LETTA_API_KEY") or os.getenv("LETTA_API_TOKEN")
    if not token:
        raise LettaGatewayError(
            "Missing Letta API token. Set LETTA_API_KEY/LETTA_API_TOKEN or "
            "pass --token (CLI) or token (API)."
        )
    return token


def _extract_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def extract_assistant_message(response: Dict[str, Any]) -> str:
    """Extract the last assistant message from a Letta response payload."""
    messages = response.get("messages") or []
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        role = message.get("message_type") or message.get("role")
        if role in {"assistant_message", "assistant"}:
            return _extract_text(message.get("content"))

    return ""


async def send_message(
    agent_id: str,
    message: str,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a message to a Letta agent and return the raw response."""
    if httpx is None:
        raise LettaGatewayError(
            "httpx is required to call the Letta API. Install with "
            "`uv pip install httpx` or ensure fastapi[standard] is installed."
        )
    resolved_base_url = resolve_base_url(base_url)
    resolved_token = resolve_token(token)

    url = f"{resolved_base_url}/v1/agents/{agent_id}/messages"
    payload = {"messages": [{"role": "user", "content": message}]}
    headers = {"Authorization": f"Bearer {resolved_token}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code >= 400:
        raise LettaGatewayError(
            f"Letta API error {response.status_code}: {response.text}"
        )

    return response.json()
