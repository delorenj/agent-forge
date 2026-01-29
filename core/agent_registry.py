"""
AgentForge Agent Registry

Centralized registry for all agents in the system:
- Imported Letta agents (.af files)
- AgentForge-created agents (.md files)
- Runtime Agno agent instances
- Indexed in QDrant for semantic search via TalentScout

Architecture:
    Agent Pool → QDrant Index → TalentScout → Team Assembly → MCP Server
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentSource(str, Enum):
    """Source type for an agent"""
    LETTA_IMPORT = "letta_import"      # Imported from Letta .af file
    AGENTFORGE_CREATED = "agentforge"  # Created by AgentDeveloper
    CUSTOM_MARKDOWN = "custom_md"      # Custom .md file in library
    RUNTIME_AGNO = "runtime_agno"      # Runtime Agno agent instance


class AgentStatus(str, Enum):
    """Agent availability status"""
    ACTIVE = "active"           # Ready to use
    INDEXING = "indexing"       # Being indexed in QDrant
    DEPRECATED = "deprecated"   # Marked for removal
    ERROR = "error"             # Failed validation


class AgentRegistryEntry(BaseModel):
    """Complete registry entry for an agent"""

    # Identity
    id: str = Field(description="Unique agent identifier (UUID)")
    name: str = Field(description="Agent display name")
    source: AgentSource = Field(description="Where this agent came from")

    # Location
    file_path: Optional[str] = Field(None, description="Path to agent file (.af, .md, .py)")
    storage_format: str = Field(description="Storage format: af, md, py, json")

    # Metadata
    description: str = Field(description="Agent description (from persona/top-level)")
    role: str = Field(description="Primary role/function")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    domain: str = Field(description="Primary domain")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Configuration
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")
    memory_blocks: Dict[str, str] = Field(default_factory=dict, description="Memory block summaries")

    # Registry metadata
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = Field(None)
    usage_count: int = Field(default=0)

    # Integration
    qdrant_indexed: bool = Field(default=False, description="Indexed in QDrant")
    qdrant_point_id: Optional[str] = Field(None, description="QDrant point ID")
    team_memberships: List[str] = Field(default_factory=list, description="Team IDs using this agent")

    # Source-specific metadata
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific data")

    class Config:
        use_enum_values = True


class AgentRegistry:
    """
    Centralized registry for all agents in AgentForge.

    Responsibilities:
    - Register agents from various sources (import, creation, custom)
    - Maintain agent metadata and status
    - Coordinate with QDrant for semantic indexing
    - Provide query interface for agent discovery
    - Track agent usage and team memberships
    """

    def __init__(self, registry_file: str = "agents/registry.json"):
        """Initialize agent registry"""
        self.registry_file = Path(registry_file)
        self.agents: Dict[str, AgentRegistryEntry] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.agents = {
                        agent_id: AgentRegistryEntry(**agent_data)
                        for agent_id, agent_data in data.items()
                    }
                logger.info(f"Loaded {len(self.agents)} agents from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.agents = {}
        else:
            logger.info("No existing registry found, starting fresh")
            self.agents = {}

    def _save_registry(self):
        """Save registry to disk"""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w') as f:
                data = {
                    agent_id: agent.model_dump()
                    for agent_id, agent in self.agents.items()
                }
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.agents)} agents to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_agent(self, entry: AgentRegistryEntry) -> str:
        """
        Register a new agent or update existing.

        Args:
            entry: Agent registry entry

        Returns:
            Agent ID
        """
        entry.updated_at = datetime.now()
        self.agents[entry.id] = entry
        self._save_registry()
        logger.info(f"Registered agent: {entry.name} ({entry.id})")
        return entry.id

    def get_agent(self, agent_id: str) -> Optional[AgentRegistryEntry]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def find_agents(
        self,
        name: Optional[str] = None,
        source: Optional[AgentSource] = None,
        domain: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentRegistryEntry]:
        """
        Find agents matching criteria.

        Args:
            name: Filter by name (partial match)
            source: Filter by source type
            domain: Filter by domain
            status: Filter by status
            tags: Filter by tags (any match)

        Returns:
            List of matching agents
        """
        results = list(self.agents.values())

        if name:
            results = [a for a in results if name.lower() in a.name.lower()]
        if source:
            results = [a for a in results if a.source == source]
        if domain:
            results = [a for a in results if a.domain == domain]
        if status:
            results = [a for a in results if a.status == status]
        if tags:
            results = [a for a in results if any(tag in a.tags for tag in tags)]

        return results

    def list_all_agents(self) -> List[AgentRegistryEntry]:
        """List all registered agents"""
        return list(self.agents.values())

    def update_usage(self, agent_id: str):
        """Update agent usage statistics"""
        if agent := self.agents.get(agent_id):
            agent.usage_count += 1
            agent.last_used = datetime.now()
            self._save_registry()

    def mark_indexed(self, agent_id: str, qdrant_point_id: str):
        """Mark agent as indexed in QDrant"""
        if agent := self.agents.get(agent_id):
            agent.qdrant_indexed = True
            agent.qdrant_point_id = qdrant_point_id
            agent.updated_at = datetime.now()
            self._save_registry()

    def add_team_membership(self, agent_id: str, team_id: str):
        """Add team membership for agent"""
        if agent := self.agents.get(agent_id):
            if team_id not in agent.team_memberships:
                agent.team_memberships.append(team_id)
                agent.updated_at = datetime.now()
                self._save_registry()

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent from registry"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._save_registry()
            logger.info(f"Deleted agent: {agent_id}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        agents = list(self.agents.values())
        return {
            "total_agents": len(agents),
            "by_source": {
                source: len([a for a in agents if a.source == source])
                for source in AgentSource
            },
            "by_status": {
                status: len([a for a in agents if a.status == status])
                for status in AgentStatus
            },
            "indexed_count": len([a for a in agents if a.qdrant_indexed]),
            "most_used": sorted(agents, key=lambda a: a.usage_count, reverse=True)[:5]
        }
