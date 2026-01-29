"""
AgentForge Team Registry

Teams are collections of agents configured for specific tasks/projects.
Teams can be packaged as standalone MCP servers.

Architecture:
    Team Definition → Agent References → MCP Server Generation → Deployment
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class OrchestrationType(str, Enum):
    """How agents in the team coordinate"""
    SEQUENTIAL = "sequential"      # Agents run in order
    PARALLEL = "parallel"          # Agents run concurrently
    HIERARCHICAL = "hierarchical"  # Coordinator delegates to specialists
    MESH = "mesh"                  # Peer-to-peer collaboration
    CUSTOM = "custom"              # Custom orchestration logic


class TeamMemberRole(BaseModel):
    """Role definition for a team member"""
    agent_id: str = Field(description="Reference to agent in registry")
    role_in_team: str = Field(description="Role within this team")
    is_coordinator: bool = Field(default=False, description="Is this the team coordinator")
    dependencies: List[str] = Field(default_factory=list, description="Agent IDs this depends on")
    tools_exposed: List[str] = Field(default_factory=list, description="Which tools to expose as MCP")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Team-specific config")


class TeamDefinition(BaseModel):
    """Complete team definition"""

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique team ID")
    name: str = Field(description="Team name")
    description: str = Field(description="Team purpose and capabilities")

    # Members
    members: List[TeamMemberRole] = Field(description="Team members and their roles")
    orchestration_type: OrchestrationType = Field(description="How team coordinates")

    # Configuration
    domain: str = Field(description="Primary domain")
    tags: List[str] = Field(default_factory=list)

    # MCP Server Config
    mcp_server_name: str = Field(description="MCP server identifier")
    mcp_tools_prefix: str = Field(description="Prefix for MCP tool names")
    expose_individual_agents: bool = Field(default=True, description="Expose each agent as tool")
    expose_team_coordinator: bool = Field(default=True, description="Expose unified team entry point")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="agentforge")
    version: str = Field(default="1.0.0")

    # Usage tracking
    deployment_count: int = Field(default=0)
    last_deployed: Optional[datetime] = None

    class Config:
        use_enum_values = True


class TeamRegistry:
    """
    Registry for agent teams.

    Responsibilities:
    - Define and store team configurations
    - Track team deployments and usage
    - Generate team-specific MCP servers
    - Manage team versioning
    """

    def __init__(self, registry_file: str = "teams/registry.json"):
        """Initialize team registry"""
        self.registry_file = Path(registry_file)
        self.teams: Dict[str, TeamDefinition] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.teams = {
                        team_id: TeamDefinition(**team_data)
                        for team_id, team_data in data.items()
                    }
                logger.info(f"Loaded {len(self.teams)} teams from registry")
            except Exception as e:
                logger.error(f"Failed to load team registry: {e}")
                self.teams = {}
        else:
            logger.info("No existing team registry found, starting fresh")
            self.teams = {}

    def _save_registry(self):
        """Save registry to disk"""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w') as f:
                data = {
                    team_id: team.model_dump()
                    for team_id, team in self.teams.items()
                }
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.teams)} teams to registry")
        except Exception as e:
            logger.error(f"Failed to save team registry: {e}")

    def create_team(self, team: TeamDefinition) -> str:
        """
        Create a new team.

        Args:
            team: Team definition

        Returns:
            Team ID
        """
        team.updated_at = datetime.now()
        self.teams[team.id] = team
        self._save_registry()
        logger.info(f"Created team: {team.name} ({team.id})")
        return team.id

    def get_team(self, team_id: str) -> Optional[TeamDefinition]:
        """Get team by ID"""
        return self.teams.get(team_id)

    def find_teams(
        self,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[TeamDefinition]:
        """Find teams matching criteria"""
        results = list(self.teams.values())

        if name:
            results = [t for t in results if name.lower() in t.name.lower()]
        if domain:
            results = [t for t in results if t.domain == domain]
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]

        return results

    def list_all_teams(self) -> List[TeamDefinition]:
        """List all teams"""
        return list(self.teams.values())

    def update_team(self, team_id: str, updates: Dict[str, Any]) -> bool:
        """Update team configuration"""
        if team := self.teams.get(team_id):
            for key, value in updates.items():
                if hasattr(team, key):
                    setattr(team, key, value)
            team.updated_at = datetime.now()
            self._save_registry()
            return True
        return False

    def delete_team(self, team_id: str) -> bool:
        """Delete team"""
        if team_id in self.teams:
            del self.teams[team_id]
            self._save_registry()
            logger.info(f"Deleted team: {team_id}")
            return True
        return False

    def record_deployment(self, team_id: str):
        """Record team deployment"""
        if team := self.teams.get(team_id):
            team.deployment_count += 1
            team.last_deployed = datetime.now()
            self._save_registry()

    def get_team_members_from_registry(self, team_id: str, agent_registry):
        """
        Get full agent details for team members.

        Args:
            team_id: Team ID
            agent_registry: AgentRegistry instance

        Returns:
            List of (TeamMemberRole, AgentRegistryEntry) tuples
        """
        team = self.get_team(team_id)
        if not team:
            return []

        members = []
        for member_role in team.members:
            agent = agent_registry.get_agent(member_role.agent_id)
            if agent:
                members.append((member_role, agent))
            else:
                logger.warning(f"Agent {member_role.agent_id} not found in registry")

        return members

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        teams = list(self.teams.values())
        return {
            "total_teams": len(teams),
            "by_orchestration": {
                orch_type: len([t for t in teams if t.orchestration_type == orch_type])
                for orch_type in OrchestrationType
            },
            "total_deployments": sum(t.deployment_count for t in teams),
            "most_deployed": sorted(teams, key=lambda t: t.deployment_count, reverse=True)[:5]
        }
