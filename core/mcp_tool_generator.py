"""
Dynamic MCP Tool Generator

Generates MCP Tool definitions and handlers from AgentRegistryEntry specifications.
Enables any agent to be exposed as an MCP tool with minimal configuration.
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from mcp.types import Tool
from core.agent_registry import AgentRegistryEntry, AgentSource
import json

logger = logging.getLogger(__name__)


class MCPToolGenerator:
    """
    Generates MCP tools dynamically from agent specifications.

    Supports:
    - Individual agent tools
    - Team coordinator tools
    - Custom input schemas
    - Async agent invocation
    """

    @staticmethod
    def generate_tool_name(agent: AgentRegistryEntry, prefix: str = "") -> str:
        """
        Generate MCP tool name from agent.

        Args:
            agent: Agent registry entry
            prefix: Optional prefix (e.g., team name)

        Returns:
            Tool name (e.g., "team_name__agent_name")
        """
        safe_name = agent.name.lower().replace(" ", "_").replace("-", "_")
        if prefix:
            safe_prefix = prefix.lower().replace(" ", "_").replace("-", "_")
            return f"{safe_prefix}__{safe_name}"
        return safe_name

    @staticmethod
    def generate_tool_description(agent: AgentRegistryEntry) -> str:
        """
        Generate tool description from agent metadata.

        Args:
            agent: Agent registry entry

        Returns:
            Formatted description
        """
        description = f"""🤖 {agent.name} - {agent.role}

{agent.description}

**Capabilities:** {', '.join(agent.capabilities[:5])}
**Domain:** {agent.domain}
**Source:** {agent.source.value}

"""
        if agent.tools:
            description += f"**Tools Available:** {', '.join(agent.tools[:3])}\n"

        return description.strip()

    @staticmethod
    def generate_input_schema(agent: AgentRegistryEntry) -> Dict[str, Any]:
        """
        Generate MCP tool input schema from agent capabilities.

        Args:
            agent: Agent registry entry

        Returns:
            JSON Schema for tool input
        """
        # Base schema with common fields
        schema = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"Task or query for {agent.name}"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the task"
                }
            },
            "required": ["task"]
        }

        # Add capability-specific parameters
        if agent.capabilities:
            schema["properties"]["use_capabilities"] = {
                "type": "array",
                "items": {"type": "string", "enum": agent.capabilities},
                "description": "Specific capabilities to use for this task"
            }

        # Add tool selection if agent has tools
        if agent.tools:
            schema["properties"]["tools_allowed"] = {
                "type": "array",
                "items": {"type": "string", "enum": agent.tools},
                "description": "Which tools the agent can use"
            }

        # Add source-specific parameters
        if agent.source == AgentSource.LETTA_IMPORT:
            schema["properties"]["memory_context"] = {
                "type": "string",
                "description": "Context to add to agent's memory"
            }

        return schema

    @staticmethod
    def generate_tool(
        agent: AgentRegistryEntry,
        prefix: str = ""
    ) -> Tool:
        """
        Generate complete MCP Tool from agent entry.

        Args:
            agent: Agent registry entry
            prefix: Optional tool name prefix

        Returns:
            MCP Tool object
        """
        tool_name = MCPToolGenerator.generate_tool_name(agent, prefix)
        description = MCPToolGenerator.generate_tool_description(agent)
        input_schema = MCPToolGenerator.generate_input_schema(agent)

        return Tool(
            name=tool_name,
            description=description,
            inputSchema=input_schema
        )

    @staticmethod
    async def invoke_agent(
        agent: AgentRegistryEntry,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Invoke an agent with given arguments.

        This is a dispatcher that handles different agent sources.

        Args:
            agent: Agent to invoke
            arguments: Tool call arguments

        Returns:
            Agent response as JSON string
        """
        try:
            if agent.source == AgentSource.LETTA_IMPORT:
                return await MCPToolGenerator._invoke_letta_agent(agent, arguments)
            elif agent.source in [AgentSource.AGENTFORGE_CREATED, AgentSource.CUSTOM_MARKDOWN]:
                return await MCPToolGenerator._invoke_agno_agent(agent, arguments)
            elif agent.source == AgentSource.RUNTIME_AGNO:
                return await MCPToolGenerator._invoke_runtime_agent(agent, arguments)
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Unsupported agent source: {agent.source}"
                })
        except Exception as e:
            logger.error(f"Error invoking agent {agent.name}: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "error": str(e),
                "agent": agent.name
            })

    @staticmethod
    async def _invoke_letta_agent(
        agent: AgentRegistryEntry,
        arguments: Dict[str, Any]
    ) -> str:
        """Invoke a Letta-imported agent"""
        # Load agent config from file
        from agentfile import load_agent

        try:
            agent_config = load_agent(agent.file_path)

            # Format response with agent details
            response = {
                "status": "success",
                "agent": agent.name,
                "task": arguments.get("task"),
                "response": f"[Letta agent {agent.name} would process: {arguments.get('task')}]",
                "config": {
                    "model": agent_config.get("model"),
                    "description": agent_config.get("description"),
                    "memory_blocks": list(agent_config.get("memory", {}).keys())
                },
                "note": "Full Letta agent execution requires Letta runtime integration"
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to load Letta agent: {e}"
            })

    @staticmethod
    async def _invoke_agno_agent(
        agent: AgentRegistryEntry,
        arguments: Dict[str, Any]
    ) -> str:
        """Invoke an Agno-based agent"""
        # This would load and execute the Agno agent
        # For now, return a structured response

        response = {
            "status": "success",
            "agent": agent.name,
            "task": arguments.get("task"),
            "response": f"[Agno agent {agent.name} would process: {arguments.get('task')}]",
            "capabilities_used": arguments.get("use_capabilities", agent.capabilities[:3]),
            "note": "Full Agno agent execution requires agent instantiation"
        }

        return json.dumps(response, indent=2)

    @staticmethod
    async def _invoke_runtime_agent(
        agent: AgentRegistryEntry,
        arguments: Dict[str, Any]
    ) -> str:
        """Invoke a runtime Agno agent instance"""
        # This would use a cached agent instance
        response = {
            "status": "success",
            "agent": agent.name,
            "task": arguments.get("task"),
            "response": f"[Runtime agent {agent.name} would process: {arguments.get('task')}]",
            "note": "Runtime agent invocation requires agent instance management"
        }

        return json.dumps(response, indent=2)

    @staticmethod
    def generate_team_coordinator_tool(
        team_name: str,
        team_description: str,
        team_members: List[AgentRegistryEntry]
    ) -> Tool:
        """
        Generate a team coordinator tool that orchestrates multiple agents.

        Args:
            team_name: Team name
            team_description: Team description
            team_members: List of team member agents

        Returns:
            MCP Tool for team coordination
        """
        tool_name = f"{team_name.lower().replace(' ', '_')}_team"

        description = f"""🚀 {team_name} Team Coordinator

{team_description}

**Team Members:**
"""
        for agent in team_members[:5]:
            description += f"  • {agent.name} ({agent.role})\n"

        if len(team_members) > 5:
            description += f"  • ...and {len(team_members) - 5} more agents\n"

        description += f"""
**How it works:**
1. Analyzes your task
2. Delegates to appropriate team members
3. Coordinates responses
4. Delivers unified result

Perfect for complex tasks requiring multiple specialized agents."""

        input_schema = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complex task for the team to accomplish"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context"
                },
                "preferred_agents": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [agent.name for agent in team_members]
                    },
                    "description": "Prefer specific team members if known"
                },
                "orchestration_mode": {
                    "type": "string",
                    "enum": ["sequential", "parallel", "auto"],
                    "default": "auto",
                    "description": "How to coordinate agents"
                }
            },
            "required": ["task"]
        }

        return Tool(
            name=tool_name,
            description=description,
            inputSchema=input_schema
        )
