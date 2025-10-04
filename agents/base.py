"""
Base agent class for AgentForge system using Agno framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
# Optional MCP integration
try:
    from agno.tools.mcp import MCPTools, StreamableHTTPClientParams
    MCP_AVAILABLE = True
except ImportError:
    MCPTools = None
    StreamableHTTPClientParams = None
    MCP_AVAILABLE = False
from os import getenv


class AgentForgeInput(BaseModel):
    """Base input model for AgentForge agents."""
    goal: str
    context: Optional[Dict[str, Any]] = None


class AgentForgeOutput(BaseModel):
    """Base output model for AgentForge agents."""
    result: Any
    status: str
    metadata: Optional[Dict[str, Any]] = None


class AgentForgeBase(ABC):
    """Base class for all AgentForge agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.agent = Agent(
            name=name,
            model=OpenRouter(id="deepseek/deepseek-v3.1"),
            db=SqliteDb(db_file=f"agentforge_{name.lower()}.db"),
            add_history_to_context=True,
            markdown=True,
        )
        
        # MCP setup for Agno documentation access (if available)
        self.mcp_params = None
        if MCP_AVAILABLE and StreamableHTTPClientParams:
            self.mcp_params = StreamableHTTPClientParams(
                url="https://mcp.delo.sh/metamcp/agentforge/mcp",
                headers={"Authorization": f"Bearer {getenv('MCP_API_KEY')}"},
                terminate_on_close=True,
            )
    
    @abstractmethod
    async def process(self, input_data: AgentForgeInput) -> AgentForgeOutput:
        """Process input and return output."""
        pass
    
    async def run_with_mcp(self, message: str, tools: Optional[list] = None) -> str:
        """Run agent with MCP tools for Agno documentation access (if available)."""
        if MCP_AVAILABLE and self.mcp_params and MCPTools:
            mcp_tools = MCPTools(self.mcp_params)
            response = await self.agent.aprint_response(
                message, 
                stream=False,
                tools=[mcp_tools] + (tools or [])
            )
            return response
        else:
            # Fallback to running without MCP tools
            response = await self.agent.aprint_response(
                message, 
                stream=False,
                tools=tools or []
            )
            return response