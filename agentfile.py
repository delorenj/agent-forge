"""
AgentFile - Letta Agent File Specification Implementation

This module implements serialization/deserialization for AgentForge agents
following the Letta agentfile spec (https://github.com/letta-ai/agent-file).

The .af format enables portable agent sharing with persistent memory and behavior.
"""

import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime


class ToolSchema(BaseModel):
    """Tool definition with code and schema."""
    name: str
    description: str
    parameters: Dict[str, Any]
    code: Optional[str] = None


class MemoryBlock(BaseModel):
    """Agent memory block (personality, user context, etc.)."""
    label: str
    value: str
    limit: Optional[int] = None


class Message(BaseModel):
    """Message in agent history."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM configuration."""
    model: str
    context_window: int = 8000
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class AgentFile(BaseModel):
    """
    Letta AgentFile specification.

    Serializes complete agent state including:
    - System prompts
    - Editable memory blocks
    - Tool configurations
    - LLM settings
    - Message history
    """

    # Metadata
    version: str = "1.0"
    name: str
    description: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Agent configuration
    llm_config: LLMConfig
    system_prompt: str

    # Memory
    memory_blocks: List[MemoryBlock] = Field(default_factory=list)

    # Tools
    tools: List[ToolSchema] = Field(default_factory=list)

    # Message history
    messages: List[Message] = Field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentFileSerializer:
    """Serialize/deserialize AgentForge agents to/from .af files."""

    @staticmethod
    def serialize(agent_data: Dict[str, Any], output_path: str) -> Path:
        """
        Serialize agent to .af file.

        Args:
            agent_data: Agent configuration and state
            output_path: Path for output .af file

        Returns:
            Path to created file
        """
        # Convert agent_data to AgentFile format
        agent_file = AgentFile(
            name=agent_data.get("name", "Unnamed Agent"),
            description=agent_data.get("description", ""),
            llm_config=LLMConfig(
                model=agent_data.get("model", "gpt-4"),
                context_window=agent_data.get("context_window", 8000),
                temperature=agent_data.get("temperature", 0.7)
            ),
            system_prompt=agent_data.get("system_prompt", ""),
            memory_blocks=[
                MemoryBlock(**block) if isinstance(block, dict) else block
                for block in agent_data.get("memory_blocks", [])
            ],
            tools=[
                ToolSchema(**tool) if isinstance(tool, dict) else tool
                for tool in agent_data.get("tools", [])
            ],
            messages=[
                Message(**msg) if isinstance(msg, dict) else msg
                for msg in agent_data.get("messages", [])
            ],
            metadata=agent_data.get("metadata", {})
        )

        # Write to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            json.dump(agent_file.model_dump(), f, indent=2)

        return output

    @staticmethod
    def _convert_letta_format(letta_data: Dict[str, Any]) -> AgentFile:
        """
        Convert Letta export format to AgentFile format.
        
        Args:
            letta_data: Letta export data
            
        Returns:
            AgentFile object
        """
        # Get the main agent (first agent in the list)
        agents = letta_data.get('agents', [])
        if not agents:
            raise ValueError("No agents found in Letta export")
        
        main_agent = agents[0]  # Use the first agent as the main one
        
        # Extract tools data
        tools_data = letta_data.get('tools', [])
        tools_map = {tool['id']: tool for tool in tools_data}
        
        # Convert tools to AgentForge format
        converted_tools = []
        for tool_id in main_agent.get('tool_ids', []):
            if tool_id in tools_map:
                tool = tools_map[tool_id]
                # Extract parameters from json_schema
                parameters = tool.get('json_schema', {}).get('properties', {})
                
                converted_tools.append(ToolSchema(
                    name=tool['name'],
                    description=tool['description'],
                    parameters=parameters,
                    code=tool.get('source_code')
                ))
        
        # Convert memory blocks
        blocks_data = letta_data.get('blocks', [])
        blocks_map = {block['id']: block for block in blocks_data}
        
        converted_memory_blocks = []
        for block_id in main_agent.get('block_ids', []):
            if block_id in blocks_map:
                block = blocks_map[block_id]
                converted_memory_blocks.append(MemoryBlock(
                    label=block['label'],
                    value=block['value'],
                    limit=block.get('limit')
                ))
        
        # Convert LLM config
        llm_config_data = main_agent.get('llm_config', {})
        converted_llm_config = LLMConfig(
            model=llm_config_data.get('model', 'gpt-4'),
            context_window=llm_config_data.get('context_window', 8000),
            temperature=llm_config_data.get('temperature', 0.7),
            max_tokens=llm_config_data.get('max_tokens')
        )
        
        # Convert messages
        converted_messages = []
        for message in main_agent.get('messages', []):
            if message.get('role') and message.get('content'):
                # Extract text content from the content array
                content_text = ""
                if isinstance(message['content'], list):
                    for content_item in message['content']:
                        if content_item.get('type') == 'text':
                            content_text = content_item.get('text', '')
                            break
                else:
                    content_text = str(message['content'])
                
                converted_messages.append(Message(
                    role=message['role'],
                    content=content_text,
                    timestamp=message.get('created_at')
                ))
        
        # Create AgentFile
        return AgentFile(
            name=main_agent.get('name', 'Imported Agent'),
            description=main_agent.get('description', 'Agent imported from Letta'),
            llm_config=converted_llm_config,
            system_prompt=main_agent.get('system', ''),
            memory_blocks=converted_memory_blocks,
            tools=converted_tools,
            messages=converted_messages,
            metadata={
                'source': 'letta',
                'original_id': main_agent.get('id'),
                'agent_type': main_agent.get('agent_type'),
                'imported_at': datetime.now().isoformat(),
                'original_export_created_at': letta_data.get('created_at')
            }
        )

    @staticmethod
    def deserialize(file_path: str) -> AgentFile:
        """
        Deserialize .af file to AgentFile object.
        
        Handles both Letta export format and AgentForge format.

        Args:
            file_path: Path to .af file

        Returns:
            AgentFile object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if it's Letta export format
        if 'agents' in data:
            # Convert from Letta format to AgentForge format
            return AgentFileSerializer._convert_letta_format(data)
        else:
            # Assume it's already AgentForge format
            return AgentFile(**data)

    @staticmethod
    def from_agentforge_agent(agent: Any) -> Dict[str, Any]:
        """
        Convert AgentForge agent to agentfile format.

        Args:
            agent: AgentForge agent instance

        Returns:
            Dict suitable for serialization
        """
        # Extract agent configuration
        agent_data = {
            "name": getattr(agent, "name", agent.__class__.__name__),
            "description": getattr(agent, "__doc__", "").strip() or f"AgentForge {agent.__class__.__name__}",
            "model": getattr(agent, "model", "anthropic/claude-3.5-sonnet"),
            "system_prompt": getattr(agent, "instructions", "") or getattr(agent, "system_prompt", ""),
            "memory_blocks": [],
            "tools": [],
            "messages": [],
            "metadata": {
                "source": "agentforge",
                "agent_class": agent.__class__.__name__,
                "created_by": "AgentForge CLI"
            }
        }

        # Extract tools if available
        if hasattr(agent, "tools"):
            for tool in agent.tools:
                tool_schema = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {})
                }
                agent_data["tools"].append(tool_schema)

        # Extract memory/state if available
        if hasattr(agent, "memory"):
            memory = agent.memory
            if isinstance(memory, dict):
                for key, value in memory.items():
                    agent_data["memory_blocks"].append({
                        "label": key,
                        "value": str(value)
                    })

        return agent_data

    @staticmethod
    def to_agentforge_agent(agent_file: AgentFile) -> Dict[str, Any]:
        """
        Convert AgentFile to AgentForge agent configuration.

        Args:
            agent_file: AgentFile object

        Returns:
            Dict with AgentForge agent configuration
        """
        config = {
            "name": agent_file.name,
            "description": agent_file.description,
            "model": agent_file.llm_config.model,
            "instructions": agent_file.system_prompt,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in agent_file.tools
            ],
            "memory": {
                block.label: block.value
                for block in agent_file.memory_blocks
            },
            "metadata": agent_file.metadata
        }

        return config


def save_agent(agent: Any, filepath: str) -> Path:
    """
    Save an AgentForge agent to .af file.

    Args:
        agent: Agent instance
        filepath: Output path

    Returns:
        Path to saved file
    """
    serializer = AgentFileSerializer()
    agent_data = serializer.from_agentforge_agent(agent)
    return serializer.serialize(agent_data, filepath)


def load_agent(filepath: str) -> Dict[str, Any]:
    """
    Load agent configuration from .af file.

    Args:
        filepath: Path to .af file

    Returns:
        Agent configuration dict
    """
    serializer = AgentFileSerializer()
    agent_file = serializer.deserialize(filepath)
    return serializer.to_agentforge_agent(agent_file)
