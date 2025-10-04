#!/usr/bin/env python3
"""
Convert Letta export (.af file) to AgentForge AgentFile format.

This script converts the Letta export format to the AgentForge AgentFile format
that the system expects.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from agentfile import AgentFile, LLMConfig, ToolSchema, MemoryBlock, Message


def convert_letta_to_agentforge(letta_file_path: str, output_path: str = None) -> str:
    """
    Convert Letta export to AgentForge format.
    
    Args:
        letta_file_path: Path to Letta export file
        output_path: Optional output path (defaults to same name with _converted suffix)
    
    Returns:
        Path to converted file
    """
    # Load Letta export
    with open(letta_file_path, 'r') as f:
        letta_data = json.load(f)
    
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
    agent_file = AgentFile(
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
            'imported_at': datetime.utcnow().isoformat(),
            'original_export_created_at': letta_data.get('created_at')
        }
    )
    
    # Determine output path
    if output_path is None:
        input_path = Path(letta_file_path)
        output_path = input_path.parent / f"{input_path.stem}_converted.af"
    
    # Write converted file
    with open(output_path, 'w') as f:
        json.dump(agent_file.model_dump(), f, indent=2)
    
    return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_letta_to_agentforge.py <letta_export_file> [output_file]")
        sys.exit(1)
    
    letta_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        converted_path = convert_letta_to_agentforge(letta_file, output_file)
        print(f"Successfully converted Letta export to: {converted_path}")
        
        # Show summary
        with open(converted_path, 'r') as f:
            data = json.load(f)
        
        print(f"\nConversion Summary:")
        print(f"- Agent name: {data['name']}")
        print(f"- Description: {data['description']}")
        print(f"- Model: {data['llm_config']['model']}")
        print(f"- Memory blocks: {len(data['memory_blocks'])}")
        print(f"- Tools: {len(data['tools'])}")
        print(f"- Messages: {len(data['messages'])}")
        
    except Exception as e:
        print(f"Error converting file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
