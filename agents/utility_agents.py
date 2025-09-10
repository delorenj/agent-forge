"""
Supporting Utility Agents for AgentForge system.

These agents are used for post-processing and deployment adaptation:
- Format Adaptation Expert: Adapts agent descriptions to different client formats
- Master Templater: Generalizes agents and creates format templates
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from .base import AgentForgeBase, AgentForgeInput, AgentForgeOutput
import json
import re


class ClientFormat(str, Enum):
    """Supported client formats for agent deployment."""
    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"
    AMAZON_Q = "amazon_q"
    GENERIC = "generic"
    CUSTOM = "custom"


class AgentTemplate(BaseModel):
    """Template structure for an agent."""
    name: str
    role: str
    system_prompt: str
    capabilities: List[str]
    tools: List[str]
    personality_traits: Optional[List[str]] = None
    format_specific: Optional[Dict[str, Any]] = None


class FormatTemplate(BaseModel):
    """Template for a specific client format."""
    format_name: str
    structure_template: str
    required_fields: List[str]
    optional_fields: List[str]
    formatting_rules: Dict[str, str]


class AdaptationRequest(BaseModel):
    """Request for agent format adaptation."""
    source_agent: AgentTemplate
    target_format: ClientFormat
    custom_template: Optional[FormatTemplate] = None
    adaptation_options: Optional[Dict[str, Any]] = None


class TemplateGenerationRequest(BaseModel):
    """Request for template generation from specific agent."""
    specific_agent_file: str
    generalization_level: str = Field(default="medium", description="low, medium, high")
    extract_format: bool = Field(default=True)


class FormatAdaptationExpert(AgentForgeBase):
    """
    Format Adaptation Expert - Adapts general agent descriptions to specific client formats.
    
    Capabilities:
    - Convert agents between different platform formats
    - Apply platform-specific formatting rules
    - Maintain agent functionality across platforms
    - Handle custom format templates
    """
    
    def __init__(self):
        super().__init__(
            name="FormatAdaptationExpert",
            description="Adapts agent descriptions to different client formats and platforms"
        )
        
        # Built-in format templates
        self.format_templates = {
            ClientFormat.CLAUDE_CODE: FormatTemplate(
                format_name="Claude Code",
                structure_template="""# {name} Agent

## Role
{role}

## System Prompt
```
{system_prompt}
```

## Capabilities
{capabilities_list}

## Tools
{tools_list}

## Usage
```python
agent = Agent(
    name="{name}",
    system_prompt="{system_prompt}",
    tools={tools_array}
)
```
""",
                required_fields=["name", "role", "system_prompt"],
                optional_fields=["capabilities", "tools", "personality_traits"],
                formatting_rules={
                    "capabilities_list": "- {capability}",
                    "tools_list": "- {tool}",
                    "tools_array": "[{tools_comma_separated}]"
                }
            ),
            
            ClientFormat.OPENCODE: FormatTemplate(
                format_name="OpenCode",
                structure_template="""Agent: {name}
Role: {role}
System: {system_prompt}
Capabilities: {capabilities_comma_separated}
Tools: {tools_comma_separated}
""",
                required_fields=["name", "role", "system_prompt"],
                optional_fields=["capabilities", "tools"],
                formatting_rules={
                    "capabilities_comma_separated": "{capabilities}",
                    "tools_comma_separated": "{tools}"
                }
            ),
            
            ClientFormat.AMAZON_Q: FormatTemplate(
                format_name="Amazon Q",
                structure_template="""{
    "agentName": "{name}",
    "agentRole": "{role}",
    "systemPrompt": "{system_prompt}",
    "capabilities": {capabilities_json},
    "tools": {tools_json},
    "personalityTraits": {personality_json}
}""",
                required_fields=["name", "role", "system_prompt"],
                optional_fields=["capabilities", "tools", "personality_traits"],
                formatting_rules={
                    "capabilities_json": "[{capabilities_quoted}]",
                    "tools_json": "[{tools_quoted}]",
                    "personality_json": "[{personality_quoted}]"
                }
            )
        }
    
    async def adapt_agent(self, request: AdaptationRequest) -> str:
        """Adapt an agent to a specific client format."""
        
        # Get the appropriate template
        if request.target_format == ClientFormat.CUSTOM and request.custom_template:
            template = request.custom_template
        else:
            template = self.format_templates.get(request.target_format)
            if not template:
                raise ValueError(f"Unsupported format: {request.target_format}")
        
        # Prepare the agent data for formatting
        agent_data = self._prepare_agent_data(request.source_agent, template)
        
        # Apply the template
        adapted_content = template.structure_template.format(**agent_data)
        
        return adapted_content
    
    def _prepare_agent_data(self, agent: AgentTemplate, template: FormatTemplate) -> Dict[str, Any]:
        """Prepare agent data according to template formatting rules."""
        
        data = {
            "name": agent.name,
            "role": agent.role,
            "system_prompt": agent.system_prompt
        }
        
        # Process capabilities
        if agent.capabilities:
            data["capabilities_list"] = "\n".join([f"- {cap}" for cap in agent.capabilities])
            data["capabilities_comma_separated"] = ", ".join(agent.capabilities)
            data["capabilities_quoted"] = ", ".join([f'"{cap}"' for cap in agent.capabilities])
            data["capabilities_json"] = json.dumps(agent.capabilities)
        
        # Process tools
        if agent.tools:
            data["tools_list"] = "\n".join([f"- {tool}" for tool in agent.tools])
            data["tools_comma_separated"] = ", ".join(agent.tools)
            data["tools_quoted"] = ", ".join([f'"{tool}"' for tool in agent.tools])
            data["tools_json"] = json.dumps(agent.tools)
            data["tools_array"] = str(agent.tools)
        
        # Process personality traits
        if agent.personality_traits:
            data["personality_quoted"] = ", ".join([f'"{trait}"' for trait in agent.personality_traits])
            data["personality_json"] = json.dumps(agent.personality_traits)
        
        return data
    
    async def batch_adapt(self, agents: List[AgentTemplate], target_format: ClientFormat) -> Dict[str, str]:
        """Adapt multiple agents to the same format."""
        results = {}
        
        for agent in agents:
            request = AdaptationRequest(
                source_agent=agent,
                target_format=target_format
            )
            results[agent.name] = await self.adapt_agent(request)
        
        return results


class MasterTemplater(AgentForgeBase):
    """
    Master Templater - Generalizes specific agents and creates format templates.
    
    Capabilities:
    - Extract general patterns from specific agent implementations
    - Create reusable templates from agent examples
    - Analyze and codify new client formats
    - Generate format specifications
    """
    
    def __init__(self):
        super().__init__(
            name="MasterTemplater",
            description="Generalizes agents and creates reusable templates"
        )
    
    async def generalize_agent(self, request: TemplateGenerationRequest) -> AgentTemplate:
        """Extract a general template from a specific agent implementation."""
        
        # Read the specific agent file
        try:
            with open(request.specific_agent_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"Agent file not found: {request.specific_agent_file}")
        
        # Extract agent components using pattern matching and AI analysis
        extracted_data = await self._extract_agent_components(content)
        
        # Apply generalization based on level
        generalized_data = await self._apply_generalization(
            extracted_data, 
            request.generalization_level
        )
        
        # Create AgentTemplate
        template = AgentTemplate(
            name=generalized_data.get("name", "GeneralizedAgent"),
            role=generalized_data.get("role", "General Purpose Agent"),
            system_prompt=generalized_data.get("system_prompt", ""),
            capabilities=generalized_data.get("capabilities", []),
            tools=generalized_data.get("tools", []),
            personality_traits=generalized_data.get("personality_traits")
        )
        
        return template
    
    async def _extract_agent_components(self, content: str) -> Dict[str, Any]:
        """Extract agent components from file content using AI analysis."""
        
        extraction_prompt = f"""
        Analyze this agent file and extract the key components:
        
        {content}
        
        Extract:
        1. Agent name
        2. Role/purpose
        3. System prompt or main instructions
        4. Capabilities and skills
        5. Tools or functions used
        6. Personality traits or behavioral patterns
        
        Return as structured data.
        """
        
        # Use AI to analyze and extract components
        response = await self.run_with_mcp(extraction_prompt)
        
        # Parse the AI response into structured data
        # This would be more sophisticated in a real implementation
        extracted = {
            "name": self._extract_pattern(content, r"name[:\s]+([^\n]+)", "ExtractedAgent"),
            "role": self._extract_pattern(content, r"role[:\s]+([^\n]+)", "General Agent"),
            "system_prompt": self._extract_system_prompt(content),
            "capabilities": self._extract_list_pattern(content, r"capabilit(?:y|ies)[:\s]*([^\n]+)"),
            "tools": self._extract_list_pattern(content, r"tools?[:\s]*([^\n]+)"),
            "personality_traits": self._extract_list_pattern(content, r"personality[:\s]*([^\n]+)")
        }
        
        return extracted
    
    def _extract_pattern(self, content: str, pattern: str, default: str = "") -> str:
        """Extract a pattern from content."""
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _extract_system_prompt(self, content: str) -> str:
        """Extract system prompt from content."""
        # Look for common system prompt patterns
        patterns = [
            r"system[_\s]*prompt[:\s]*[\"']([^\"']+)[\"']",
            r"prompt[:\s]*[\"']([^\"']+)[\"']",
            r"instructions?[:\s]*[\"']([^\"']+)[\"']"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no specific prompt found, extract first large text block
        lines = content.split('\n')
        for line in lines:
            if len(line) > 50 and not line.startswith('#'):
                return line.strip()
        
        return "General purpose agent"
    
    def _extract_list_pattern(self, content: str, pattern: str) -> List[str]:
        """Extract list items from content."""
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            return []
        
        items_str = match.group(1)
        # Split by common separators
        items = re.split(r'[,;|\n]', items_str)
        return [item.strip() for item in items if item.strip()]
    
    async def _apply_generalization(self, data: Dict[str, Any], level: str) -> Dict[str, Any]:
        """Apply generalization based on the specified level."""
        
        generalization_prompt = f"""
        Generalize this agent data to {level} level:
        
        {json.dumps(data, indent=2)}
        
        Generalization levels:
        - low: Make minor adjustments, keep most specifics
        - medium: Remove domain-specific details, keep general structure
        - high: Create highly generic, reusable patterns
        
        Return generalized version maintaining the same structure.
        """
        
        # Use AI to perform intelligent generalization
        response = await self.run_with_mcp(generalization_prompt)
        
        # For now, apply basic generalization rules
        generalized = data.copy()
        
        if level == "high":
            # High generalization - make very generic
            generalized["system_prompt"] = self._generalize_prompt_high(data.get("system_prompt", ""))
            generalized["capabilities"] = self._generalize_list_high(data.get("capabilities", []))
        elif level == "medium":
            # Medium generalization - remove specifics
            generalized["system_prompt"] = self._generalize_prompt_medium(data.get("system_prompt", ""))
            generalized["capabilities"] = self._generalize_list_medium(data.get("capabilities", []))
        # Low level keeps most original content
        
        return generalized
    
    def _generalize_prompt_high(self, prompt: str) -> str:
        """Apply high-level generalization to system prompt."""
        # Replace specific terms with generic ones
        replacements = {
            r'\b\w+\.com\b': 'website',
            r'\b\d{4}\b': 'year',
            r'\b\$\d+\b': 'budget',
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': 'person name'
        }
        
        generalized = prompt
        for pattern, replacement in replacements.items():
            generalized = re.sub(pattern, replacement, generalized)
        
        return generalized
    
    def _generalize_prompt_medium(self, prompt: str) -> str:
        """Apply medium-level generalization to system prompt."""
        # Remove some specifics but keep domain context
        replacements = {
            r'\b\d{4}-\d{2}-\d{2}\b': 'date',
            r'\bversion \d+\.\d+\b': 'version number'
        }
        
        generalized = prompt
        for pattern, replacement in replacements.items():
            generalized = re.sub(pattern, replacement, generalized)
        
        return generalized
    
    def _generalize_list_high(self, items: List[str]) -> List[str]:
        """Apply high-level generalization to lists."""
        generalized = []
        for item in items:
            # Replace specific technologies with categories
            if any(tech in item.lower() for tech in ['react', 'vue', 'angular']):
                generalized.append('frontend framework')
            elif any(tech in item.lower() for tech in ['python', 'java', 'javascript']):
                generalized.append('programming language')
            else:
                generalized.append(item)
        
        return list(set(generalized))  # Remove duplicates
    
    def _generalize_list_medium(self, items: List[str]) -> List[str]:
        """Apply medium-level generalization to lists."""
        # Keep more specifics than high level
        return items
    
    async def create_format_template(self, examples: List[str]) -> FormatTemplate:
        """Create a format template by analyzing examples."""
        
        analysis_prompt = f"""
        Analyze these agent format examples and create a template:
        
        {chr(10).join([f"Example {i+1}:\n{ex}\n" for i, ex in enumerate(examples)])}
        
        Extract:
        1. Common structure pattern
        2. Required fields present in all examples
        3. Optional fields present in some examples
        4. Formatting rules and conventions
        
        Create a template that can reproduce similar formats.
        """
        
        response = await self.run_with_mcp(analysis_prompt)
        
        # Create a basic template (would be more sophisticated in real implementation)
        template = FormatTemplate(
            format_name="ExtractedFormat",
            structure_template=self._extract_structure_pattern(examples),
            required_fields=self._extract_required_fields(examples),
            optional_fields=self._extract_optional_fields(examples),
            formatting_rules=self._extract_formatting_rules(examples)
        )
        
        return template
    
    def _extract_structure_pattern(self, examples: List[str]) -> str:
        """Extract common structure pattern from examples."""
        # Simplified pattern extraction
        return "# {name}\nRole: {role}\nPrompt: {system_prompt}\nCapabilities: {capabilities}"
    
    def _extract_required_fields(self, examples: List[str]) -> List[str]:
        """Extract required fields from examples."""
        return ["name", "role", "system_prompt"]
    
    def _extract_optional_fields(self, examples: List[str]) -> List[str]:
        """Extract optional fields from examples."""
        return ["capabilities", "tools", "personality_traits"]
    
    def _extract_formatting_rules(self, examples: List[str]) -> Dict[str, str]:
        """Extract formatting rules from examples."""
        return {
            "capabilities": "comma_separated",
            "tools": "bullet_list"
        }


# Factory functions for easy instantiation
def create_format_expert() -> FormatAdaptationExpert:
    """Create a Format Adaptation Expert."""
    return FormatAdaptationExpert()

def create_master_templater() -> MasterTemplater:
    """Create a Master Templater."""
    return MasterTemplater()


# Example usage
async def demo_utility_agents():
    """Demonstrate the utility agents."""
    print("🔧 AgentForge Utility Agents Demo")
    print("=" * 40)
    
    # Create agents
    format_expert = create_format_expert()
    templater = create_master_templater()
    
    # Create a sample agent template
    sample_agent = AgentTemplate(
        name="TaskManager",
        role="Project task management specialist",
        system_prompt="You are a task management expert who helps organize and prioritize work.",
        capabilities=["task prioritization", "deadline management", "team coordination"],
        tools=["calendar", "task tracker", "notification system"],
        personality_traits=["organized", "detail-oriented", "proactive"]
    )
    
    print(f"\n📋 Original Agent: {sample_agent.name}")
    print(f"Role: {sample_agent.role}")
    print(f"Capabilities: {len(sample_agent.capabilities)}")
    
    # Test Format Adaptation Expert
    print("\n🔄 Testing Format Adaptation Expert:")
    
    for format_type in [ClientFormat.CLAUDE_CODE, ClientFormat.OPENCODE, ClientFormat.AMAZON_Q]:
        adaptation_request = AdaptationRequest(
            source_agent=sample_agent,
            target_format=format_type
        )
        
        adapted = await format_expert.adapt_agent(adaptation_request)
        print(f"\n--- {format_type.value.upper()} Format ---")
        print(adapted[:200] + "..." if len(adapted) > 200 else adapted)
    
    print(f"\n✅ Format Adaptation Expert: Successfully adapted to {len(format_expert.format_templates)} formats")
    
    # Note: Master Templater would require actual agent files to demonstrate
    print(f"\n✅ Master Templater: Ready for agent generalization and template creation")
    
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_utility_agents())