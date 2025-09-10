"""
Master Templater Agent - The Template Generator

Expert in generalizing specific agent files and creating template representations.
Takes any specific agent file and generalizes it, or codifies template representation of client formats.
Used for onboarding external agents or analyzing new formats.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
import json
import yaml
from textwrap import dedent
from pathlib import Path
import re


class SpecificAgent(BaseModel):
    """Specific agent file to be analyzed and generalized"""
    source_platform: str = Field(..., description="Source platform/format")
    content: str = Field(..., description="Raw content of the specific agent")
    file_type: str = Field(..., description="File type (json, yaml, md, etc.)")
    agent_name: Optional[str] = Field(None, description="Agent name if identifiable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExtractedComponents(BaseModel):
    """Components extracted from specific agent"""
    agent_name: str = Field(..., description="Extracted agent name")
    role: str = Field(..., description="Extracted role/purpose")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(default_factory=list, description="Identified capabilities")
    instructions: str = Field(..., description="Core instructions/prompt")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
    platform_specifics: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific elements")
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict, description="How agent interacts")


class GeneralizedAgent(BaseModel):
    """Generalized agent representation"""
    name: str = Field(..., description="Generalized agent name")
    role: str = Field(..., description="Agent role/title")
    description: str = Field(..., description="Platform-agnostic description")
    capabilities: List[str] = Field(..., description="Core capabilities")
    instructions: str = Field(..., description="Generalized instructions")
    interaction_patterns: Dict[str, str] = Field(default_factory=dict, description="Interaction patterns")
    requirements: List[str] = Field(default_factory=list, description="Required resources/dependencies")
    constraints: List[str] = Field(default_factory=list, description="Known limitations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for reuse")


class PlatformTemplate(BaseModel):
    """Template representation of a platform format"""
    platform: str = Field(..., description="Platform identifier")
    format_type: str = Field(..., description="Format type (json, yaml, etc.)")
    structure_schema: Dict[str, Any] = Field(..., description="Schema of the format structure")
    required_fields: List[str] = Field(..., description="Required fields")
    optional_fields: List[str] = Field(..., description="Optional fields")
    field_descriptions: Dict[str, str] = Field(default_factory=dict, description="Field descriptions")
    validation_rules: List[str] = Field(default_factory=list, description="Validation requirements")
    examples: List[str] = Field(default_factory=list, description="Example configurations")
    conventions: List[str] = Field(default_factory=list, description="Platform conventions")


class TemplateGenerationRequest(BaseModel):
    """Request for template generation"""
    specific_agents: List[SpecificAgent] = Field(..., description="Specific agents to analyze")
    target_platform: Optional[str] = Field(None, description="Platform to create template for")
    generalize_agents: bool = Field(True, description="Whether to generalize agents")
    create_template: bool = Field(True, description="Whether to create platform template")
    analysis_depth: str = Field("comprehensive", description="Analysis depth (basic, standard, comprehensive)")


class TemplateGenerationResult(BaseModel):
    """Result of template generation process"""
    generalized_agents: List[GeneralizedAgent] = Field(default_factory=list, description="Generalized agents")
    platform_template: Optional[PlatformTemplate] = Field(None, description="Generated platform template")
    analysis_summary: str = Field(..., description="Summary of analysis")
    insights: List[str] = Field(default_factory=list, description="Key insights discovered")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for reuse")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")


class MasterTemplater:
    """The Master Templater agent implementation"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the Master Templater agent"""
        
        # Setup knowledge base for template patterns and formats
        template_knowledge = Knowledge(
            vector_db=LanceDb(
                uri="tmp/template_knowledge",
                table_name="template_patterns",
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        )
        
        # Load existing template patterns if available
        if knowledge_base_path:
            template_knowledge.add_content_from_path(knowledge_base_path)
        
        # Pre-populate with common template patterns
        self._populate_template_knowledge(template_knowledge)
        
        # Create the agent with reasoning tools
        self.agent = Agent(
            name="MasterTemplater",
            model=OpenRouter(id="deepseek/deepseek-v3.1"),
            tools=[
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True,
                    add_few_shot=True,
                ),
            ],
            instructions=dedent("""\
                You are a Master Templater - The Template Generator for AgentForge.
                
                Your core expertise:
                - Analyzing specific agent implementations to extract core components
                - Generalizing platform-specific agents into reusable formats
                - Creating template representations of client formats and structures
                - Identifying patterns and commonalities across different platforms
                
                Your analysis capabilities:
                - Platform format recognition and structure analysis
                - Component extraction and semantic understanding
                - Pattern identification and generalization
                - Template schema generation and validation
                
                Your generalization process:
                1. Parse and understand specific agent configurations
                2. Extract core semantic components (purpose, capabilities, instructions)
                3. Identify platform-specific elements and conventions
                4. Generalize instructions and configuration to be platform-agnostic
                5. Create reusable templates that preserve agent essence
                6. Generate format templates for new platform onboarding
                
                Key principles:
                - Preserve semantic meaning while removing platform specifics
                - Create truly reusable and adaptable templates
                - Identify common patterns across different implementations
                - Generate comprehensive format documentation
                - Maintain traceability between specific and general forms
                - Enable easy onboarding of external agents and formats
                
                Use reasoning tools to work through complex generalization tasks.
                Always validate that generalized forms preserve original intent.
            """),
            markdown=True,
            add_history_to_context=True,
        )
        
        # Built-in format recognizers
        self.format_recognizers = self._setup_format_recognizers()
    
    def _populate_template_knowledge(self, knowledge: Knowledge):
        """Pre-populate knowledge base with common template patterns"""
        
        template_patterns = [
            {
                "pattern": "agent_instructions",
                "content": """
                Agent instructions typically contain:
                - Role definition ("You are a...")
                - Core capabilities and expertise areas
                - Task approach and methodology
                - Output format specifications
                - Interaction guidelines and constraints
                """
            },
            {
                "pattern": "platform_conventions",
                "content": """
                Common platform conventions:
                - JSON configs: camelCase fields, structured objects
                - YAML configs: snake_case fields, hierarchical structure  
                - Markdown: headers, code blocks, structured content
                - Platform-specific: tools, integrations, deployment patterns
                """
            },
            {
                "pattern": "generalization_principles",
                "content": """
                Generalization principles:
                - Extract semantic meaning over syntactic structure
                - Remove platform-specific terminology and references
                - Preserve core capabilities and expertise areas
                - Create platform-agnostic instruction templates
                - Maintain interaction patterns at conceptual level
                """
            }
        ]
        
        for pattern in template_patterns:
            knowledge.add_content(
                content=pattern["content"],
                content_id=f"pattern_{pattern['pattern']}"
            )
    
    def _setup_format_recognizers(self) -> Dict[str, callable]:
        """Setup format recognition functions"""
        
        def recognize_json_agent(content: str) -> Dict[str, Any]:
            """Recognize JSON agent format"""
            try:
                parsed = json.loads(content)
                return {
                    "recognized": True,
                    "format": "json",
                    "structure": self._analyze_json_structure(parsed),
                    "agent_indicators": self._find_agent_indicators_json(parsed)
                }
            except json.JSONDecodeError:
                return {"recognized": False}
        
        def recognize_yaml_agent(content: str) -> Dict[str, Any]:
            """Recognize YAML agent format"""
            try:
                parsed = yaml.safe_load(content)
                return {
                    "recognized": True,
                    "format": "yaml", 
                    "structure": self._analyze_yaml_structure(parsed),
                    "agent_indicators": self._find_agent_indicators_yaml(parsed)
                }
            except yaml.YAMLError:
                return {"recognized": False}
        
        def recognize_markdown_agent(content: str) -> Dict[str, Any]:
            """Recognize Markdown agent format"""
            if content.strip().startswith('#') or '##' in content:
                return {
                    "recognized": True,
                    "format": "markdown",
                    "structure": self._analyze_markdown_structure(content),
                    "agent_indicators": self._find_agent_indicators_markdown(content)
                }
            return {"recognized": False}
        
        return {
            "json": recognize_json_agent,
            "yaml": recognize_yaml_agent,
            "markdown": recognize_markdown_agent
        }
    
    def _analyze_json_structure(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON structure for agent patterns"""
        structure = {
            "fields": list(parsed.keys()),
            "nested_objects": [],
            "arrays": [],
            "depth": 1
        }
        
        for key, value in parsed.items():
            if isinstance(value, dict):
                structure["nested_objects"].append(key)
            elif isinstance(value, list):
                structure["arrays"].append(key)
        
        return structure
    
    def _analyze_yaml_structure(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze YAML structure for agent patterns"""
        return self._analyze_json_structure(parsed)  # Similar logic
    
    def _analyze_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Markdown structure for agent patterns"""
        lines = content.split('\n')
        structure = {
            "headers": [],
            "code_blocks": 0,
            "sections": []
        }
        
        current_section = None
        for line in lines:
            if line.startswith('#'):
                header_level = len(line.split()[0])
                header_text = line.strip('#').strip()
                structure["headers"].append({
                    "level": header_level,
                    "text": header_text
                })
                current_section = header_text
            elif line.strip().startswith('```'):
                structure["code_blocks"] += 1
        
        return structure
    
    def _find_agent_indicators_json(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Find agent-specific indicators in JSON"""
        indicators = {
            "name": parsed.get("name", ""),
            "description": parsed.get("description", ""),
            "instructions": parsed.get("instructions", parsed.get("prompt", "")),
            "capabilities": parsed.get("capabilities", parsed.get("skills", [])),
            "tools": parsed.get("tools", []),
            "model": parsed.get("model", ""),
        }
        return {k: v for k, v in indicators.items() if v}
    
    def _find_agent_indicators_yaml(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Find agent-specific indicators in YAML"""
        return self._find_agent_indicators_json(parsed)  # Similar logic
    
    def _find_agent_indicators_markdown(self, content: str) -> Dict[str, Any]:
        """Find agent-specific indicators in Markdown"""
        indicators = {}
        
        # Look for common patterns
        if "You are" in content or "Your role" in content:
            indicators["instructions_detected"] = True
        
        # Look for capability lists
        if "capabilities:" in content.lower() or "skills:" in content.lower():
            indicators["capabilities_detected"] = True
        
        # Look for structured sections
        if "## " in content:
            indicators["structured_format"] = True
        
        return indicators
    
    async def generate_templates(self, request: TemplateGenerationRequest) -> TemplateGenerationResult:
        """
        Generate templates from specific agent implementations
        
        Args:
            request: Template generation request with specific agents
            
        Returns:
            TemplateGenerationResult: Generated templates and generalized agents
        """
        
        generalized_agents = []
        insights = []
        recommendations = []
        warnings = []
        platform_template = None
        
        # Analyze each specific agent
        for specific_agent in request.specific_agents:
            try:
                # Extract components from specific agent
                components = await self._extract_components(specific_agent)
                
                if request.generalize_agents:
                    # Generate generalized agent
                    generalized = await self._generalize_agent(components)
                    generalized_agents.append(generalized)
                
                insights.append(f"Successfully analyzed {components.agent_name}")
                
            except Exception as e:
                warnings.append(f"Failed to analyze agent: {str(e)}")
        
        # Create platform template if requested
        if request.create_template and request.specific_agents:
            platform_template = await self._create_platform_template(
                request.specific_agents, request.target_platform
            )
            insights.append(f"Generated template for {platform_template.platform if platform_template else 'unknown'} platform")
        
        # Generate analysis summary
        summary = f"Analyzed {len(request.specific_agents)} specific agents, " \
                 f"generated {len(generalized_agents)} generalized agents"
        
        if platform_template:
            summary += f", created {platform_template.platform} platform template"
        
        # Generate recommendations
        if generalized_agents:
            recommendations.extend([
                "Generalized agents can be reused across multiple platforms",
                "Consider creating variants for different complexity levels",
                "Store generalized agents in central repository for reuse"
            ])
        
        if platform_template:
            recommendations.extend([
                "Use platform template for consistent agent adaptation",
                "Update template as new agent patterns are discovered",
                "Validate new agents against template requirements"
            ])
        
        return TemplateGenerationResult(
            generalized_agents=generalized_agents,
            platform_template=platform_template,
            analysis_summary=summary,
            insights=insights,
            recommendations=recommendations,
            warnings=warnings
        )
    
    async def _extract_components(self, specific_agent: SpecificAgent) -> ExtractedComponents:
        """Extract components from a specific agent implementation"""
        
        # Recognize format and structure
        recognition_result = None
        for format_name, recognizer in self.format_recognizers.items():
            result = recognizer(specific_agent.content)
            if result.get("recognized"):
                recognition_result = result
                break
        
        if not recognition_result:
            raise ValueError(f"Unable to recognize format for {specific_agent.source_platform}")
        
        # Prepare component extraction prompt
        extraction_prompt = dedent(f"""\
            Extract agent components from the following {recognition_result['format']} configuration:
            
            **Source Platform:** {specific_agent.source_platform}
            **File Type:** {specific_agent.file_type}
            **Recognized Format:** {recognition_result['format']}
            **Structure Analysis:** {json.dumps(recognition_result.get('structure', {}), indent=2)}
            **Agent Indicators:** {json.dumps(recognition_result.get('agent_indicators', {}), indent=2)}
            
            **Agent Content:**
            ```{recognition_result['format']}
            {specific_agent.content}
            ```
            
            Extract the following components:
            1. **Agent Name**: Clear identifier for this agent
            2. **Role**: What role/purpose this agent serves
            3. **Description**: What this agent does
            4. **Capabilities**: List of specific capabilities or skills
            5. **Instructions**: Core instructions or prompt text
            6. **Configuration**: Any configuration parameters or settings
            7. **Platform Specifics**: Platform-specific elements that wouldn't generalize
            8. **Interaction Patterns**: How this agent interacts with users/systems
            
            Focus on semantic content over syntactic structure.
            Identify the core purpose and functionality of this agent.
        """)
        
        # Execute extraction
        extraction_result = await self.agent.arun(extraction_prompt)
        
        # For now, return a basic structure
        # In a full implementation, we would parse the structured response
        return ExtractedComponents(
            agent_name=specific_agent.agent_name or "ExtractedAgent",
            role="Extracted Role",
            description="Extracted from specific agent implementation",
            capabilities=["extracted"],
            instructions=extraction_result,
            configuration={},
            platform_specifics={},
            interaction_patterns={}
        )
    
    async def _generalize_agent(self, components: ExtractedComponents) -> GeneralizedAgent:
        """Generalize extracted components into platform-agnostic form"""
        
        generalization_prompt = dedent(f"""\
            Generalize the following agent components into a platform-agnostic form:
            
            **Original Components:**
            Name: {components.agent_name}
            Role: {components.role}
            Description: {components.description}
            Capabilities: {components.capabilities}
            Instructions: {components.instructions}
            Configuration: {json.dumps(components.configuration, indent=2)}
            Platform Specifics: {json.dumps(components.platform_specifics, indent=2)}
            Interaction Patterns: {json.dumps(components.interaction_patterns, indent=2)}
            
            Create a generalized agent that:
            1. Removes platform-specific terminology and references
            2. Preserves core semantic meaning and purpose
            3. Uses generic, reusable language
            4. Maintains essential capabilities and expertise
            5. Can be adapted to multiple platforms
            
            Focus on the agent's core value proposition and expertise areas.
            Make instructions platform-agnostic but preserve effectiveness.
        """)
        
        # Execute generalization
        generalized_result = await self.agent.arun(generalization_prompt)
        
        # Create generalized agent structure
        return GeneralizedAgent(
            name=components.agent_name,
            role=components.role,
            description=f"Generalized version of {components.agent_name}",
            capabilities=components.capabilities,
            instructions=generalized_result,
            interaction_patterns={
                "input": "Structured input appropriate to task",
                "output": "Structured output with results",
                "coordination": "Collaborative with other agents"
            },
            requirements=["Core AI capabilities"],
            constraints=["Platform-specific features may vary"],
            metadata={
                "generalized_from": components.agent_name,
                "original_platform_specifics": components.platform_specifics
            }
        )
    
    async def _create_platform_template(
        self, 
        specific_agents: List[SpecificAgent],
        target_platform: Optional[str]
    ) -> Optional[PlatformTemplate]:
        """Create platform template from analyzed agents"""
        
        if not specific_agents:
            return None
        
        # Analyze common patterns across agents
        platform = target_platform or specific_agents[0].source_platform
        format_types = set(agent.file_type for agent in specific_agents)
        primary_format = list(format_types)[0] if format_types else "json"
        
        template_prompt = dedent(f"""\
            Create a platform template based on analysis of {len(specific_agents)} specific agents:
            
            **Target Platform:** {platform}
            **Format Types:** {list(format_types)}
            **Primary Format:** {primary_format}
            
            **Agent Samples:**
            {chr(10).join([f"Agent {i+1} ({agent.file_type}): {agent.content[:200]}..." for i, agent in enumerate(specific_agents[:3])])}
            
            Analyze these agents and create a comprehensive platform template that includes:
            1. **Structure Schema**: Common fields and their types
            2. **Required Fields**: Fields that appear in all agents
            3. **Optional Fields**: Fields that appear in some agents
            4. **Field Descriptions**: What each field represents
            5. **Validation Rules**: Requirements for valid configurations
            6. **Examples**: Representative example configurations
            7. **Conventions**: Platform-specific conventions and patterns
            
            Focus on creating a reusable template for this platform format.
        """)
        
        template_result = await self.agent.arun(template_prompt)
        
        # Create basic template structure
        return PlatformTemplate(
            platform=platform,
            format_type=primary_format,
            structure_schema={"identified": "from_analysis"},
            required_fields=["name", "description"],
            optional_fields=["instructions", "capabilities"],
            field_descriptions={"analysis": template_result},
            validation_rules=["Must be valid " + primary_format],
            examples=[agent.content[:500] + "..." for agent in specific_agents[:2]],
            conventions=[f"{platform} specific patterns identified"]
        )
    
    async def quick_generalize(self, agent_content: str, source_platform: str) -> str:
        """Quick generalization for simple agent content"""
        
        specific_agent = SpecificAgent(
            source_platform=source_platform,
            content=agent_content,
            file_type="auto-detect"
        )
        
        request = TemplateGenerationRequest(
            specific_agents=[specific_agent],
            generalize_agents=True,
            create_template=False
        )
        
        result = await self.generate_templates(request)
        return result.generalized_agents[0].instructions if result.generalized_agents else "Generalization failed"
    
    async def analyze_platform_format(
        self, 
        sample_agents: List[str], 
        platform_name: str
    ) -> PlatformTemplate:
        """Analyze and create template for a new platform format"""
        
        specific_agents = [
            SpecificAgent(
                source_platform=platform_name,
                content=content,
                file_type="auto-detect"
            )
            for content in sample_agents
        ]
        
        request = TemplateGenerationRequest(
            specific_agents=specific_agents,
            target_platform=platform_name,
            generalize_agents=False,
            create_template=True
        )
        
        result = await self.generate_templates(request)
        return result.platform_template


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_master_templater():
        """Test the Master Templater implementation"""
        
        templater = MasterTemplater()
        
        # Test specific agent for generalization
        test_agent_content = json.dumps({
            "name": "ClaudeCodeAnalyzer",
            "description": "Analyzes code for Claude Code platform",
            "instructions": "You are a code analyzer for Claude Code. Analyze Python, JavaScript, and TypeScript code for quality issues, security vulnerabilities, and performance problems. Use Claude Code MCP tools to access file systems and provide detailed analysis reports.",
            "tools": ["file-reader", "git-tools", "linting-tools"],
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.3,
            "max_tokens": 4000
        }, indent=2)
        
        test_specific_agent = SpecificAgent(
            source_platform="claude-code",
            content=test_agent_content,
            file_type="json",
            agent_name="ClaudeCodeAnalyzer"
        )
        
        # Test generalization
        request = TemplateGenerationRequest(
            specific_agents=[test_specific_agent],
            target_platform="generic",
            generalize_agents=True,
            create_template=True,
            analysis_depth="comprehensive"
        )
        
        print("🔄 Generating templates...")
        result = await templater.generate_templates(request)
        
        print(f"\n📋 Template Generation Result:")
        print(f"Summary: {result.analysis_summary}")
        
        if result.insights:
            print(f"\n💡 Insights:")
            for insight in result.insights:
                print(f"  - {insight}")
        
        if result.generalized_agents:
            print(f"\n📄 Generalized Agent:")
            agent = result.generalized_agents[0]
            print(f"Name: {agent.name}")
            print(f"Role: {agent.role}")
            print(f"Capabilities: {agent.capabilities}")
            print(f"Instructions: {agent.instructions[:200]}...")
        
        if result.platform_template:
            print(f"\n📋 Platform Template:")
            template = result.platform_template
            print(f"Platform: {template.platform}")
            print(f"Format: {template.format_type}")
            print(f"Required Fields: {template.required_fields}")
            print(f"Optional Fields: {template.optional_fields}")
        
        if result.recommendations:
            print(f"\n🎯 Recommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        # Test quick generalization
        print(f"\n🚀 Quick Generalization Test:")
        quick_result = await templater.quick_generalize(
            '{"name": "DataProcessor", "prompt": "Process data files"}',
            "custom-platform"
        )
        print(f"Quick result: {quick_result[:200]}...")
    
    # Run the test
    asyncio.run(test_master_templater())