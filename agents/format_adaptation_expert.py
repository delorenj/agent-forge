"""
Format Adaptation Expert Agent - The Platform Adapter

Expert in adapting general agent descriptions to specific client formats.
Converts AgentForge roster outputs to platform-specific formats like Claude Code, OpenCode, AmazonQ.
Used for deploying finalized rosters to specific environments.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import logging
import uuid
from os import getenv
from textwrap import dedent
from pathlib import Path

logger = logging.getLogger(__name__)


class SourceAgent(BaseModel):
    """Source agent definition from AgentForge"""
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role/title")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    instructions: str = Field(..., description="Agent instructions/prompt")
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict, description="How agent interacts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PlatformTemplate(BaseModel):
    """Platform-specific template structure"""
    platform: str = Field(..., description="Target platform (claude-code, opencode, amazonq, etc.)")
    format_type: str = Field(..., description="Format type (json, yaml, md, etc.)")
    structure: Dict[str, Any] = Field(..., description="Template structure and fields")
    example: Optional[str] = Field(None, description="Example formatted output")
    requirements: List[str] = Field(default_factory=list, description="Platform-specific requirements")
    constraints: List[str] = Field(default_factory=list, description="Platform limitations")


class AdaptationRequest(BaseModel):
    """Request for format adaptation"""
    source_agents: List[SourceAgent] = Field(..., description="Source agent definitions")
    target_platform: str = Field(..., description="Target platform identifier")
    platform_template: Optional[PlatformTemplate] = Field(None, description="Platform template if provided")
    customizations: Dict[str, Any] = Field(default_factory=dict, description="Custom adaptations")
    preserve_semantics: bool = Field(True, description="Whether to preserve semantic meaning")


class AdaptedAgent(BaseModel):
    """Agent adapted to target platform"""
    original_name: str = Field(..., description="Original agent name")
    adapted_content: str = Field(..., description="Platform-specific formatted content")
    platform: str = Field(..., description="Target platform")
    format_type: str = Field(..., description="Output format type")
    adaptation_notes: List[str] = Field(default_factory=list, description="Notes about adaptations made")
    validation_status: str = Field(..., description="Validation status (valid, warning, error)")


class AdaptationResult(BaseModel):
    """Complete adaptation result"""
    adapted_agents: List[AdaptedAgent] = Field(..., description="All adapted agents")
    platform: str = Field(..., description="Target platform")
    summary: str = Field(..., description="Adaptation summary")
    warnings: List[str] = Field(default_factory=list, description="Adaptation warnings")
    deployment_instructions: List[str] = Field(default_factory=list, description="How to deploy adapted agents")


class FormatAdaptationExpert:
    """The Format Adaptation Expert agent implementation"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the Format Adaptation Expert agent"""
        
        # Initialize embedder for format knowledge
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup QdrantClient for knowledge storage
        self.qdrant_client = QdrantClient(host="localhost", port=6333, api_key="touchmyflappyfoldyholds")
        self.collection_name = "platform_formats"
        
        # Initialize collection if it doesn't exist
        self._initialize_collection()
        
        # Load format examples and templates if available
        if knowledge_base_path:
            self._load_knowledge_from_path(knowledge_base_path)
        
        # Pre-populate with common platform knowledge
        self._populate_format_knowledge()
        
        # Agent configuration
        self.name = "FormatAdaptationExpert"
        self.instructions = dedent("""\
                You are a Format Adaptation Expert - The Platform Adapter for AgentForge.
                
                Your core expertise:
                - Converting general agent descriptions to specific platform formats
                - Understanding platform-specific requirements and constraints
                - Preserving semantic meaning while adapting to format requirements
                - Generating deployment-ready agent configurations
                
                Platform specializations:
                - Claude Code: MCP tools, system prompts, function definitions
                - OpenCode: VS Code extensions, configuration schemas
                - Amazon Q: AWS-specific integrations, skill formats
                - Custom platforms: Flexible adaptation to any schema
                
                Your adaptation process:
                1. Analyze source agent definitions and extract core components
                2. Study target platform requirements and format specifications
                3. Map agent capabilities to platform-specific features
                4. Transform instructions and prompts to platform conventions
                5. Generate valid, deployable configurations
                6. Provide deployment instructions and validation
                
                Key principles:
                - Preserve the agent's core purpose and capabilities
                - Follow platform conventions and best practices
                - Validate output against platform requirements
                - Provide clear deployment instructions
                - Note any semantic changes or limitations
                - Ensure configurations are production-ready
                
                Use reasoning tools to work through complex format transformations.
                Always validate adapted output for platform compliance.
            """)
        
        # Built-in platform templates
        self.platform_templates = self._load_builtin_templates()
    
    def _initialize_collection(self):
        """Initialize QdrantClient collection for format knowledge"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")

    def _load_knowledge_from_path(self, knowledge_base_path: str):
        """Load format knowledge from path"""
        try:
            knowledge_path = Path(knowledge_base_path)
            if knowledge_path.exists():
                # Load platform format examples from files
                logger.info(f"Loading knowledge from: {knowledge_base_path}")
                # Implementation would read format files and add to QdrantClient
            else:
                logger.warning(f"Knowledge path does not exist: {knowledge_base_path}")
        except Exception as e:
            logger.error(f"Error loading knowledge from path: {e}")

    def _populate_format_knowledge(self):
        """Pre-populate knowledge base with common platform formats"""
        
        # Add common platform examples
        platform_examples = [
            {
                "platform": "claude-code",
                "content": """
                Claude Code agents use MCP (Model Context Protocol) tools and system prompts.
                Format: JSON with name, description, instructions, and tools array.
                Example: {"name": "Agent", "instructions": "You are...", "tools": []}
                """
            },
            {
                "platform": "opencode",
                "content": """
                OpenCode uses VS Code extension format with contribution points.
                Format: package.json with contributes section defining commands, views, etc.
                Example: {"contributes": {"commands": [{"command": "extension.command"}]}}
                """
            },
            {
                "platform": "amazonq",
                "content": """
                Amazon Q uses skill-based format with intents, slots, and responses.
                Format: JSON with skill definition, intents, and sample utterances.
                Example: {"skill": {"intents": [{"name": "Intent", "slots": []}]}}
                """
            }
        ]
        
        try:
            # Add examples to QdrantClient
            points = []
            for i, example in enumerate(platform_examples):
                content = example["content"]
                embedding = self.embedder.encode(content)
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "platform": example["platform"],
                        "content": content,
                        "content_id": f"platform_{example['platform']}"
                    }
                ))
            
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} platform examples to knowledge base")
                
        except Exception as e:
            logger.error(f"Error populating format knowledge: {e}")
    
    def _load_builtin_templates(self) -> Dict[str, PlatformTemplate]:
        """Load built-in platform templates"""
        
        templates = {}
        
        # Claude Code template
        templates["claude-code"] = PlatformTemplate(
            platform="claude-code",
            format_type="json",
            structure={
                "name": "string",
                "description": "string", 
                "instructions": "string",
                "tools": "array",
                "model": "string",
                "temperature": "number"
            },
            example=json.dumps({
                "name": "ExampleAgent",
                "description": "An example agent for Claude Code",
                "instructions": "You are an expert assistant...",
                "tools": [],
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7
            }, indent=2),
            requirements=[
                "Instructions must be clear and specific",
                "Tools must be valid MCP tool definitions",
                "Name should be PascalCase"
            ],
            constraints=[
                "Instructions limited to reasonable length",
                "Tools must be available in MCP ecosystem"
            ]
        )
        
        # OpenCode template  
        templates["opencode"] = PlatformTemplate(
            platform="opencode",
            format_type="json",
            structure={
                "name": "string",
                "displayName": "string",
                "description": "string",
                "version": "string",
                "contributes": "object"
            },
            example=json.dumps({
                "name": "example-agent",
                "displayName": "Example Agent",
                "description": "An example agent for OpenCode",
                "version": "1.0.0",
                "contributes": {
                    "commands": [{
                        "command": "example.activate",
                        "title": "Activate Example Agent"
                    }]
                }
            }, indent=2),
            requirements=[
                "Must follow VS Code extension conventions",
                "Commands must be unique",
                "Version must be semver format"
            ],
            constraints=[
                "Name must be lowercase with hyphens",
                "Limited to VS Code API capabilities"
            ]
        )
        
        # Amazon Q template
        templates["amazonq"] = PlatformTemplate(
            platform="amazonq",
            format_type="json",
            structure={
                "skillId": "string",
                "name": "string",
                "description": "string",
                "intents": "array",
                "responses": "object"
            },
            example=json.dumps({
                "skillId": "example-agent-skill",
                "name": "Example Agent",
                "description": "An example agent skill for Amazon Q",
                "intents": [{
                    "name": "ProcessRequest",
                    "slots": [],
                    "sampleUtterances": ["help me with", "assist with"]
                }],
                "responses": {
                    "ProcessRequest": "I'll help you with that request."
                }
            }, indent=2),
            requirements=[
                "Must define at least one intent",
                "Sample utterances required for each intent",
                "Responses must be defined for each intent"
            ],
            constraints=[
                "Limited to AWS Q capabilities",
                "Intent names must be unique"
            ]
        )
        
        return templates
    
    async def adapt_agents(self, request: AdaptationRequest) -> AdaptationResult:
        """
        Adapt source agents to target platform format
        
        Args:
            request: Adaptation request with source agents and target platform
            
        Returns:
            AdaptationResult: Complete adaptation with all transformed agents
        """
        
        # Get or validate platform template
        platform_template = request.platform_template or self.platform_templates.get(request.target_platform)
        
        if not platform_template:
            raise ValueError(f"No template available for platform: {request.target_platform}")
        
        adapted_agents = []
        warnings = []
        
        # Adapt each source agent
        for source_agent in request.source_agents:
            try:
                adapted_agent = await self._adapt_single_agent(
                    source_agent, platform_template, request
                )
                adapted_agents.append(adapted_agent)
                
            except Exception as e:
                warnings.append(f"Failed to adapt {source_agent.name}: {str(e)}")
                # Create error placeholder
                adapted_agents.append(AdaptedAgent(
                    original_name=source_agent.name,
                    adapted_content=f"# ERROR: Failed to adapt {source_agent.name}\n# {str(e)}",
                    platform=request.target_platform,
                    format_type="error",
                    adaptation_notes=[f"Adaptation failed: {str(e)}"],
                    validation_status="error"
                ))
        
        # Generate deployment instructions
        deployment_instructions = self._generate_deployment_instructions(
            platform_template, adapted_agents
        )
        
        # Create summary
        successful_adaptations = len([a for a in adapted_agents if a.validation_status == "valid"])
        summary = f"Adapted {successful_adaptations}/{len(request.source_agents)} agents for {request.target_platform}"
        
        return AdaptationResult(
            adapted_agents=adapted_agents,
            platform=request.target_platform,
            summary=summary,
            warnings=warnings,
            deployment_instructions=deployment_instructions
        )
    
    async def _adapt_single_agent(
        self, 
        source_agent: SourceAgent, 
        platform_template: PlatformTemplate,
        request: AdaptationRequest
    ) -> AdaptedAgent:
        """Adapt a single agent to target platform format"""
        
        # Prepare adaptation prompt
        adaptation_prompt = dedent(f"""\
            Adapt the following agent to {platform_template.platform} format:
            
            **Source Agent:**
            Name: {source_agent.name}
            Role: {source_agent.role}
            Description: {source_agent.description}
            Capabilities: {source_agent.capabilities}
            Instructions: {source_agent.instructions}
            Interaction Patterns: {source_agent.interaction_patterns}
            Metadata: {source_agent.metadata}
            
            **Target Platform:** {platform_template.platform}
            **Format Type:** {platform_template.format_type}
            **Platform Structure:** {json.dumps(platform_template.structure, indent=2)}
            **Platform Requirements:** {platform_template.requirements}
            **Platform Constraints:** {platform_template.constraints}
            
            **Example Format:**
            {platform_template.example}
            
            **Customizations:** {json.dumps(request.customizations, indent=2)}
            **Preserve Semantics:** {request.preserve_semantics}
            
            Create a valid {platform_template.platform} configuration that:
            1. Preserves the agent's core purpose and capabilities
            2. Follows platform conventions and requirements
            3. Maps agent features to platform-specific equivalents
            4. Includes all required fields and structures
            5. Is ready for deployment
            
            Return ONLY the adapted configuration in the correct format.
            Add adaptation notes as comments where appropriate.
        """)
        
        # Execute the adaptation using built-in logic
        adapted_content = self._perform_adaptation_logic(source_agent, platform_template, adaptation_prompt)
        
        # Validate the adapted content
        validation_status, validation_notes = self._validate_adapted_content(
            adapted_content, platform_template
        )
        
        return AdaptedAgent(
            original_name=source_agent.name,
            adapted_content=adapted_content,
            platform=platform_template.platform,
            format_type=platform_template.format_type,
            adaptation_notes=validation_notes,
            validation_status=validation_status
        )
    
    def _validate_adapted_content(
        self, 
        content: str, 
        template: PlatformTemplate
    ) -> tuple[str, List[str]]:
        """Validate adapted content against platform requirements"""
        
        notes = []
        
        try:
            # Try to parse JSON if it's a JSON format
            if template.format_type == "json":
                parsed = json.loads(content)
                
                # Check required fields
                for field in template.structure.keys():
                    if field not in parsed:
                        notes.append(f"Missing required field: {field}")
                
                if notes:
                    return "warning", notes
                else:
                    notes.append("All required fields present")
                    return "valid", notes
                    
            else:
                # Basic validation for other formats
                notes.append(f"Basic validation passed for {template.format_type}")
                return "valid", notes
                
        except json.JSONDecodeError as e:
            notes.append(f"JSON parsing error: {str(e)}")
            return "error", notes
        except Exception as e:
            notes.append(f"Validation error: {str(e)}")
            return "warning", notes
    
    def _generate_deployment_instructions(
        self, 
        template: PlatformTemplate, 
        adapted_agents: List[AdaptedAgent]
    ) -> List[str]:
        """Generate deployment instructions for the target platform"""
        
        instructions = []
        valid_agents = [a for a in adapted_agents if a.validation_status == "valid"]
        
        if template.platform == "claude-code":
            instructions.extend([
                "Claude Code Deployment:",
                "1. Save each agent configuration as a .json file",
                "2. Use 'claude mcp add <agent-name> <config-file>' to register",
                "3. Restart Claude Code to load new agents",
                "4. Verify agents appear in tools menu"
            ])
        
        elif template.platform == "opencode":
            instructions.extend([
                "OpenCode Deployment:",
                "1. Create VS Code extension directory",
                "2. Copy adapted configurations to package.json",
                "3. Run 'npm install' and 'vsce package'",
                "4. Install .vsix file in VS Code",
                "5. Activate extension in VS Code"
            ])
        
        elif template.platform == "amazonq":
            instructions.extend([
                "Amazon Q Deployment:",
                "1. Upload skill configurations to Q console",
                "2. Define intents and sample utterances",
                "3. Configure response templates",
                "4. Test skills in Q developer console",
                "5. Deploy to production environment"
            ])
        
        else:
            instructions.extend([
                f"{template.platform} Deployment:",
                "1. Review platform-specific documentation",
                "2. Follow platform deployment procedures",
                "3. Test adapted agents in target environment",
                "4. Monitor for compatibility issues"
            ])
        
        instructions.append(f"\nSuccessfully adapted {len(valid_agents)} agents ready for deployment.")
        
        return instructions
    
    async def quick_adapt(
        self, 
        agent_name: str, 
        agent_instructions: str, 
        target_platform: str
    ) -> str:
        """Quick adaptation for simple agent definitions"""
        
        source_agent = SourceAgent(
            name=agent_name,
            role="Generic Agent",
            description=f"Agent: {agent_name}",
            capabilities=["general"],
            instructions=agent_instructions
        )
        
        request = AdaptationRequest(
            source_agents=[source_agent],
            target_platform=target_platform
        )
        
        # Since adapt_agents is now synchronous, remove await
        # For now, implement simple quick adaptation
        template = self.platform_templates.get(target_platform)
        if template:
            adapted_content = self._perform_adaptation_logic(source_agent, template, "Quick adaptation request")
            return adapted_content
        else:
            return f"Platform {target_platform} not supported"

    def _perform_adaptation_logic(self, source_agent: SourceAgent, platform_template: PlatformTemplate, adaptation_prompt: str) -> str:
        """Perform agent adaptation using built-in logic instead of Agno agent"""
        try:
            # Extract key components from source agent
            name = source_agent.name
            role = source_agent.role
            description = source_agent.description
            capabilities = source_agent.capabilities
            instructions = source_agent.instructions
            
            # Platform-specific adaptation logic
            if platform_template.platform == "claude-code":
                adapted_config = {
                    "name": name,
                    "description": description,
                    "role": role,
                    "instructions": instructions,
                    "capabilities": capabilities,
                    "tools": [],  # MCP tools would be added here
                    "metadata": {
                        "platform": "claude-code",
                        "format": "mcp"
                    }
                }
                return json.dumps(adapted_config, indent=2)
                
            elif platform_template.platform == "opencode":
                adapted_config = {
                    "name": name,
                    "displayName": description,
                    "version": "1.0.0",
                    "contributes": {
                        "commands": [{
                            "command": f"extension.{name.lower()}",
                            "title": description
                        }]
                    },
                    "main": f"./out/{name.lower()}.js",
                    "scripts": {
                        "vscode:prepublish": "npm run compile"
                    }
                }
                return json.dumps(adapted_config, indent=2)
                
            elif platform_template.platform == "amazonq":
                adapted_config = {
                    "skill": {
                        "name": name,
                        "description": description,
                        "intents": [{
                            "name": f"{name}Intent",
                            "slots": [],
                            "samples": [description]
                        }],
                        "responses": {
                            f"{name}Intent": instructions[:500]  # Limit response length
                        }
                    }
                }
                return json.dumps(adapted_config, indent=2)
            
            else:
                # Generic JSON format adaptation
                adapted_config = {
                    "name": name,
                    "role": role,
                    "description": description,
                    "capabilities": capabilities,
                    "instructions": instructions,
                    "platform": platform_template.platform,
                    "format": platform_template.format_type
                }
                return json.dumps(adapted_config, indent=2)
                
        except Exception as e:
            logger.error(f"Error in adaptation logic: {e}")
            return f"{{\"error\": \"Adaptation failed: {str(e)}\", \"original_agent\": \"{source_agent.name}\"}}"


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_format_adaptation():
        """Test the Format Adaptation Expert implementation"""
        
        expert = FormatAdaptationExpert()
        
        # Test source agent
        test_agent = SourceAgent(
            name="CodeReviewer",
            role="Senior Code Reviewer",
            description="Expert in reviewing code for quality, security, and best practices",
            capabilities=[
                "Code analysis",
                "Security review", 
                "Performance optimization",
                "Best practices enforcement"
            ],
            instructions="You are a senior code reviewer with expertise in multiple programming languages. Review code for quality, security vulnerabilities, performance issues, and adherence to best practices. Provide constructive feedback with specific suggestions for improvement.",
            interaction_patterns={
                "input": "Code files or pull requests",
                "output": "Detailed review with recommendations",
                "coordination": "Works with development team"
            },
            metadata={
                "priority": "high",
                "complexity": "medium"
            }
        )
        
        # Test adaptation to Claude Code
        request = AdaptationRequest(
            source_agents=[test_agent],
            target_platform="claude-code",
            preserve_semantics=True
        )
        
        print("🔄 Adapting agent to Claude Code format...")
        result = await expert.adapt_agents(request)
        
        print(f"\n📋 Adaptation Result:")
        print(f"Platform: {result.platform}")
        print(f"Summary: {result.summary}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        print(f"\n📄 Adapted Agent:")
        print("=" * 50)
        print(result.adapted_agents[0].adapted_content)
        
        print(f"\n📋 Deployment Instructions:")
        for instruction in result.deployment_instructions:
            print(f"  {instruction}")
        
        # Test quick adaptation
        print(f"\n🚀 Quick Adaptation Test:")
        quick_result = await expert.quick_adapt(
            "DataAnalyst",
            "You are a data analyst expert. Analyze datasets and provide insights.",
            "opencode"
        )
        print(quick_result)
    
    # Run the test
    asyncio.run(test_format_adaptation())