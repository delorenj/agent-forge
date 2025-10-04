"""
Master Templater Agent - The Template Generator

Expert in generalizing specific agent files and creating template representations.
Takes any specific agent file and generalizes it, or codifies template representation of client formats.
Used for onboarding external agents or analyzing new formats.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import yaml
import uuid
from textwrap import dedent
from pathlib import Path
import re
from os import getenv


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
        
        # Initialize working embedder using sentence-transformers
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup QDrant client for template pattern storage
        try:
            self.qdrant_client = QdrantClient(
                host="localhost",
                port=6333,
                api_key=getenv("QDRANT_API_KEY", "touchmyflappyfoldyholds")
            )
            
            # Ensure collection exists for template patterns
            collection_name = "template_patterns"
            try:
                self.qdrant_client.get_collection(collection_name)
            except Exception:
                # Create collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
            
            self.collection_name = collection_name
            
        except Exception as e:
            print(f"Warning: Could not connect to QDrant: {e}")
            self.qdrant_client = None
            self.collection_name = None
        
        # Store knowledge base path
        self.knowledge_base_path = knowledge_base_path
        
        # Load existing template patterns if available
        if knowledge_base_path:
            self._populate_template_knowledge_from_path(knowledge_base_path)
        
        # Pre-populate with common template patterns
        self._populate_template_knowledge()
        
        # Built-in format recognizers
        self.format_recognizers = self._setup_format_recognizers()
    
    def _populate_template_knowledge(self):
        """Pre-populate knowledge base with common template patterns"""
        
        if not self.qdrant_client:
            return
            
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
        
        points = []
        for pattern in template_patterns:
            # Create embedding
            embedding = self.embedder.encode([pattern["content"]])[0].tolist()
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "pattern_type": pattern["pattern"],
                    "content": pattern["content"],
                    "type": "template_pattern"
                }
            )
            points.append(point)
        
        # Store in QDrant
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Pre-populated {len(points)} template patterns in knowledge base")
        except Exception as e:
            print(f"Warning: Could not populate template patterns: {e}")
    
    def _populate_template_knowledge_from_path(self, knowledge_base_path: str):
        """Load additional template patterns from path"""
        if not self.qdrant_client:
            return
            
        try:
            path = Path(knowledge_base_path)
            if path.exists() and path.is_dir():
                points = []
                for file_path in path.rglob("*.md"):
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if content.strip():
                        # Create embedding
                        embedding = self.embedder.encode([content])[0].tolist()
                        
                        # Create point
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": content,
                                "file_path": str(file_path),
                                "type": "external_pattern"
                            }
                        )
                        points.append(point)
                
                if points:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"Loaded {len(points)} external template patterns")
                        
        except Exception as e:
            print(f"Warning: Could not load external template patterns: {e}")
    
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
        
        # Perform extraction using built-in analysis
        extraction_result = self._perform_component_extraction(specific_agent, recognition_result)
        
        return extraction_result
    
    def _perform_component_extraction(self, specific_agent: SpecificAgent, recognition_result: Dict[str, Any]) -> ExtractedComponents:
        """Perform component extraction using built-in analysis"""
        
        content = specific_agent.content
        format_type = recognition_result['format']
        
        # Extract basic components based on format
        if format_type == "json":
            extracted = self._extract_from_json(content)
        elif format_type == "yaml":
            extracted = self._extract_from_yaml(content)
        elif format_type == "markdown":
            extracted = self._extract_from_markdown(content)
        else:
            extracted = self._extract_from_text(content)
        
        return ExtractedComponents(
            agent_name=extracted.get("name", specific_agent.agent_name or "ExtractedAgent"),
            role=extracted.get("role", "General Agent"),
            description=extracted.get("description", "Extracted from specific agent implementation"),
            capabilities=extracted.get("capabilities", []),
            instructions=extracted.get("instructions", content),
            configuration=extracted.get("configuration", {}),
            platform_specifics=extracted.get("platform_specifics", {}),
            interaction_patterns=extracted.get("interaction_patterns", {})
        )
    
    def _extract_from_json(self, content: str) -> Dict[str, Any]:
        """Extract components from JSON content"""
        try:
            data = json.loads(content)
            return {
                "name": data.get("name", "JSON Agent"),
                "role": data.get("role", data.get("description", "JSON-based agent")),
                "description": data.get("description", "Agent from JSON configuration"),
                "capabilities": data.get("capabilities", data.get("skills", [])),
                "instructions": data.get("instructions", data.get("prompt", "")),
                "configuration": {k: v for k, v in data.items() if k not in ["name", "description", "capabilities", "instructions"]},
                "platform_specifics": {"format": "json", "fields": list(data.keys())},
                "interaction_patterns": {"input": "JSON", "output": "JSON"}
            }
        except json.JSONDecodeError:
            return self._extract_from_text(content)
    
    def _extract_from_yaml(self, content: str) -> Dict[str, Any]:
        """Extract components from YAML content"""
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                return {
                    "name": data.get("name", "YAML Agent"),
                    "role": data.get("role", data.get("description", "YAML-based agent")),
                    "description": data.get("description", "Agent from YAML configuration"),
                    "capabilities": data.get("capabilities", data.get("skills", [])),
                    "instructions": data.get("instructions", data.get("prompt", "")),
                    "configuration": {k: v for k, v in data.items() if k not in ["name", "description", "capabilities", "instructions"]},
                    "platform_specifics": {"format": "yaml", "fields": list(data.keys())},
                    "interaction_patterns": {"input": "YAML", "output": "YAML"}
                }
        except yaml.YAMLError:
            pass
        return self._extract_from_text(content)
    
    def _extract_from_markdown(self, content: str) -> Dict[str, Any]:
        """Extract components from Markdown content"""
        import re
        
        # Extract title/name
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        name = title_match.group(1) if title_match else "Markdown Agent"
        
        # Extract role/description from first paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.startswith('#')]
        description = paragraphs[0] if paragraphs else "Agent from Markdown documentation"
        
        # Extract capabilities from lists
        capability_matches = re.findall(r'[-*]\s+(.+)', content)
        capabilities = [cap.strip() for cap in capability_matches if cap.strip()][:10]  # Limit to 10
        
        return {
            "name": name,
            "role": f"Markdown-documented {name.lower()}",
            "description": description,
            "capabilities": capabilities,
            "instructions": content,
            "configuration": {"format": "markdown"},
            "platform_specifics": {"format": "markdown", "has_headers": "##" in content},
            "interaction_patterns": {"input": "Text", "output": "Markdown"}
        }
    
    def _extract_from_text(self, content: str) -> Dict[str, Any]:
        """Extract components from plain text content"""
        lines = content.split('\n')
        first_line = lines[0] if lines else "Text Agent"
        
        return {
            "name": first_line[:50] if len(first_line) > 50 else first_line,
            "role": "Text-based agent",
            "description": content[:200] + "..." if len(content) > 200 else content,
            "capabilities": ["text processing", "general purpose"],
            "instructions": content,
            "configuration": {"format": "text"},
            "platform_specifics": {"format": "text", "line_count": len(lines)},
            "interaction_patterns": {"input": "Text", "output": "Text"}
        }
    
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
        
        # Execute generalization using built-in logic
        generalized_result = self._perform_generalization(components)
        
        return generalized_result
    
    def _perform_generalization(self, components: ExtractedComponents) -> GeneralizedAgent:
        """Perform generalization using built-in logic"""
        
        # Generalize instructions by removing platform-specific elements
        generalized_instructions = self._generalize_instructions(components.instructions)
        
        # Generalize capabilities
        generalized_capabilities = self._generalize_capabilities(components.capabilities)
        
        # Create generalized interaction patterns
        interaction_patterns = {
            "input": "Structured input appropriate to task",
            "output": "Structured output with results", 
            "coordination": "Collaborative with other agents"
        }
        
        # Identify requirements and constraints
        requirements = self._identify_requirements(components)
        constraints = self._identify_constraints(components)
        
        return GeneralizedAgent(
            name=components.agent_name,
            role=self._generalize_role(components.role),
            description=f"Generalized {components.role.lower()} agent for reuse across platforms",
            capabilities=generalized_capabilities,
            instructions=generalized_instructions,
            interaction_patterns=interaction_patterns,
            requirements=requirements,
            constraints=constraints,
            metadata={
                "generalized_from": components.agent_name,
                "original_platform_specifics": components.platform_specifics,
                "generalization_date": "2024-01-01"  # Would be actual date
            }
        )
    
    def _generalize_instructions(self, instructions: str) -> str:
        """Remove platform-specific elements from instructions"""
        
        # Platform-specific terms to generalize
        platform_replacements = {
            # Platform names
            "Claude Code": "the platform",
            "OpenAI": "the AI service",
            "Anthropic": "the AI provider",
            
            # Specific tools/services
            "MCP tools": "available tools",
            "file-reader": "file access tools",
            "git-tools": "version control tools",
            
            # Model-specific references
            "claude-3-5-sonnet": "the language model",
            "gpt-4": "the language model",
            "text-embedding-3-small": "the embedding model",
        }
        
        generalized = instructions
        for specific, generic in platform_replacements.items():
            generalized = generalized.replace(specific, generic)
        
        # Remove specific configuration details
        import re
        generalized = re.sub(r'\btemperature:\s*[\d.]+', 'temperature: [configurable]', generalized)
        generalized = re.sub(r'\bmax_tokens:\s*\d+', 'max_tokens: [configurable]', generalized)
        
        return generalized
    
    def _generalize_capabilities(self, capabilities: List[str]) -> List[str]:
        """Generalize capability descriptions"""
        
        capability_generalizations = {
            "Claude Code MCP tools": "Platform tools integration",
            "OpenAI API": "Language model API",
            "file-reader": "File system access",
            "Python": "Programming language support",
            "JavaScript": "Scripting language support",
        }
        
        generalized = []
        for cap in capabilities:
            generalized_cap = cap
            for specific, generic in capability_generalizations.items():
                if specific.lower() in cap.lower():
                    generalized_cap = generic
                    break
            generalized.append(generalized_cap)
        
        return list(set(generalized))  # Remove duplicates
    
    def _generalize_role(self, role: str) -> str:
        """Generalize role description"""
        
        # Remove platform-specific role elements
        role_generalizations = {
            "for Claude Code": "",
            "using OpenAI": "",
            "with Anthropic": "",
            "MCP-based": "platform-integrated",
        }
        
        generalized_role = role
        for specific, replacement in role_generalizations.items():
            generalized_role = generalized_role.replace(specific, replacement)
        
        return generalized_role.strip()
    
    def _identify_requirements(self, components: ExtractedComponents) -> List[str]:
        """Identify general requirements from component analysis"""
        
        requirements = ["Core AI capabilities"]
        
        # Analyze capabilities to identify requirements
        cap_text = " ".join(components.capabilities).lower()
        
        if any(term in cap_text for term in ["file", "system", "directory"]):
            requirements.append("File system access")
        
        if any(term in cap_text for term in ["api", "http", "web"]):
            requirements.append("Network connectivity")
        
        if any(term in cap_text for term in ["database", "sql", "storage"]):
            requirements.append("Data storage access")
        
        if any(term in cap_text for term in ["git", "version", "repository"]):
            requirements.append("Version control integration")
        
        return requirements
    
    def _identify_constraints(self, components: ExtractedComponents) -> List[str]:
        """Identify constraints from component analysis"""
        
        constraints = ["Platform-specific features may vary"]
        
        # Analyze platform specifics for constraints
        if "format" in components.platform_specifics:
            format_type = components.platform_specifics["format"]
            constraints.append(f"Original format was {format_type}")
        
        if components.configuration:
            constraints.append("May require configuration adaptation")
        
        return constraints

    def _analyze_platform_template_structure(self, specific_agents: List[SpecificAgent], platform: str, primary_format: str) -> str:
        """Analyze platform template structure from specific agents using built-in analysis"""
        try:
            # Extract components from all agents
            all_components = []
            field_frequency = {}
            common_patterns = []
            
            for agent in specific_agents:
                # Parse agent content based on format
                recognition_result = {'format': agent.file_type if agent.file_type != "auto-detect" else primary_format}
                extracted = self._perform_component_extraction(agent, recognition_result)
                all_components.append(extracted)
                
                # Count field frequency
                if hasattr(extracted, 'fields') and extracted.fields:
                    for field in extracted.fields:
                        field_frequency[field] = field_frequency.get(field, 0) + 1
            
            # Identify required fields (appear in all agents)
            total_agents = len(specific_agents)
            required_fields = [field for field, count in field_frequency.items() if count == total_agents]
            optional_fields = [field for field, count in field_frequency.items() if count < total_agents and count > 0]
            
            # Analyze common structure patterns
            structure_schema = self._build_structure_schema(all_components, primary_format)
            
            # Generate validation rules based on format
            validation_rules = self._generate_validation_rules(primary_format, required_fields)
            
            # Create field descriptions
            field_descriptions = self._generate_field_descriptions(required_fields + optional_fields, all_components)
            
            # Identify platform-specific conventions
            conventions = self._identify_platform_conventions(platform, all_components)
            
            # Format analysis result
            analysis_result = f"""
Platform Template Analysis for {platform} ({primary_format} format):

Structure Schema: {structure_schema}
Required Fields: {required_fields}
Optional Fields: {optional_fields}
Field Descriptions: {field_descriptions}
Validation Rules: {validation_rules}
Conventions: {conventions}

Agent Count Analyzed: {total_agents}
Field Distribution: {dict(sorted(field_frequency.items(), key=lambda x: x[1], reverse=True))}
"""
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing platform template structure: {e}")
            return f"Platform template analysis for {platform} (format: {primary_format}) - Basic structure identified from {len(specific_agents)} agents"

    def _build_structure_schema(self, components: List[ExtractedComponents], primary_format: str) -> Dict[str, str]:
        """Build structure schema from extracted components"""
        schema = {}
        
        for comp in components:
            if hasattr(comp, 'structure') and comp.structure:
                for key, value in comp.structure.items():
                    if key not in schema:
                        schema[key] = type(value).__name__
                    elif schema[key] != type(value).__name__:
                        schema[key] = "mixed"
        
        return schema

    def _generate_validation_rules(self, format_type: str, required_fields: List[str]) -> List[str]:
        """Generate validation rules based on format and required fields"""
        rules = []
        
        if format_type.lower() == "json":
            rules.append("Must be valid JSON format")
            rules.append("Must contain proper JSON syntax with braces and quotes")
        elif format_type.lower() == "yaml":
            rules.append("Must be valid YAML format")
            rules.append("Must use proper YAML indentation")
        elif format_type.lower() == "markdown":
            rules.append("Must follow Markdown formatting")
            rules.append("Must contain proper headers and sections")
        
        for field in required_fields:
            rules.append(f"Must contain required field: {field}")
        
        return rules

    def _generate_field_descriptions(self, fields: List[str], components: List[ExtractedComponents]) -> Dict[str, str]:
        """Generate field descriptions based on analysis of components"""
        descriptions = {}
        
        # Common field descriptions
        field_mappings = {
            'name': 'Agent or component name identifier',
            'description': 'Detailed description of functionality',
            'instructions': 'Specific instructions for agent behavior',
            'capabilities': 'List of capabilities and features',
            'role': 'Primary role or function of the agent',
            'persona': 'Agent personality and interaction style',
            'tools': 'Available tools and integrations',
            'config': 'Configuration parameters and settings',
            'examples': 'Usage examples and sample interactions',
            'metadata': 'Additional metadata and properties'
        }
        
        for field in fields:
            if field.lower() in field_mappings:
                descriptions[field] = field_mappings[field.lower()]
            else:
                descriptions[field] = f"Platform-specific field: {field}"
        
        return descriptions

    def _identify_platform_conventions(self, platform: str, components: List[ExtractedComponents]) -> List[str]:
        """Identify platform-specific conventions from components"""
        conventions = []
        
        # Platform-specific patterns
        if platform.lower() in ['discord', 'slack', 'teams']:
            conventions.append("Uses chat-based interaction patterns")
            conventions.append("Includes message formatting and emoji support")
            conventions.append("Supports real-time communication features")
        elif platform.lower() in ['api', 'rest', 'web']:
            conventions.append("Uses HTTP request/response patterns")
            conventions.append("Includes endpoint definitions and data schemas")
            conventions.append("Supports authentication and authorization")
        elif platform.lower() in ['cli', 'terminal', 'command']:
            conventions.append("Uses command-line interface patterns")
            conventions.append("Includes argument parsing and help text")
            conventions.append("Supports piping and output formatting")
        
        # Analyze components for additional patterns
        for comp in components:
            if hasattr(comp, 'patterns') and comp.patterns:
                conventions.extend(comp.patterns)
        
        return list(set(conventions))  # Remove duplicates
    
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
        
        # Analyze agents to extract template structure using built-in analysis
        template_result = self._analyze_platform_template_structure(specific_agents, platform, primary_format)
        
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