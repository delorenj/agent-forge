"""
Agent Developer - The Creator

A master prompt engineer for AgentForge that creates new specialized agents when the
Talent Scout identifies capability gaps. This agent crafts precise, effective, and
robust agent definitions based on specifications in the Strategy Document.

Capabilities:
- Master prompt engineering for creating new agents
- Understanding of agent format standards and templates  
- Use Agno reasoning patterns for systematic agent creation
- Create robust, effective agent definitions that follow best practices
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools.knowledge import KnowledgeTools
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from os import getenv
import json
from datetime import datetime
from textwrap import dedent, indent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VacantRole(BaseModel):
    """A role that needs to be filled with a new agent"""
    role_name: str = Field(..., description="Name of the vacant role")
    title: str = Field(..., description="Professional title/designation")
    core_responsibilities: List[str] = Field(..., description="Primary responsibilities")
    required_capabilities: List[str] = Field(..., description="Essential skills and capabilities")
    interaction_patterns: Dict[str, str] = Field(..., description="How this role interacts with others")
    success_metrics: List[str] = Field(..., description="How success is measured for this role")
    priority_level: str = Field(..., description="Critical, High, Medium, or Low priority")
    domain_context: Optional[str] = Field(None, description="Domain-specific context")
    complexity_level: str = Field(default="medium", description="simple, medium, complex, or enterprise")


class ScoutingReport(BaseModel):
    """Scouting report from Talent Scout identifying gaps and matches"""
    matched_agents: List[Dict[str, Any]] = Field(..., description="Existing agents that match roles")
    vacant_roles: List[VacantRole] = Field(..., description="Roles that need new agents")
    capability_gaps: List[str] = Field(..., description="Missing capabilities across the team")
    reuse_analysis: Dict[str, Any] = Field(..., description="Analysis of reuse opportunities")
    priority_recommendations: List[str] = Field(..., description="Priority order for agent creation")


class AgentSpecification(BaseModel):
    """Complete specification for a newly created agent"""
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Primary role/responsibility")
    title: str = Field(..., description="Professional title")
    
    # Agent definition components
    system_prompt: str = Field(..., description="Complete system prompt for the agent")
    instructions: List[str] = Field(..., description="Specific instructions for behavior")
    tools_required: List[str] = Field(..., description="Required tools and capabilities")
    model_recommendations: Dict[str, str] = Field(..., description="Recommended models for different scenarios")
    
    # Behavioral specifications
    communication_style: str = Field(..., description="How the agent communicates")
    decision_making_approach: str = Field(..., description="How the agent makes decisions")
    collaboration_patterns: Dict[str, str] = Field(..., description="How it collaborates with other agents")
    
    # Quality specifications
    success_criteria: List[str] = Field(..., description="How to measure agent effectiveness")
    failure_modes: List[str] = Field(..., description="Common failure patterns to avoid")
    quality_checks: List[str] = Field(..., description="Validation checks for agent outputs")
    
    # Implementation details
    initialization_code: str = Field(..., description="Python code to initialize the agent")
    example_usage: str = Field(..., description="Example usage code")
    test_cases: List[Dict[str, str]] = Field(..., description="Test cases for validation")
    
    # Metadata
    created_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field(default="1.0.0")
    format_compliance: str = Field(default="agno_standard", description="Agent format standard")


class AgentGenerationResult(BaseModel):
    """Result of agent generation process"""
    success: bool = Field(..., description="Whether generation was successful")
    agents_created: List[AgentSpecification] = Field(..., description="Successfully created agents")
    generation_summary: str = Field(..., description="Summary of the generation process")
    recommendations: List[str] = Field(..., description="Recommendations for using the agents")
    
    # Files and artifacts
    generated_files: Dict[str, str] = Field(..., description="Generated agent files with paths")
    documentation: str = Field(..., description="Usage documentation for the new agents")
    integration_notes: str = Field(..., description="Notes for integrating with existing system")
    
    # Quality assessment
    quality_score: float = Field(..., description="Overall quality score (0-1)")
    validation_results: List[Dict[str, Any]] = Field(..., description="Validation test results")
    
    # Process metadata
    processing_time: float = Field(..., description="Time taken for generation in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed during generation")


class AgentDeveloper:
    """
    The Agent Developer - Master prompt engineer for creating new agents.
    
    Analyzes vacant roles from Scouting Report and creates comprehensive, well-structured
    agent definitions that follow Agno framework best practices and internal format standards.
    """
    
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        knowledge_base_path: Optional[str] = None,
        agent_library_path: Optional[str] = None
    ):
        """Initialize the Agent Developer with required components."""
        
        # Setup knowledge base for agent patterns and best practices
        self.knowledge_base = Knowledge(
            vector_db=LanceDb(
                uri="tmp/agent_developer_knowledge",
                table_name="agent_patterns",
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        )
        
        # Load existing agent library for pattern analysis
        self.agent_library_path = agent_library_path or "/home/delorenj/code/DeLoDocs/AI/Agents"
        
        # Load local documentation and patterns
        if knowledge_base_path:
            self.knowledge_base.add_content_from_path(knowledge_base_path)
        
        # Load local Agno docs for reference
        try:
            self.knowledge_base.add_content_from_path("docs/agno/")
        except Exception as e:
            logger.warning(f"Could not load local Agno docs: {e}")
        
        # Create the main agent with reasoning and knowledge tools
        self.agent = Agent(
            name="AgentDeveloper",
            model=OpenRouter(id=model_id),
            tools=[
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True,
                    add_few_shot=True,
                ),
                KnowledgeTools(
                    knowledge=self.knowledge_base,
                    think=True,
                    search=True,
                    analyze=True,
                    add_few_shot=True,
                ),
            ],
            instructions=dedent("""\
                You are the Agent Developer - The Creator for AgentForge.
                
                Your expertise:
                - Master prompt engineering for creating new AI agents
                - Deep understanding of agent architecture and system prompts
                - Expert in Agno framework patterns and best practices
                - Systematic approach to agent creation using reasoning tools
                
                Your mission:
                When the Talent Scout identifies capability gaps, you create comprehensive,
                well-structured agent definitions that are:
                - Precisely targeted to fill the identified gaps
                - Robust and effective in their domain
                - Compliant with Agno framework standards
                - Well-documented and easily deployable
                
                Your systematic approach:
                1. ANALYZE vacant roles deeply using reasoning tools
                2. RESEARCH existing patterns using knowledge tools
                3. DESIGN comprehensive system prompts and specifications
                4. VALIDATE agent definitions against requirements
                5. GENERATE complete implementation code and documentation
                6. TEST agent specifications with realistic scenarios
                
                Key principles for agent creation:
                - Each agent should have a clear, focused purpose
                - System prompts should be comprehensive but not overwhelming
                - Include specific behavior patterns and decision-making frameworks  
                - Define clear success metrics and quality checks
                - Ensure agents can collaborate effectively with others
                - Follow the "single responsibility principle" for agent design
                - Include proper error handling and failure mode detection
                
                Always use reasoning tools to work through complex agent design decisions.
                Use knowledge tools to reference best practices and existing patterns.
                Create agents that are production-ready and enterprise-grade.
            """),
            markdown=True,
            add_history_to_context=True,
        )
    
    async def develop_agents(self, scouting_report: ScoutingReport, strategy_context: Optional[Dict[str, Any]] = None) -> AgentGenerationResult:
        """
        Main method to develop new agents based on scouting report.
        
        Args:
            scouting_report: Report identifying vacant roles and capability gaps
            strategy_context: Additional context from strategy document
            
        Returns:
            AgentGenerationResult: Complete results of agent generation process
        """
        start_time = datetime.now()
        logger.info(f"Starting agent development for {len(scouting_report.vacant_roles)} vacant roles")
        
        try:
            # Step 1: Analyze vacant roles and prioritize development
            logger.info("Step 1: Analyzing vacant roles and prioritizing development...")
            analysis_result = await self._analyze_vacant_roles(scouting_report, strategy_context)
            
            # Step 2: Design agents systematically
            logger.info("Step 2: Designing agents with systematic approach...")
            agent_specifications = []
            
            for role in scouting_report.vacant_roles:
                logger.info(f"Creating agent for role: {role.role_name}")
                agent_spec = await self._design_agent_for_role(role, scouting_report, strategy_context)
                agent_specifications.append(agent_spec)
            
            # Step 3: Validate and optimize agent definitions
            logger.info("Step 3: Validating and optimizing agent definitions...")
            validated_agents = await self._validate_agent_specifications(agent_specifications)
            
            # Step 4: Generate implementation files and documentation
            logger.info("Step 4: Generating implementation files and documentation...")
            generated_files = await self._generate_agent_files(validated_agents)
            
            # Step 5: Create comprehensive documentation
            logger.info("Step 5: Creating comprehensive documentation...")
            documentation = await self._generate_documentation(validated_agents, scouting_report)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create final result
            result = AgentGenerationResult(
                success=True,
                agents_created=validated_agents,
                generation_summary=f"Successfully created {len(validated_agents)} new agents to fill capability gaps",
                recommendations=await self._generate_recommendations(validated_agents, scouting_report),
                generated_files=generated_files,
                documentation=documentation,
                integration_notes=await self._generate_integration_notes(validated_agents),
                quality_score=await self._calculate_quality_score(validated_agents),
                validation_results=await self._run_validation_tests(validated_agents),
                processing_time=processing_time
            )
            
            logger.info(f"Agent development completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Agent development failed: {str(e)}")
            
            return AgentGenerationResult(
                success=False,
                agents_created=[],
                generation_summary=f"Agent development failed: {str(e)}",
                recommendations=["Review error logs and retry with adjusted parameters"],
                generated_files={},
                documentation="",
                integration_notes="",
                quality_score=0.0,
                validation_results=[],
                processing_time=processing_time
            )
    
    async def _analyze_vacant_roles(self, scouting_report: ScoutingReport, strategy_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze vacant roles and create development strategy."""
        
        analysis_prompt = dedent(f"""\
            Analyze these vacant roles that need new agents created:
            
            VACANT ROLES:
            {json.dumps([role.model_dump() for role in scouting_report.vacant_roles], indent=2)}
            
            CAPABILITY GAPS:
            {json.dumps(scouting_report.capability_gaps, indent=2)}
            
            STRATEGY CONTEXT:
            {json.dumps(strategy_context or {}, indent=2)}
            
            Using reasoning tools, analyze:
            1. The complexity and priority of each role
            2. Interdependencies between roles
            3. Optimal creation order considering dependencies
            4. Common patterns and capabilities that can be reused
            5. Specific challenges for each role type
            
            Create a development strategy that maximizes effectiveness while minimizing redundancy.
        """)
        
        response = await self.agent.arun(analysis_prompt)
        return {"analysis": response}
    
    async def _design_agent_for_role(self, role: VacantRole, scouting_report: ScoutingReport, strategy_context: Optional[Dict[str, Any]] = None) -> AgentSpecification:
        """Design a comprehensive agent specification for a specific role."""
        
        design_prompt = dedent(f"""\
            Design a comprehensive agent specification for this vacant role:
            
            ROLE DETAILS:
            - Name: {role.role_name}
            - Title: {role.title}  
            - Responsibilities: {json.dumps(role.core_responsibilities, indent=2)}
            - Required Capabilities: {json.dumps(role.required_capabilities, indent=2)}
            - Success Metrics: {json.dumps(role.success_metrics, indent=2)}
            - Priority Level: {role.priority_level}
            - Domain Context: {role.domain_context}
            - Complexity: {role.complexity_level}
            
            INTERACTION PATTERNS:
            {json.dumps(role.interaction_patterns, indent=2)}
            
            Using reasoning and knowledge tools, create a complete agent specification that includes:
            
            1. **System Prompt**: Comprehensive prompt that defines the agent's identity, expertise, and approach
            2. **Instructions**: Specific behavioral instructions and guidelines
            3. **Tools Required**: List of tools and capabilities the agent needs
            4. **Communication Style**: How the agent should communicate and collaborate
            5. **Decision Making**: Framework for how the agent makes decisions
            6. **Quality Measures**: Success criteria and validation checks
            
            Research existing agent patterns in the knowledge base for inspiration.
            Think through the design systematically using reasoning tools.
            
            The agent should be:
            - Focused and specialized for its role
            - Able to collaborate effectively with other agents  
            - Robust and production-ready
            - Compliant with Agno framework standards
        """)
        
        response = await self.agent.arun(design_prompt)
        
        # Parse the response into structured format
        # In production, this would have more sophisticated parsing
        agent_name = role.role_name.replace(" ", "").replace("-", "")
        
        return AgentSpecification(
            name=agent_name,
            role=role.role_name,
            title=role.title,
            system_prompt=self._extract_system_prompt(response, role),
            instructions=self._extract_instructions(response, role),
            tools_required=self._extract_tools_required(response, role),
            model_recommendations={"primary": "deepseek/deepseek-v3.1", "reasoning": "anthropic/claude-3-5-sonnet"},
            communication_style=self._extract_communication_style(response, role),
            decision_making_approach=self._extract_decision_making(response, role),
            collaboration_patterns=role.interaction_patterns,
            success_criteria=role.success_metrics,
            failure_modes=self._generate_failure_modes(role),
            quality_checks=self._generate_quality_checks(role),
            initialization_code=self._generate_initialization_code(agent_name, role),
            example_usage=self._generate_example_usage(agent_name, role),
            test_cases=self._generate_test_cases(role)
        )
    
    def _extract_system_prompt(self, response: str, role: VacantRole) -> str:
        """Extract or generate system prompt from agent design response."""
        
        # This is a template-based approach - in production, would parse from LLM response
        return dedent(f"""\
            You are the {role.title} for AgentForge, specializing in {role.role_name.lower()}.
            
            Your core expertise:
            {chr(10).join(f'- {resp}' for resp in role.core_responsibilities)}
            
            Your essential capabilities:  
            {chr(10).join(f'- {cap}' for cap in role.required_capabilities)}
            
            Your approach to work:
            1. Use systematic reasoning to break down complex problems
            2. Leverage your specialized knowledge and tools effectively
            3. Collaborate seamlessly with other agents in the team
            4. Maintain high quality standards in all deliverables
            5. Continuously validate your outputs against success criteria
            
            Success is measured by:
            {chr(10).join(f'- {metric}' for metric in role.success_metrics)}
            
            Domain context: {role.domain_context or 'General purpose'}
            Priority level: {role.priority_level}
            
            Always use reasoning tools to think through complex decisions.
            Use knowledge tools to access relevant information and patterns.
            Maintain clear communication and provide structured outputs.
        """)
    
    def _extract_instructions(self, response: str, role: VacantRole) -> List[str]:
        """Extract behavioral instructions from agent design response."""
        
        base_instructions = [
            f"Focus on {role.role_name.lower()} as your primary responsibility",
            "Use reasoning tools to analyze complex problems step by step",
            "Maintain high quality standards in all work products",
            "Collaborate effectively with other team members",
            "Document your decisions and reasoning process"
        ]
        
        # Add domain-specific instructions based on role
        if "analyst" in role.role_name.lower():
            base_instructions.extend([
                "Provide data-driven insights and recommendations",
                "Use structured analysis frameworks",
                "Validate assumptions with evidence"
            ])
        elif "developer" in role.role_name.lower() or "coder" in role.role_name.lower():
            base_instructions.extend([
                "Follow software engineering best practices",
                "Write clean, maintainable code",
                "Include comprehensive testing"
            ])
        elif "architect" in role.role_name.lower():
            base_instructions.extend([
                "Think systemically about design decisions",
                "Consider scalability and maintainability",
                "Document architectural decisions and rationale"
            ])
        
        return base_instructions
    
    def _extract_tools_required(self, response: str, role: VacantRole) -> List[str]:
        """Determine required tools based on role capabilities."""
        
        tools = ["ReasoningTools"]  # All agents get reasoning tools
        
        # Add tools based on capabilities
        for capability in role.required_capabilities:
            if any(keyword in capability.lower() for keyword in ["research", "knowledge", "information"]):
                tools.append("KnowledgeTools")
            if any(keyword in capability.lower() for keyword in ["web", "search", "internet"]):
                tools.append("WebSearchTools") 
            if any(keyword in capability.lower() for keyword in ["code", "programming", "development"]):
                tools.append("CodeTools")
            if any(keyword in capability.lower() for keyword in ["file", "document", "content"]):
                tools.append("FileTools")
            if any(keyword in capability.lower() for keyword in ["data", "analysis", "analytics"]):
                tools.append("DataAnalysisTools")
        
        return list(set(tools))  # Remove duplicates
    
    def _extract_communication_style(self, response: str, role: VacantRole) -> str:
        """Define communication style based on role characteristics."""
        
        if "analyst" in role.role_name.lower():
            return "Data-driven and analytical, using structured frameworks and clear metrics"
        elif "developer" in role.role_name.lower():
            return "Technical and precise, focusing on implementation details and best practices"
        elif "architect" in role.role_name.lower():
            return "Strategic and systematic, emphasizing design decisions and long-term implications"
        elif "coordinator" in role.role_name.lower() or "manager" in role.role_name.lower():
            return "Clear and directive, focusing on coordination and process management"
        else:
            return "Professional and collaborative, adapting style to the audience and context"
    
    def _extract_decision_making(self, response: str, role: VacantRole) -> str:
        """Define decision-making approach based on role characteristics."""
        
        base_approach = "Use reasoning tools to systematically analyze options, consider trade-offs, and make evidence-based decisions"
        
        if role.priority_level.lower() == "critical":
            return f"{base_approach}. Prioritize speed and reliability for critical decisions."
        elif "complex" in role.complexity_level.lower():
            return f"{base_approach}. Take time for thorough analysis of complex interdependencies."
        else:
            return f"{base_approach}. Balance thoroughness with efficiency based on context."
    
    def _generate_failure_modes(self, role: VacantRole) -> List[str]:
        """Generate common failure modes for this type of role."""
        
        common_failures = [
            "Producing outputs that don't meet quality standards",
            "Failing to collaborate effectively with other agents",
            "Not using available tools appropriately",
            "Losing focus on primary responsibilities"
        ]
        
        # Add role-specific failure modes
        if "analyst" in role.role_name.lower():
            common_failures.extend([
                "Making recommendations without sufficient data",
                "Bias in analysis or interpretation",
                "Overlooking important edge cases"
            ])
        elif "developer" in role.role_name.lower():
            common_failures.extend([
                "Writing code that doesn't meet requirements",
                "Insufficient error handling or testing",
                "Poor code organization or documentation"
            ])
        elif "architect" in role.role_name.lower():
            common_failures.extend([
                "Designs that are too complex or overengineered",
                "Ignoring non-functional requirements",
                "Poor consideration of future scalability"
            ])
        
        return common_failures
    
    def _generate_quality_checks(self, role: VacantRole) -> List[str]:
        """Generate quality validation checks for this role."""
        
        return [
            "Verify outputs meet all specified success criteria",
            "Check that reasoning process is documented and sound",
            "Ensure collaboration protocols are followed",
            "Validate that all required capabilities are demonstrated",
            f"Confirm deliverables align with {role.role_name} responsibilities"
        ]
    
    def _generate_initialization_code(self, agent_name: str, role: VacantRole) -> str:
        """Generate Python initialization code for the agent."""
        
        tools_list = ", ".join(f"{tool}()" for tool in self._extract_tools_required("", role))
        
        return dedent(f'''\
            from agno.agent import Agent
            from agno.models.openrouter import OpenRouter
            from agno.tools.reasoning import ReasoningTools
            from agno.tools.knowledge import KnowledgeTools
            from textwrap import dedent
            
            def create_{agent_name.lower()}_agent(model_id="deepseek/deepseek-v3.1"):
                """Create and configure the {role.title} agent."""
                
                agent = Agent(
                    name="{agent_name}",
                    model=OpenRouter(id=model_id),
                    tools=[
                        {tools_list}
                    ],
                    instructions=dedent(""\\
                        {self._extract_system_prompt("", role)}
                    ""),
                    markdown=True,
                    add_history_to_context=True,
                )
                
                return agent
            
            # Example usage
            if __name__ == "__main__":
                agent = create_{agent_name.lower()}_agent()
                # Use the agent for {role.role_name.lower()}
        ''')
    
    def _generate_example_usage(self, agent_name: str, role: VacantRole) -> str:
        """Generate example usage code for the agent."""
        
        return dedent(f'''\
            import asyncio
            from {agent_name.lower()}_agent import create_{agent_name.lower()}_agent
            
            async def example_usage():
                """Example usage of the {role.title} agent."""
                
                agent = create_{agent_name.lower()}_agent()
                
                # Example task for {role.role_name}
                task = "Example task related to {role.role_name.lower()}"
                
                result = await agent.arun(task)
                print(f"Result: {{result}}")
                
                return result
            
            if __name__ == "__main__":
                asyncio.run(example_usage())
        ''')
    
    def _generate_test_cases(self, role: VacantRole) -> List[Dict[str, str]]:
        """Generate test cases for validating the agent."""
        
        return [
            {
                "name": "basic_functionality_test",
                "description": f"Test basic {role.role_name.lower()} functionality",
                "input": f"Simple task requiring {role.required_capabilities[0] if role.required_capabilities else 'core capability'}",
                "expected_behavior": "Agent should successfully complete the task using appropriate tools and reasoning"
            },
            {
                "name": "collaboration_test", 
                "description": "Test collaboration with other agents",
                "input": "Task requiring coordination with other team members",
                "expected_behavior": "Agent should demonstrate proper collaboration patterns and communication"
            },
            {
                "name": "quality_validation_test",
                "description": "Test quality of outputs against success criteria", 
                "input": "Complex task with specific quality requirements",
                "expected_behavior": f"Agent output should meet all success criteria: {', '.join(role.success_metrics)}"
            }
        ]
    
    async def _validate_agent_specifications(self, agent_specs: List[AgentSpecification]) -> List[AgentSpecification]:
        """Validate and optimize agent specifications."""
        
        validated_specs = []
        
        for spec in agent_specs:
            logger.info(f"Validating agent specification: {spec.name}")
            
            # Basic validation checks
            validation_issues = []
            
            if len(spec.system_prompt) < 100:
                validation_issues.append("System prompt too short")
            if len(spec.instructions) < 3:
                validation_issues.append("Insufficient instructions")
            if not spec.tools_required:
                validation_issues.append("No tools specified")
            
            if validation_issues:
                logger.warning(f"Validation issues for {spec.name}: {validation_issues}")
                # In production, would fix these issues automatically
            
            validated_specs.append(spec)
        
        return validated_specs
    
    async def _generate_agent_files(self, agent_specs: List[AgentSpecification]) -> Dict[str, str]:
        """Generate actual agent implementation files."""
        
        generated_files = {}
        
        for spec in agent_specs:
            # Generate main agent file
            agent_filename = f"agents/{spec.name.lower()}_agent.py"
            agent_content = dedent(f'''\
                """
                {spec.title} Agent - {spec.role}
                
                Generated by AgentForge Agent Developer
                Creation time: {spec.created_timestamp}
                Version: {spec.version}
                """
                
                {spec.initialization_code}
            ''')
            
            generated_files[agent_filename] = agent_content
            
            # Generate test file
            test_filename = f"tests/test_{spec.name.lower()}_agent.py"
            test_content = self._generate_test_file_content(spec)
            generated_files[test_filename] = test_content
            
            # Generate documentation file
            doc_filename = f"docs/{spec.name.lower()}_agent.md"
            doc_content = self._generate_agent_documentation(spec)
            generated_files[doc_filename] = doc_content
        
        return generated_files
    
    def _generate_test_file_content(self, spec: AgentSpecification) -> str:
        """Generate test file content for an agent specification."""
        
        test_methods = []
        
        for i, test_case in enumerate(spec.test_cases):
            method_name = f"test_{test_case['name']}"
            test_methods.append(dedent(f'''\
                async def {method_name}(self):
                    """
                    {test_case['description']}
                    
                    Expected: {test_case['expected_behavior']}
                    """
                    result = await self.agent.arun("{test_case['input']}")
                    
                    # Basic validation
                    self.assertIsNotNone(result)
                    self.assertGreater(len(result), 10)
                    
                    # Add specific validation logic here
                    return result
            '''))
        
        return dedent(f'''\
            """
            Test cases for {spec.title} Agent
            
            Generated by AgentForge Agent Developer
            """
            
            import asyncio
            import unittest
            from {spec.name.lower()}_agent import create_{spec.name.lower()}_agent
            
            class Test{spec.name}Agent(unittest.TestCase):
                """Test cases for {spec.name} agent."""
                
                @classmethod
                def setUpClass(cls):
                    """Set up test fixtures."""
                    cls.agent = create_{spec.name.lower()}_agent()
                
                {chr(10).join(indent(method, "    ") for method in test_methods)}
            
            if __name__ == "__main__":
                unittest.main()
        ''')
    
    def _generate_agent_documentation(self, spec: AgentSpecification) -> str:
        """Generate comprehensive documentation for an agent."""
        
        return dedent(f'''\
            # {spec.title} Agent
            
            **Role:** {spec.role}  
            **Version:** {spec.version}  
            **Created:** {spec.created_timestamp}
            
            ## Overview
            
            {spec.system_prompt}
            
            ## Capabilities
            
            - **Tools Required:** {", ".join(spec.tools_required)}
            - **Communication Style:** {spec.communication_style}
            - **Decision Making:** {spec.decision_making_approach}
            
            ## Usage
            
            ```python
            {spec.example_usage}
            ```
            
            ## Success Criteria
            
            {chr(10).join(f"- {criterion}" for criterion in spec.success_criteria)}
            
            ## Quality Checks
            
            {chr(10).join(f"- {check}" for check in spec.quality_checks)}
            
            ## Collaboration Patterns
            
            {chr(10).join(f"- **{role}:** {pattern}" for role, pattern in spec.collaboration_patterns.items())}
            
            ## Common Failure Modes
            
            {chr(10).join(f"- {mode}" for mode in spec.failure_modes)}
            
            ## Model Recommendations
            
            {chr(10).join(f"- **{use_case}:** {model}" for use_case, model in spec.model_recommendations.items())}
        ''')
    
    async def _generate_documentation(self, agent_specs: List[AgentSpecification], scouting_report: ScoutingReport) -> str:
        """Generate comprehensive documentation for all created agents."""
        
        doc_prompt = dedent(f"""\
            Generate comprehensive documentation for the newly created agents:
            
            CREATED AGENTS:
            {json.dumps([{"name": spec.name, "role": spec.role, "title": spec.title} for spec in agent_specs], indent=2)}
            
            ORIGINAL CAPABILITY GAPS:
            {json.dumps(scouting_report.capability_gaps, indent=2)}
            
            Create documentation that includes:
            1. Overview of how these agents fill the identified gaps
            2. Team integration guidelines
            3. Usage examples and best practices
            4. Deployment instructions
            5. Monitoring and maintenance guidelines
        """)
        
        return await self.agent.arun(doc_prompt)
    
    async def _generate_recommendations(self, agent_specs: List[AgentSpecification], scouting_report: ScoutingReport) -> List[str]:
        """Generate recommendations for using the created agents."""
        
        return [
            f"Deploy agents in priority order: {', '.join(spec.name for spec in sorted(agent_specs, key=lambda x: x.role))}",
            "Test each agent individually before team integration",
            "Monitor agent performance against defined success criteria",
            "Establish feedback loops for continuous improvement",
            f"Consider creating additional agents if capability gaps remain after deployment"
        ]
    
    async def _generate_integration_notes(self, agent_specs: List[AgentSpecification]) -> str:
        """Generate integration notes for the new agents."""
        
        return dedent(f"""\
            # Integration Notes for New Agents
            
            ## Deployment Steps
            1. Install generated agent files in appropriate directories
            2. Update team configuration to include new agents
            3. Test individual agent functionality
            4. Validate agent interactions and collaboration
            5. Monitor performance and adjust as needed
            
            ## Agent Dependencies
            {chr(10).join(f"- {spec.name}: Requires {', '.join(spec.tools_required)}" for spec in agent_specs)}
            
            ## Configuration Requirements
            - Ensure all required models are accessible
            - Verify tool permissions and API keys
            - Configure appropriate resource limits
            
            ## Monitoring Recommendations
            - Track success criteria metrics for each agent
            - Monitor collaboration effectiveness
            - Set up alerts for failure mode detection
        """)
    
    async def _calculate_quality_score(self, agent_specs: List[AgentSpecification]) -> float:
        """Calculate overall quality score for generated agents."""
        
        total_score = 0.0
        
        for spec in agent_specs:
            spec_score = 0.0
            
            # System prompt quality (30%)
            if len(spec.system_prompt) > 200:
                spec_score += 0.3
            
            # Instruction completeness (20%)
            if len(spec.instructions) >= 5:
                spec_score += 0.2
                
            # Tool appropriateness (20%) 
            if len(spec.tools_required) >= 2:
                spec_score += 0.2
                
            # Test coverage (15%)
            if len(spec.test_cases) >= 3:
                spec_score += 0.15
                
            # Documentation quality (15%)
            if len(spec.success_criteria) > 0 and len(spec.quality_checks) > 0:
                spec_score += 0.15
            
            total_score += spec_score
        
        return total_score / len(agent_specs) if agent_specs else 0.0
    
    async def _run_validation_tests(self, agent_specs: List[AgentSpecification]) -> List[Dict[str, Any]]:
        """Run validation tests on agent specifications."""
        
        validation_results = []
        
        for spec in agent_specs:
            result = {
                "agent_name": spec.name,
                "tests_passed": 0,
                "tests_failed": 0,
                "issues": []
            }
            
            # Validate system prompt
            if len(spec.system_prompt) > 100:
                result["tests_passed"] += 1
            else:
                result["tests_failed"] += 1
                result["issues"].append("System prompt too short")
            
            # Validate instructions
            if len(spec.instructions) >= 3:
                result["tests_passed"] += 1
            else:
                result["tests_failed"] += 1
                result["issues"].append("Insufficient instructions")
            
            # Validate tools
            if spec.tools_required:
                result["tests_passed"] += 1
            else:
                result["tests_failed"] += 1
                result["issues"].append("No tools specified")
            
            validation_results.append(result)
        
        return validation_results
    
    async def quick_agent_creation(self, role_name: str, capabilities: List[str], priority: str = "medium") -> AgentSpecification:
        """
        Quick method to create a single agent from basic parameters.
        
        Args:
            role_name: Name of the role to create
            capabilities: List of required capabilities
            priority: Priority level (low, medium, high, critical)
            
        Returns:
            AgentSpecification: Complete agent specification
        """
        
        # Create a simple vacant role
        role = VacantRole(
            role_name=role_name,
            title=role_name.title(),
            core_responsibilities=[f"Handle {role_name.lower()} tasks efficiently"],
            required_capabilities=capabilities,
            interaction_patterns={"team": "collaborative"},
            success_metrics=["Task completion", "Quality standards", "Team collaboration"],
            priority_level=priority
        )
        
        # Create scouting report with single role
        scouting_report = ScoutingReport(
            matched_agents=[],
            vacant_roles=[role],
            capability_gaps=capabilities,
            reuse_analysis={},
            priority_recommendations=[role_name]
        )
        
        # Generate the agent
        result = await self.develop_agents(scouting_report)
        
        if result.success and result.agents_created:
            return result.agents_created[0]
        else:
            raise Exception(f"Failed to create agent: {result.generation_summary}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_agent_developer():
        """Test the Agent Developer implementation"""
        
        developer = AgentDeveloper()
        
        # Test scenario: Create agents for a web development team
        vacant_roles = [
            VacantRole(
                role_name="Full Stack Developer",
                title="Senior Full Stack Developer",
                core_responsibilities=[
                    "Develop both frontend and backend components",
                    "Integrate APIs and databases", 
                    "Ensure responsive design and performance",
                    "Implement security best practices"
                ],
                required_capabilities=[
                    "JavaScript/TypeScript expertise",
                    "React/Node.js development",
                    "Database design and optimization",
                    "API development and integration",
                    "Testing and deployment"
                ],
                interaction_patterns={
                    "UI/UX Designer": "Collaborate on design implementation",
                    "DevOps Engineer": "Coordinate deployment and infrastructure",
                    "Product Manager": "Align development with business requirements"
                },
                success_metrics=[
                    "Feature delivery within timeline",
                    "Code quality and maintainability",
                    "Performance optimization",
                    "Security compliance"
                ],
                priority_level="high",
                domain_context="Web application development",
                complexity_level="complex"
            ),
            VacantRole(
                role_name="QA Automation Engineer", 
                title="Senior QA Automation Engineer",
                core_responsibilities=[
                    "Design and implement automated test frameworks",
                    "Create comprehensive test suites",
                    "Perform continuous integration testing",
                    "Identify and report quality issues"
                ],
                required_capabilities=[
                    "Test automation frameworks",
                    "Continuous integration/deployment",
                    "Performance testing",
                    "Security testing",
                    "Cross-browser testing"
                ],
                interaction_patterns={
                    "Full Stack Developer": "Collaborate on testable code design",
                    "DevOps Engineer": "Integrate testing into CI/CD pipeline"
                },
                success_metrics=[
                    "Test coverage percentage",
                    "Bug detection rate",
                    "Test execution time",
                    "Quality gate compliance"
                ],
                priority_level="high",
                domain_context="Quality assurance and testing",
                complexity_level="medium"
            )
        ]
        
        scouting_report = ScoutingReport(
            matched_agents=[],
            vacant_roles=vacant_roles,
            capability_gaps=[
                "Full-stack web development",
                "Automated testing and QA",
                "Performance optimization",
                "Security implementation"
            ],
            reuse_analysis={"existing_patterns": "Limited reuse opportunities"},
            priority_recommendations=["Full Stack Developer", "QA Automation Engineer"]
        )
        
        # Test agent development
        print("🚀 Testing Agent Developer...")
        print(f"📋 Creating agents for {len(vacant_roles)} vacant roles")
        
        result = await developer.develop_agents(scouting_report)
        
        print(f"\n✅ Agent Development Results:")
        print(f"Success: {result.success}")
        print(f"Agents Created: {len(result.agents_created)}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        
        print(f"\nCreated Agents:")
        for agent in result.agents_created:
            print(f"- {agent.name} ({agent.role})")
        
        print(f"\nGenerated Files:")
        for filepath in result.generated_files.keys():
            print(f"- {filepath}")
        
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"- {rec}")
    
    # Run the test
    asyncio.run(test_agent_developer())