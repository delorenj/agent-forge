"""
Simple Agent Developer Demo - Data Models Only

This demonstrates the Agent Developer data structures and core logic
without external dependencies like LanceDB or MCP.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from textwrap import dedent
import json


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


class SimpleAgentDeveloper:
    """Simplified Agent Developer for demonstration purposes"""
    
    def generate_system_prompt(self, role: VacantRole) -> str:
        """Generate system prompt for a role"""
        
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
    
    def determine_tools(self, role: VacantRole) -> List[str]:
        """Determine required tools based on role capabilities"""
        
        tools = ["ReasoningTools"]  # All agents get reasoning tools
        
        # Add tools based on capabilities
        for capability in role.required_capabilities:
            if any(keyword in capability.lower() for keyword in ["research", "knowledge", "information"]):
                if "KnowledgeTools" not in tools:
                    tools.append("KnowledgeTools")
            if any(keyword in capability.lower() for keyword in ["web", "search", "internet"]):
                if "WebSearchTools" not in tools:
                    tools.append("WebSearchTools") 
            if any(keyword in capability.lower() for keyword in ["code", "programming", "development"]):
                if "CodeTools" not in tools:
                    tools.append("CodeTools")
            if any(keyword in capability.lower() for keyword in ["file", "document", "content"]):
                if "FileTools" not in tools:
                    tools.append("FileTools")
            if any(keyword in capability.lower() for keyword in ["data", "analysis", "analytics"]):
                if "DataAnalysisTools" not in tools:
                    tools.append("DataAnalysisTools")
        
        return tools
    
    def generate_instructions(self, role: VacantRole) -> List[str]:
        """Generate behavioral instructions for a role"""
        
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
    
    def generate_initialization_code(self, agent_name: str, role: VacantRole) -> str:
        """Generate Python initialization code for the agent"""
        
        tools = self.determine_tools(role)
        tools_list = ", ".join(f"{tool}()" for tool in tools)
        
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
                        {self.generate_system_prompt(role)}
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
    
    def create_agent_specification(self, role: VacantRole) -> AgentSpecification:
        """Create a complete agent specification for a role"""
        
        agent_name = role.role_name.replace(" ", "").replace("-", "")
        
        return AgentSpecification(
            name=agent_name,
            role=role.role_name,
            title=role.title,
            system_prompt=self.generate_system_prompt(role),
            instructions=self.generate_instructions(role),
            tools_required=self.determine_tools(role),
            model_recommendations={"primary": "deepseek/deepseek-v3.1", "reasoning": "anthropic/claude-3-5-sonnet"},
            communication_style="Professional and collaborative, adapting to context",
            decision_making_approach="Systematic reasoning with evidence-based decisions",
            collaboration_patterns=role.interaction_patterns,
            success_criteria=role.success_metrics,
            failure_modes=[
                "Producing outputs that don't meet quality standards",
                "Failing to collaborate effectively with other agents",
                "Not using available tools appropriately"
            ],
            quality_checks=[
                "Verify outputs meet all specified success criteria",
                "Check that reasoning process is documented",
                "Ensure collaboration protocols are followed"
            ],
            initialization_code=self.generate_initialization_code(agent_name, role),
            example_usage=f"# Example usage for {agent_name} agent\n# Implementation would go here",
            test_cases=[
                {
                    "name": "basic_functionality_test",
                    "description": f"Test basic {role.role_name.lower()} functionality",
                    "expected_behavior": "Agent should complete task using appropriate reasoning"
                }
            ]
        )


def demonstrate_agent_developer():
    """Demonstrate the Agent Developer functionality"""
    
    print("🎯 Agent Developer - The Creator")
    print("Simple demonstration of agent specification generation")
    print("=" * 65)
    print()
    
    # Create sample vacant roles
    vacant_roles = [
        VacantRole(
            role_name="API Developer",
            title="Senior API Developer",
            core_responsibilities=[
                "Design and implement RESTful APIs",
                "Ensure API security and performance",
                "Create comprehensive API documentation"
            ],
            required_capabilities=[
                "REST API design principles",
                "Authentication and authorization",
                "OpenAPI/Swagger documentation",
                "Performance optimization"
            ],
            interaction_patterns={
                "Frontend Developer": "Provide API specifications",
                "QA Engineer": "Support API testing"
            },
            success_metrics=[
                "API response times under 200ms",
                "99.9% API uptime",
                "Complete documentation coverage"
            ],
            priority_level="high",
            domain_context="Web API development",
            complexity_level="complex"
        ),
        VacantRole(
            role_name="Data Scientist",
            title="Senior Data Scientist", 
            core_responsibilities=[
                "Analyze datasets for business insights",
                "Develop predictive models",
                "Create data visualizations"
            ],
            required_capabilities=[
                "Statistical analysis and modeling",
                "Machine learning algorithms", 
                "Data visualization tools",
                "Python/R programming"
            ],
            interaction_patterns={
                "Business Analyst": "Translate business needs",
                "Data Engineer": "Coordinate data pipelines"
            },
            success_metrics=[
                "Model accuracy above 85%",
                "Actionable business insights",
                "Reports delivered on schedule"
            ],
            priority_level="high",
            domain_context="Data science and analytics", 
            complexity_level="complex"
        )
    ]
    
    # Create scouting report
    scouting_report = ScoutingReport(
        matched_agents=[],
        vacant_roles=vacant_roles,
        capability_gaps=[
            "API development and design",
            "Data science and machine learning",
            "Statistical modeling"
        ],
        reuse_analysis={"existing_patterns": "Limited specialized patterns available"},
        priority_recommendations=["API Developer", "Data Scientist"]
    )
    
    print(f"📊 Scouting Report:")
    print(f"  • Vacant Roles: {len(scouting_report.vacant_roles)}")
    print(f"  • Capability Gaps: {len(scouting_report.capability_gaps)}")
    print(f"  • Priorities: {', '.join(scouting_report.priority_recommendations)}")
    print()
    
    # Create agent developer
    developer = SimpleAgentDeveloper()
    
    # Generate agent specifications
    agent_specs = []
    for role in vacant_roles:
        print(f"🔧 Creating agent specification for: {role.title}")
        spec = developer.create_agent_specification(role)
        agent_specs.append(spec)
        
        print(f"  ✅ Agent: {spec.name}")
        print(f"  📝 System Prompt: {len(spec.system_prompt)} characters")
        print(f"  📋 Instructions: {len(spec.instructions)} items")
        print(f"  🛠️  Tools: {', '.join(spec.tools_required)}")
        print(f"  🎯 Success Criteria: {len(spec.success_criteria)} metrics")
        print()
    
    # Show detailed example for first agent
    example_spec = agent_specs[0]
    print("📋 Detailed Example - API Developer Agent:")
    print("-" * 50)
    print()
    
    print("🔍 System Prompt:")
    print(example_spec.system_prompt[:300] + "..." if len(example_spec.system_prompt) > 300 else example_spec.system_prompt)
    print()
    
    print("📝 Instructions:")
    for i, instruction in enumerate(example_spec.instructions, 1):
        print(f"  {i}. {instruction}")
    print()
    
    print("🛠️ Required Tools:")
    for tool in example_spec.tools_required:
        print(f"  • {tool}")
    print()
    
    print("💻 Initialization Code (excerpt):")
    code_lines = example_spec.initialization_code.split('\n')
    for line in code_lines[:15]:  # Show first 15 lines
        print(f"    {line}")
    print("    ... (truncated)")
    print()
    
    print("✅ Quality Assessment:")
    quality_score = 0.0
    
    # System prompt quality (30%)
    if len(example_spec.system_prompt) > 200:
        quality_score += 0.3
        print("  ✅ System prompt: Comprehensive")
    
    # Instruction completeness (20%)  
    if len(example_spec.instructions) >= 5:
        quality_score += 0.2
        print("  ✅ Instructions: Complete")
        
    # Tool appropriateness (20%)
    if len(example_spec.tools_required) >= 2:
        quality_score += 0.2 
        print("  ✅ Tools: Appropriate selection")
        
    # Test coverage (15%)
    if len(example_spec.test_cases) >= 1:
        quality_score += 0.15
        print("  ✅ Tests: Basic coverage")
        
    # Documentation quality (15%)
    if len(example_spec.success_criteria) > 0:
        quality_score += 0.15
        print("  ✅ Documentation: Success criteria defined")
    
    print(f"\n  🎯 Overall Quality Score: {quality_score:.2f}/1.0")
    print()
    
    print("🚀 Agent Developer Summary:")
    print(f"  • Successfully created {len(agent_specs)} agent specifications")
    print(f"  • Generated comprehensive system prompts and instructions")
    print(f"  • Selected appropriate tools for each role")
    print(f"  • Created ready-to-use initialization code")
    print(f"  • Included quality validation and test cases")
    print()
    
    print("✅ Agent Developer Implementation Complete!")
    print("The Agent Developer can create comprehensive agent specifications")
    print("that follow Agno framework standards and best practices.")
    print()
    print("🔗 Integration with AgentForge workflow:")
    print("  1. Talent Scout identifies vacant roles")
    print("  2. Agent Developer creates specifications")
    print("  3. Integration Architect assembles team")
    print("  4. Engineering Manager delivers final package")


if __name__ == "__main__":
    demonstrate_agent_developer()