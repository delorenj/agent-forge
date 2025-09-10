"""
Agent Developer Demo - Simple demonstration of the Agent Developer functionality

This demonstrates the Agent Developer (The Creator) without external dependencies.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_developer import VacantRole, ScoutingReport


def create_sample_vacant_roles():
    """Create sample vacant roles for demonstration"""
    
    return [
        VacantRole(
            role_name="API Developer",
            title="Senior API Developer", 
            core_responsibilities=[
                "Design and implement RESTful APIs",
                "Ensure API security and performance",
                "Create comprehensive API documentation",
                "Handle API versioning and backwards compatibility"
            ],
            required_capabilities=[
                "REST API design principles",
                "Authentication and authorization",
                "API rate limiting and throttling",
                "OpenAPI/Swagger documentation",
                "Performance optimization"
            ],
            interaction_patterns={
                "Frontend Developer": "Provide API specifications and endpoints",
                "Database Developer": "Coordinate data access patterns", 
                "QA Engineer": "Support API testing and validation"
            },
            success_metrics=[
                "API response times under 200ms",
                "99.9% API uptime",
                "Complete API documentation coverage",
                "Zero security vulnerabilities"
            ],
            priority_level="high",
            domain_context="Web API development",
            complexity_level="complex"
        ),
        VacantRole(
            role_name="Data Scientist",
            title="Senior Data Scientist",
            core_responsibilities=[
                "Analyze large datasets for business insights",
                "Develop predictive models and algorithms",
                "Create data visualizations and reports",
                "Collaborate with stakeholders on data strategy"
            ],
            required_capabilities=[
                "Statistical analysis and modeling",
                "Machine learning algorithms",
                "Data visualization tools",
                "SQL and database querying",
                "Python/R programming"
            ],
            interaction_patterns={
                "Business Analyst": "Translate business needs to data questions",
                "Data Engineer": "Coordinate data pipeline requirements",
                "Product Manager": "Provide insights for product decisions"
            },
            success_metrics=[
                "Model accuracy above 85%",
                "Insights lead to measurable business impact",
                "Reports delivered on schedule",
                "Stakeholder satisfaction with analysis"
            ],
            priority_level="high",
            domain_context="Data science and analytics",
            complexity_level="complex"
        )
    ]


def demonstrate_agent_specifications():
    """Demonstrate agent specification creation"""
    
    print("🎯 Agent Developer - The Creator Demonstration")
    print("=" * 60)
    print()
    
    # Create sample roles
    vacant_roles = create_sample_vacant_roles()
    
    print(f"📋 Created {len(vacant_roles)} sample vacant roles:")
    for i, role in enumerate(vacant_roles, 1):
        print(f"  {i}. {role.title} - {role.role_name}")
        print(f"     Priority: {role.priority_level}, Complexity: {role.complexity_level}")
        print(f"     Capabilities: {len(role.required_capabilities)} required")
        print()
    
    # Create scouting report
    scouting_report = ScoutingReport(
        matched_agents=[
            {"name": "existing_qa_agent", "match_score": 0.8, "role": "Quality Assurance"}
        ],
        vacant_roles=vacant_roles,
        capability_gaps=[
            "API development and design",
            "Data science and machine learning",
            "Statistical modeling",
            "API security implementation"
        ],
        reuse_analysis={
            "existing_patterns": "Limited API and data science patterns available",
            "reuse_potential": "Medium - some general patterns exist"
        },
        priority_recommendations=["API Developer", "Data Scientist"]
    )
    
    print("📊 Scouting Report Summary:")
    print(f"  • Matched Agents: {len(scouting_report.matched_agents)}")
    print(f"  • Vacant Roles: {len(scouting_report.vacant_roles)}")
    print(f"  • Capability Gaps: {len(scouting_report.capability_gaps)}")
    print(f"  • Priority Recommendations: {', '.join(scouting_report.priority_recommendations)}")
    print()
    
    # Demonstrate agent specification structure
    print("🔧 Agent Specification Structure:")
    print("The Agent Developer creates comprehensive specifications including:")
    print("  • System Prompt - Defines agent identity and approach")
    print("  • Instructions - Specific behavioral guidelines")
    print("  • Tools Required - Necessary capabilities and tools")
    print("  • Communication Style - How the agent interacts")
    print("  • Success Criteria - Measurable outcomes")
    print("  • Quality Checks - Validation requirements")
    print("  • Implementation Code - Ready-to-use Python code")
    print("  • Test Cases - Validation scenarios")
    print()
    
    # Show example system prompt generation
    api_role = vacant_roles[0]  # API Developer
    
    print("📝 Example System Prompt Generation for API Developer:")
    print("-" * 50)
    
    example_prompt = f"""You are the {api_role.title} for AgentForge, specializing in {api_role.role_name.lower()}.

Your core expertise:
{chr(10).join(f'- {resp}' for resp in api_role.core_responsibilities)}

Your essential capabilities:  
{chr(10).join(f'- {cap}' for cap in api_role.required_capabilities)}

Your approach to work:
1. Use systematic reasoning to break down complex problems
2. Leverage your specialized knowledge and tools effectively
3. Collaborate seamlessly with other agents in the team
4. Maintain high quality standards in all deliverables
5. Continuously validate your outputs against success criteria

Success is measured by:
{chr(10).join(f'- {metric}' for metric in api_role.success_metrics)}

Domain context: {api_role.domain_context}
Priority level: {api_role.priority_level}

Always use reasoning tools to think through complex decisions.
Use knowledge tools to access relevant information and patterns.
Maintain clear communication and provide structured outputs."""
    
    print(example_prompt)
    print("-" * 50)
    print()
    
    # Show example tools determination
    print("🛠️ Tool Selection Logic:")
    tools = ["ReasoningTools"]  # All agents get reasoning tools
    
    for capability in api_role.required_capabilities:
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
    
    print(f"Selected tools for API Developer: {', '.join(tools)}")
    print()
    
    # Show example initialization code
    print("💻 Example Generated Initialization Code:")
    print("-" * 50)
    
    agent_name = api_role.role_name.replace(" ", "").replace("-", "")
    tools_list = ", ".join(f"{tool}()" for tool in tools)
    
    init_code = f'''from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools.knowledge import KnowledgeTools
from textwrap import dedent

def create_{agent_name.lower()}_agent(model_id="deepseek/deepseek-v3.1"):
    """Create and configure the {api_role.title} agent."""
    
    agent = Agent(
        name="{agent_name}",
        model=OpenRouter(id=model_id),
        tools=[
            {tools_list}
        ],
        instructions=dedent(""\\
            # System prompt would go here
        ""),
        markdown=True,
        add_history_to_context=True,
    )
    
    return agent

# Example usage
if __name__ == "__main__":
    agent = create_{agent_name.lower()}_agent()
    # Use the agent for {api_role.role_name.lower()}'''
    
    print(init_code)
    print("-" * 50)
    print()
    
    # Quality assessment
    print("✅ Quality Assessment Criteria:")
    print("The Agent Developer validates each generated agent against:")
    print("  • System prompt completeness (>100 characters)")
    print("  • Instruction adequacy (≥3 instructions)")
    print("  • Tool appropriateness (≥2 tools including ReasoningTools)")
    print("  • Test coverage (≥3 test cases)")
    print("  • Documentation quality (success criteria + quality checks)")
    print()
    
    print("🎯 Agent Developer Capabilities Summary:")
    print("✅ Creates comprehensive agent specifications")
    print("✅ Generates role-specific system prompts")
    print("✅ Determines appropriate tools and capabilities")
    print("✅ Creates ready-to-use implementation code")
    print("✅ Includes comprehensive test cases")
    print("✅ Validates quality and standards compliance")
    print("✅ Integrates with AgentForge workflow")
    print("✅ Supports both batch and quick agent creation")
    print()
    
    print("🚀 The Agent Developer is ready to create specialized agents!")
    print("   Use it to fill capability gaps identified by the Talent Scout")
    print("   and create production-ready agents that follow best practices.")
    

if __name__ == "__main__":
    demonstrate_agent_specifications()