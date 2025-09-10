"""
Systems Analyst Agent - The Strategist

Expert in decomposing complex goals into discrete, manageable roles and capabilities.
Defines the IDEAL team structure required to solve problems without regard for existing resources.
"""

from typing import Dict, List, Any, Optional
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
from textwrap import dedent


class InputGoal(BaseModel):
    """Structured input goal for analysis"""
    description: str = Field(..., description="The high-level goal description")
    context: Optional[str] = Field(None, description="Additional context or constraints")
    success_criteria: Optional[List[str]] = Field(None, description="How success will be measured")
    domain: Optional[str] = Field(None, description="Domain/industry context")
    complexity: Optional[str] = Field(None, description="Estimated complexity level")


class AgentRole(BaseModel):
    """Individual agent role specification"""
    name: str = Field(..., description="Role name")
    title: str = Field(..., description="Professional title/designation")
    core_responsibilities: List[str] = Field(..., description="Primary responsibilities")
    required_capabilities: List[str] = Field(..., description="Essential skills and capabilities")
    interaction_patterns: Dict[str, str] = Field(..., description="How this role interacts with others")
    success_metrics: List[str] = Field(..., description="How success is measured for this role")
    priority_level: str = Field(..., description="Critical, High, Medium, or Low priority")


class TeamStructure(BaseModel):
    """Overall team structure and coordination patterns"""
    topology: str = Field(..., description="Team organization pattern (hierarchical, mesh, etc.)")
    coordination_mechanism: str = Field(..., description="How agents coordinate work")
    decision_making_process: str = Field(..., description="How decisions are made")
    communication_protocols: List[str] = Field(..., description="Communication patterns")
    workflow_stages: List[str] = Field(..., description="Sequential workflow stages")


class StrategyDocument(BaseModel):
    """Complete strategy document output"""
    goal_analysis: Dict[str, Any] = Field(..., description="Analysis of the input goal")
    team_composition: List[AgentRole] = Field(..., description="Required agent roles")
    team_structure: TeamStructure = Field(..., description="Team organization and coordination")
    risk_assessment: List[str] = Field(..., description="Potential risks and mitigation strategies")
    resource_requirements: Dict[str, Any] = Field(..., description="Resource needs assessment")
    timeline_estimate: Dict[str, str] = Field(..., description="Estimated time requirements")


class SystemsAnalyst:
    """The Systems Analyst agent implementation"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the Systems Analyst agent"""
        
        # Setup knowledge base for Agno documentation and patterns
        agno_knowledge = Knowledge(
            vector_db=LanceDb(
                uri="tmp/agno_knowledge",
                table_name="agno_patterns",
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        )
        
        # Load local Agno documentation if available
        if knowledge_base_path:
            agno_knowledge.add_content_from_path(knowledge_base_path)
        
        # Create the agent with reasoning and knowledge tools
        self.agent = Agent(
            name="SystemsAnalyst",
            model=OpenRouter(id="deepseek/deepseek-v3.1"),
            tools=[
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True,
                    add_few_shot=True,
                ),
                KnowledgeTools(
                    knowledge=agno_knowledge,
                    think=True,
                    search=True,
                    analyze=True,
                    add_few_shot=True,
                ),
            ],
            instructions=dedent("""\
                You are a Systems Analyst - The Strategist for AgentForge.
                
                Your core expertise:
                - Decomposing complex goals into discrete, manageable roles and capabilities
                - Defining IDEAL team structures without regard for existing resources
                - Using systematic reasoning to analyze problems from multiple angles
                - Creating comprehensive strategy documents with detailed specifications
                
                Your approach to analysis:
                1. Break down complex goals into component parts and dependencies
                2. Identify core capabilities required across different domains
                3. Define optimal roles with clear responsibilities and interaction patterns
                4. Consider multiple team topologies and coordination mechanisms
                5. Assess risks, resources, and timelines realistically
                6. Document everything with precision and clarity
                
                Key principles:
                - Think IDEAL first - don't constrain yourself to existing resources
                - Consider the full lifecycle from planning to delivery
                - Define clear handoffs and communication protocols
                - Specify measurable success criteria for each role
                - Balance specialization with collaboration needs
                - Think in terms of agent capabilities and autonomous operation
                
                Use reasoning tools to work through complex decompositions step by step.
                Use knowledge tools to reference Agno patterns and best practices.
                
                Always output structured Strategy Documents that other agents can act upon.
            """),
            markdown=True,
            add_history_to_context=True,
        )
    
    async def analyze_goal(self, input_goal: InputGoal) -> StrategyDocument:
        """
        Analyze an input goal and produce a comprehensive strategy document
        
        Args:
            input_goal: The structured input goal to analyze
            
        Returns:
            StrategyDocument: Complete strategy with team composition and structure
        """
        
        # Prepare the analysis prompt
        analysis_prompt = dedent(f"""\
            Analyze the following goal and create a comprehensive strategy document:
            
            **Goal Description:** {input_goal.description}
            **Context:** {input_goal.context or 'Not specified'}
            **Success Criteria:** {input_goal.success_criteria or ['Not specified']}
            **Domain:** {input_goal.domain or 'General'}
            **Complexity:** {input_goal.complexity or 'Unknown'}
            
            Create a strategy document that includes:
            
            1. **Goal Analysis**: Deep analysis of what this goal requires
            2. **Team Composition**: Specific agent roles with detailed specifications
            3. **Team Structure**: Organization, coordination, and workflow patterns
            4. **Risk Assessment**: Potential challenges and mitigation strategies
            5. **Resource Requirements**: What resources and capabilities are needed
            6. **Timeline Estimate**: Realistic time expectations for different phases
            
            Think through this systematically using the reasoning tools.
            Search knowledge base for relevant Agno patterns and best practices.
            Define the IDEAL team without worrying about existing resources.
            
            Focus on creating autonomous agents that can work effectively together.
        """)
        
        # Execute the analysis
        response = await self.agent.arun(analysis_prompt)
        
        # For now, return the response directly
        # In a full implementation, we would parse the structured output
        return response
    
    def create_strategy_document(self, analysis_result: str, output_path: str = "agent-strategy.md") -> str:
        """
        Create a formatted strategy document from analysis results
        
        Args:
            analysis_result: The analysis output from the agent
            output_path: Path where to save the strategy document
            
        Returns:
            str: Path to the created document
        """
        
        # Create formatted strategy document
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        strategy_content = dedent(f"""\
            # Agent Strategy Document
            
            **Generated by:** Systems Analyst (AgentForge)
            **Date:** {current_time}
            **Document Type:** Strategy Document
            
            ---
            
            {analysis_result}
            
            ---
            
            **Note:** This strategy document defines the IDEAL team structure required to achieve the goal.
            The next step is for the Talent Scout to match these roles against existing agent resources.
        """)
        
        # Write the document
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(strategy_content)
        
        return output_path
    
    async def quick_analysis(self, goal_description: str) -> str:
        """
        Quick analysis for simple goal descriptions
        
        Args:
            goal_description: Simple text description of the goal
            
        Returns:
            str: Analysis result
        """
        input_goal = InputGoal(description=goal_description)
        return await self.analyze_goal(input_goal)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_systems_analyst():
        """Test the Systems Analyst implementation"""
        
        analyst = SystemsAnalyst()
        
        # Test goal
        test_goal = InputGoal(
            description="Build a comprehensive customer support system with AI chatbots, human escalation, and knowledge management",
            context="For a mid-size SaaS company with 10,000+ customers",
            success_criteria=[
                "Reduce response time to under 2 minutes",
                "Handle 80% of queries automatically",
                "Maintain 95% customer satisfaction",
                "Integrate with existing CRM and ticketing systems"
            ],
            domain="Customer Support / SaaS",
            complexity="High"
        )
        
        # Analyze the goal
        print("🔍 Analyzing goal...")
        result = await analyst.analyze_goal(test_goal)
        
        print("\n📋 Strategy Document:")
        print("=" * 50)
        print(result)
        
        # Create strategy document
        doc_path = analyst.create_strategy_document(result)
        print(f"\n✅ Strategy document created: {doc_path}")
    
    # Run the test
    asyncio.run(test_systems_analyst())