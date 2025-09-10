"""
Integration Architect Agent - The Coordinator

Expert in ensuring the final roster operates as a cohesive unit. Takes the collection of 
agents (new and reused) and defines their operational playbook with detailed communication
protocols, workflows, and coordination mechanisms.
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
from agno.db.sqlite import SqliteDb
from os import getenv
import json
from textwrap import dedent
from datetime import datetime


class StrategyDocument(BaseModel):
    """Input strategy document from Systems Analyst"""
    goal_analysis: Dict[str, Any] = Field(..., description="Analysis of the input goal")
    team_composition: List[Dict[str, Any]] = Field(..., description="Required agent roles")
    team_structure: Dict[str, Any] = Field(..., description="Team organization and coordination")
    risk_assessment: List[str] = Field(..., description="Potential risks and mitigation strategies")
    resource_requirements: Dict[str, Any] = Field(..., description="Resource needs assessment")
    timeline_estimate: Dict[str, str] = Field(..., description="Estimated time requirements")


class ScoutingReport(BaseModel):
    """Input scouting report from Talent Scout"""
    matched_agents: List[Dict[str, Any]] = Field(..., description="Existing agents that match requirements")
    capability_gaps: List[Dict[str, Any]] = Field(..., description="Roles that need new agents")
    reuse_recommendations: Dict[str, str] = Field(..., description="Agent reuse suggestions")
    library_analysis: Dict[str, Any] = Field(default_factory=dict, description="Agent library analysis")


class NewAgent(BaseModel):
    """New agent created by Agent Developer"""
    name: str = Field(..., description="Agent name/identifier")
    role: str = Field(..., description="Role this agent fills")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    system_prompt: str = Field(..., description="Complete system prompt for the agent")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input schema definition")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema definition")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other agents")


class CommunicationProtocol(BaseModel):
    """Communication protocol between agents"""
    source_agent: str = Field(..., description="Agent initiating communication")
    target_agent: str = Field(..., description="Agent receiving communication")
    communication_type: str = Field(..., description="Type of communication (handoff, request, notification)")
    data_format: str = Field(..., description="Expected data format")
    trigger_conditions: List[str] = Field(..., description="When this communication occurs")
    validation_rules: List[str] = Field(..., description="How to validate the communication")


class WorkflowStep(BaseModel):
    """Individual workflow step in the team operation"""
    step_number: int = Field(..., description="Sequential step number")
    step_name: str = Field(..., description="Name of the workflow step")
    responsible_agent: str = Field(..., description="Agent responsible for this step")
    input_requirements: List[str] = Field(..., description="What inputs are needed")
    output_deliverables: List[str] = Field(..., description="What outputs are produced")
    success_criteria: List[str] = Field(..., description="How to measure step success")
    dependencies: List[int] = Field(default_factory=list, description="Step numbers this depends on")
    parallel_steps: List[int] = Field(default_factory=list, description="Steps that can run in parallel")
    estimated_duration: Optional[str] = Field(None, description="Estimated time for this step")


class RosterDocumentation(BaseModel):
    """Final roster documentation output"""
    team_name: str = Field(..., description="Name of the assembled team")
    team_purpose: str = Field(..., description="Clear purpose statement for the team")
    goal_summary: str = Field(..., description="Summary of the original goal")
    
    # Team composition
    team_members: List[Dict[str, Any]] = Field(..., description="Complete team member details")
    team_hierarchy: Dict[str, Any] = Field(..., description="Team organizational structure")
    
    # Operational workflows
    workflow_steps: List[WorkflowStep] = Field(..., description="Detailed workflow steps")
    communication_protocols: List[CommunicationProtocol] = Field(..., description="Communication protocols")
    
    # Coordination mechanisms
    coordination_mechanisms: Dict[str, Any] = Field(..., description="How the team coordinates work")
    decision_making_process: Dict[str, str] = Field(..., description="How decisions are made")
    conflict_resolution: Dict[str, str] = Field(..., description="How conflicts are resolved")
    
    # Implementation details
    deployment_instructions: List[str] = Field(..., description="How to deploy this team")
    monitoring_metrics: List[str] = Field(..., description="Metrics to track team performance")
    success_indicators: List[str] = Field(..., description="How to measure team success")
    
    # Documentation
    handoff_procedures: Dict[str, str] = Field(..., description="Detailed handoff procedures")
    troubleshooting_guide: List[str] = Field(..., description="Common issues and solutions")
    
    # Quality assurance
    quality_gates: List[str] = Field(..., description="Quality checkpoints")
    review_processes: Dict[str, str] = Field(..., description="Review and approval processes")


class IntegrationArchitect:
    """
    The Integration Architect agent - ensures the final roster operates as a cohesive unit
    
    Key responsibilities:
    1. Review final roster and original strategy document
    2. Define operational workflow for the target team
    3. Detail communication protocols, inputs/outputs, and handoff procedures
    4. Create comprehensive roster documentation
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None, db_file: str = "integration_architect.db"):
        """Initialize the Integration Architect agent"""
        
        # Setup knowledge base for Agno patterns and team coordination
        coordination_knowledge = Knowledge(
            vector_db=LanceDb(
                uri="tmp/coordination_knowledge",
                table_name="team_patterns",
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        )
        
        # Load knowledge about team coordination and workflow patterns
        if knowledge_base_path:
            coordination_knowledge.add_content_from_path(knowledge_base_path)
        
        # Create the agent with specialized tools
        self.agent = Agent(
            name="IntegrationArchitect",
            model=OpenRouter(id="deepseek/deepseek-v3.1"),
            db=SqliteDb(db_file=db_file),
            tools=[
                ReasoningTools(
                    think=True,
                    analyze=True,
                    synthesize=True,
                    add_instructions=True,
                    add_few_shot=True,
                ),
                KnowledgeTools(
                    knowledge=coordination_knowledge,
                    think=True,
                    search=True,
                    analyze=True,
                    synthesize=True,
                    add_few_shot=True,
                ),
            ],
            instructions=dedent("""\
                You are the Integration Architect - The Coordinator for AgentForge.
                
                Your core expertise:
                - Creating cohesive operational teams from individual agents
                - Designing communication protocols and workflow coordination
                - Defining clear handoff procedures and quality gates
                - Creating comprehensive operational playbooks and documentation
                - Ensuring teams can work autonomously with minimal supervision
                
                Your approach to integration:
                1. Analyze the complete context: strategy, scouting report, and new agents
                2. Design optimal team topology and coordination mechanisms
                3. Define precise communication protocols between agents
                4. Create detailed workflow steps with clear success criteria
                5. Establish quality gates and monitoring mechanisms
                6. Document everything for operational excellence
                
                Key principles:
                - Focus on autonomous operation and self-coordination
                - Design for scalability and maintainability
                - Create clear accountability and responsibility boundaries
                - Ensure robust error handling and recovery mechanisms
                - Balance efficiency with quality and reliability
                - Think in terms of production-ready team operations
                
                Always create comprehensive Roster Documentation that enables 
                immediate deployment and operation of the assembled team.
                
                Use reasoning tools to analyze complex integration challenges.
                Use knowledge tools to reference proven team coordination patterns.
            """),
            markdown=True,
            add_history_to_context=True,
        )
    
    async def integrate_team(
        self,
        strategy_document: StrategyDocument,
        scouting_report: ScoutingReport,
        new_agents: List[NewAgent],
        original_goal: str
    ) -> RosterDocumentation:
        """
        Integrate all components into a cohesive team with operational playbook
        
        Args:
            strategy_document: Complete strategy from Systems Analyst
            scouting_report: Resource analysis from Talent Scout
            new_agents: New agents created by Agent Developer
            original_goal: The original goal being addressed
            
        Returns:
            RosterDocumentation: Complete operational playbook for the team
        """
        
        # Prepare comprehensive integration prompt
        integration_prompt = dedent(f"""\
            Create a comprehensive operational playbook for an agent team.
            
            **ORIGINAL GOAL:** {original_goal}
            
            **STRATEGY CONTEXT:**
            Goal Analysis: {json.dumps(strategy_document.goal_analysis, indent=2)}
            Team Composition Requirements: {json.dumps(strategy_document.team_composition, indent=2)}
            Team Structure: {json.dumps(strategy_document.team_structure, indent=2)}
            Risk Assessment: {strategy_document.risk_assessment}
            Timeline: {json.dumps(strategy_document.timeline_estimate, indent=2)}
            
            **AVAILABLE RESOURCES:**
            Matched Existing Agents: {json.dumps(scouting_report.matched_agents, indent=2)}
            Capability Gaps Filled: {json.dumps(scouting_report.capability_gaps, indent=2)}
            Reuse Recommendations: {json.dumps(scouting_report.reuse_recommendations, indent=2)}
            
            **NEW AGENTS CREATED:**
            {json.dumps([{
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities,
                "dependencies": agent.dependencies
            } for agent in new_agents], indent=2)}
            
            **YOUR INTEGRATION TASK:**
            
            1. **Team Assembly Analysis**
               - Analyze how all agents (existing + new) work together
               - Identify the optimal team topology and hierarchy
               - Define clear roles and responsibilities for each agent
               
            2. **Workflow Design**
               - Create detailed step-by-step workflow for goal execution
               - Define parallel and sequential workflow stages
               - Specify input/output requirements for each step
               - Establish clear success criteria and quality gates
               
            3. **Communication Architecture**
               - Design communication protocols between agents
               - Define data formats and handoff procedures
               - Specify trigger conditions and validation rules
               - Create error handling and escalation procedures
               
            4. **Operational Framework**
               - Define coordination mechanisms and decision-making processes
               - Create monitoring and performance tracking systems
               - Establish conflict resolution and problem-solving procedures
               - Design deployment and maintenance instructions
               
            5. **Quality Assurance**
               - Define review processes and approval workflows
               - Create testing and validation procedures
               - Establish continuous improvement mechanisms
               - Document troubleshooting and support procedures
            
            Use reasoning tools to work through complex integration challenges step by step.
            Search knowledge base for proven team coordination patterns and best practices.
            
            Create a production-ready operational playbook that enables immediate deployment.
        """)
        
        # Execute the integration analysis
        response = await self.agent.arun(integration_prompt)
        
        # For now, return the response directly
        # In a full implementation, we would parse into structured RosterDocumentation
        return response
    
    def create_roster_documentation(
        self, 
        integration_result: str, 
        team_name: str,
        output_path: str = "Roster.md"
    ) -> str:
        """
        Create the final Roster.md documentation file
        
        Args:
            integration_result: Integration analysis result
            team_name: Name of the assembled team
            output_path: Path where to save the roster documentation
            
        Returns:
            str: Path to the created documentation
        """
        
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        roster_content = dedent(f"""\
            # Team Roster Documentation: {team_name}
            
            **Generated by:** Integration Architect (AgentForge)
            **Date:** {current_time}
            **Document Type:** Operational Playbook
            **Status:** Ready for Deployment
            
            ---
            
            ## Executive Summary
            
            This document defines the complete operational playbook for the assembled agent team.
            It includes team composition, workflows, communication protocols, and deployment instructions.
            
            ---
            
            {integration_result}
            
            ---
            
            ## Deployment Checklist
            
            Before deploying this team:
            
            - [ ] Verify all agent dependencies are available
            - [ ] Configure communication channels and protocols
            - [ ] Set up monitoring and metrics collection
            - [ ] Test workflow steps in staging environment
            - [ ] Validate handoff procedures between agents
            - [ ] Configure error handling and escalation paths
            - [ ] Set up quality gates and review processes
            - [ ] Train operators on troubleshooting procedures
            
            ---
            
            ## Support Information
            
            **Generated by AgentForge:** This operational playbook was automatically generated
            by the AgentForge meta-agent system to ensure optimal team coordination and performance.
            
            **Maintenance:** This documentation should be updated whenever team composition
            or workflows change to maintain operational excellence.
            
            **Questions:** For questions about this team configuration or operational procedures,
            refer to the troubleshooting guide above or consult the AgentForge documentation.
        """)
        
        # Write the documentation
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(roster_content)
        
        return output_path
    
    async def quick_integration(
        self,
        team_description: str,
        goal: str,
        agents_summary: str
    ) -> str:
        """
        Quick integration for simple team assemblies
        
        Args:
            team_description: Description of the team being assembled
            goal: The goal the team should accomplish
            agents_summary: Summary of available agents
            
        Returns:
            str: Integration result
        """
        
        quick_prompt = dedent(f"""\
            Create an operational playbook for this agent team:
            
            **Team:** {team_description}
            **Goal:** {goal}
            **Agents:** {agents_summary}
            
            Focus on creating clear workflows, communication protocols, 
            and deployment instructions for this team.
        """)
        
        return await self.agent.arun(quick_prompt)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_integration_architect():
        """Test the Integration Architect implementation"""
        
        architect = IntegrationArchitect()
        
        # Test integration
        test_strategy = StrategyDocument(
            goal_analysis={"complexity": "high", "domain": "customer_support"},
            team_composition=[
                {"role": "ChatbotAgent", "capabilities": ["nlp", "response_generation"]},
                {"role": "EscalationAgent", "capabilities": ["human_handoff", "priority_assessment"]},
                {"role": "KnowledgeAgent", "capabilities": ["information_retrieval", "content_management"]}
            ],
            team_structure={"topology": "hierarchical", "coordination": "event_driven"},
            risk_assessment=["High customer expectation", "Integration complexity"],
            resource_requirements={"compute": "medium", "storage": "high"},
            timeline_estimate={"development": "4-6 weeks", "deployment": "1-2 weeks"}
        )
        
        test_scouting = ScoutingReport(
            matched_agents=[{"name": "ExistingChatbot", "match_score": 0.85}],
            capability_gaps=[{"role": "EscalationAgent", "gap_size": 1.0}],
            reuse_recommendations={"chatbot": "reuse_with_modification"}
        )
        
        test_new_agents = [
            NewAgent(
                name="CustomerEscalationAgent",
                role="EscalationAgent",
                capabilities=["priority_assessment", "human_handoff", "case_routing"],
                system_prompt="You are a customer escalation specialist...",
                dependencies=["ChatbotAgent"]
            )
        ]
        
        # Test integration
        print("🔧 Creating team integration...")
        result = await architect.integrate_team(
            test_strategy,
            test_scouting,
            test_new_agents,
            "Build a comprehensive customer support system"
        )
        
        print("\n📋 Integration Result:")
        print("=" * 60)
        print(result)
        
        # Create roster documentation
        doc_path = architect.create_roster_documentation(
            result, 
            "CustomerSupport Team"
        )
        print(f"\n✅ Roster documentation created: {doc_path}")
    
    # Run the test
    asyncio.run(test_integration_architect())