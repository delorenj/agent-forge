"""
Engineering Manager (Orchestrator) - The Central Nervous System of AgentForge

This module implements the central orchestrator that manages the entire AgentForge workflow
from initial goal intake to final deliverable. It delegates tasks to specialized agents
and coordinates their interactions according to the PRD workflow:

1. Receive Input Goal
2. Delegate analysis to Systems Analyst
3. Receive Strategy Document and delegate to Talent Scout
4. Receive Scouting Report and delegate to Agent Developer
5. Delegate final assembly to Integration Architect
6. Package and deliver final output
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent, RunOutput
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from os import getenv
import asyncio
from datetime import datetime
import json
import logging

# Optional MCP integration
try:
    from agno.tools.mcp import MCPTools, StreamableHTTPClientParams

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPTools = None
    StreamableHTTPClientParams = None

# Optional Knowledge and Vector DB integration
try:
    from agno.tools.knowledge import KnowledgeTools
    from agno.knowledge.knowledge import Knowledge
    from agno.embedder.openai import OpenAIEmbedder
    from agno.vectordb.lancedb import LanceDb, SearchType

    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False
    KnowledgeTools = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Input Schema - Typed Input Goal
class InputGoal(BaseModel):
    """Structured input goal with validation requirements for AgentForge."""

    goal_description: str = Field(
        ...,
        description="High-level description of what needs to be achieved",
        min_length=10,
        max_length=1000,
    )

    domain: str = Field(
        ...,
        description="Primary domain or field (e.g., 'web development', 'data science', 'marketing')",
        min_length=2,
        max_length=100,
    )

    complexity_level: str = Field(
        default="medium",
        description="Complexity level of the goal",
        pattern="^(simple|medium|complex|enterprise)$",
    )

    constraints: List[str] = Field(
        default_factory=list, description="Any constraints or limitations to consider"
    )

    success_criteria: List[str] = Field(
        default_factory=list, description="Specific criteria that define success"
    )

    existing_resources: List[str] = Field(
        default_factory=list,
        description="Any existing resources or tools that should be leveraged",
    )


# Output Schema - Final Deliverable
class TeamPackage(BaseModel):
    """Final deliverable containing the complete agent team and documentation."""

    team_name: str = Field(..., description="Name of the assembled team")
    goal_summary: str = Field(..., description="Summary of the original goal")

    # Team composition
    team_members: List[Dict[str, Any]] = Field(
        ..., description="List of agents with their roles and capabilities"
    )

    # Workflow documentation
    workflow_steps: List[str] = Field(
        ..., description="Step-by-step workflow for the team to execute"
    )

    communication_protocols: Dict[str, str] = Field(
        ..., description="How agents should communicate and handoff tasks"
    )

    # Documentation files
    strategy_document: str = Field(..., description="Complete strategy analysis")
    scouting_report: str = Field(
        ..., description="Resource analysis and agent matching"
    )
    roster_documentation: str = Field(
        ..., description="Final team operational playbook"
    )

    # Implementation details
    new_agents_created: List[Dict[str, str]] = Field(
        default_factory=list, description="Details of any new agents that were created"
    )

    existing_agents_used: List[Dict[str, str]] = Field(
        default_factory=list, description="Details of existing agents that were reused"
    )

    deployment_instructions: str = Field(
        ..., description="Instructions for deploying this team"
    )

    success_metrics: List[str] = Field(
        ..., description="How to measure the team's success"
    )


# Intermediate data structures for workflow coordination
class StrategyDocument(BaseModel):
    """Strategy document from Systems Analyst."""

    ideal_roles: List[Dict[str, Any]]
    capabilities_required: List[str]
    interaction_patterns: Dict[str, List[str]]
    technical_requirements: List[str]


class ScoutingReport(BaseModel):
    """Scouting report from Talent Scout."""

    matched_agents: List[Dict[str, Any]]
    capability_gaps: List[Dict[str, Any]]
    reuse_recommendations: Dict[str, str]


class EngineeringManager:
    """
    The Engineering Manager (Orchestrator) - Central nervous system of AgentForge.

    Manages the complete workflow from goal intake to final deliverable by coordinating
    with specialized agents: Systems Analyst, Talent Scout, Agent Developer, and
    Integration Architect.
    """

    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = "agentforge.db",
        mcp_url: str = None,
        knowledge_base_path: str = None,
    ):
        """Initialize the Engineering Manager with required components."""

        # Set up MCP connection for AgentForge tools (if available)
        self.mcp_url = mcp_url or "https://mcp.delo.sh/metamcp/agentforge/mcp"
        self.server_params = None
        if MCP_AVAILABLE and StreamableHTTPClientParams:
            self.server_params = StreamableHTTPClientParams(
                url=self.mcp_url,
                headers={"Authorization": f"Bearer {getenv('MCP_API_KEY')}"},
                terminate_on_close=True,
            )

        # Initialize knowledge base for Agno documentation (if available)
        self.knowledge_base = None
        if KNOWLEDGE_AVAILABLE and knowledge_base_path:
            self.knowledge_base = Knowledge(
                vector_db=LanceDb(
                    uri=knowledge_base_path,
                    table_name="agno_docs",
                    search_type=SearchType.hybrid,
                    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
                )
            )

        # Create the main orchestrator agent
        tools = [ReasoningTools(add_instructions=True)]
        if self.knowledge_base and KNOWLEDGE_AVAILABLE and KnowledgeTools:
            tools.append(KnowledgeTools(knowledge=self.knowledge_base))

        self.agent = Agent(
            name="EngineeringManager",
            model=OpenRouter(id=model_id),
            db=SqliteDb(db_file=db_file),
            input_schema=InputGoal,
            output_schema=TeamPackage,
            tools=tools,
            add_history_to_context=True,
            markdown=True,
            role="Engineering Manager and System Orchestrator",
            instructions=[
                "You are the Engineering Manager for AgentForge - the central orchestrator",
                "Your primary function is orchestration and delegation, not direct execution",
                "Follow the exact workflow: Analyst → Scout → Developer → Architect",
                "Always validate input goals and ensure comprehensive team packages",
                "Use reasoning tools to think through complex delegation decisions",
                "Maintain clear communication protocols between agents",
                "Focus on maximizing reuse of existing agents while filling gaps with new ones",
            ],
        )

        # Track workflow state
        self.current_workflow: Optional[Dict[str, Any]] = None
        self.workflow_history: List[Dict[str, Any]] = []

    async def orchestrate_goal(self, input_goal: InputGoal) -> TeamPackage:
        """
        Main orchestration method that executes the complete AgentForge workflow.

        Args:
            input_goal: Validated input goal with all requirements

        Returns:
            TeamPackage: Complete team package ready for deployment
        """
        logger.info(
            f"Starting orchestration for goal: {input_goal.goal_description[:50]}..."
        )

        # Initialize workflow tracking
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_workflow = {
            "id": workflow_id,
            "input_goal": input_goal.model_dump(),
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "steps": [],
        }

        try:
            # Step 1: Delegate to Systems Analyst for strategy
            logger.info("Step 1: Delegating to Systems Analyst...")
            strategy_doc = await self._delegate_to_systems_analyst(input_goal)
            self._track_step("systems_analyst", "completed", {"strategy": "generated"})

            # Step 2: Delegate to Talent Scout for resource analysis
            logger.info("Step 2: Delegating to Talent Scout...")
            scouting_report = await self._delegate_to_talent_scout(
                strategy_doc, input_goal
            )
            self._track_step(
                "talent_scout",
                "completed",
                {"matches": "identified", "gaps": "identified"},
            )

            # Step 3: Delegate to Agent Developer for gap filling (if needed)
            new_agents = []
            if scouting_report.capability_gaps:
                logger.info("Step 3: Delegating to Agent Developer...")
                new_agents = await self._delegate_to_agent_developer(
                    scouting_report, strategy_doc
                )
                self._track_step(
                    "agent_developer", "completed", {"new_agents": len(new_agents)}
                )

            # Step 4: Delegate to Integration Architect for final assembly
            logger.info("Step 4: Delegating to Integration Architect...")
            team_package = await self._delegate_to_integration_architect(
                strategy_doc, scouting_report, new_agents, input_goal
            )
            self._track_step(
                "integration_architect", "completed", {"team_assembled": True}
            )

            # Step 5: Finalize and package output
            logger.info("Step 5: Finalizing team package...")
            final_package = await self._finalize_team_package(team_package, input_goal)

            # Complete workflow tracking
            self.current_workflow["status"] = "completed"
            self.current_workflow["end_time"] = datetime.now().isoformat()
            self.workflow_history.append(self.current_workflow)

            logger.info(
                f"Orchestration completed successfully for workflow {workflow_id}"
            )
            return final_package

        except Exception as e:
            logger.error(f"Orchestration failed for workflow {workflow_id}: {str(e)}")
            self.current_workflow["status"] = "failed"
            self.current_workflow["error"] = str(e)
            self.current_workflow["end_time"] = datetime.now().isoformat()
            self.workflow_history.append(self.current_workflow)
            raise

    async def _delegate_to_systems_analyst(
        self, input_goal: InputGoal
    ) -> StrategyDocument:
        """Delegate analysis and strategy to the Systems Analyst."""

        # Use MCP tools if available, otherwise use agent directly
        tools = []
        if MCP_AVAILABLE and self.server_params and MCPTools:
            try:
                async with MCPTools(self.server_params) as mcp_tools:
                    tools = [mcp_tools]
            except Exception as e:
                logger.warning(f"MCP tools unavailable: {e}")

        prompt = f"""
            As the Engineering Manager, I need you to act as the Systems Analyst for this goal:
            
            GOAL: {input_goal.goal_description}
            DOMAIN: {input_goal.domain}
            COMPLEXITY: {input_goal.complexity_level}
            CONSTRAINTS: {', '.join(input_goal.constraints) if input_goal.constraints else 'None'}
            SUCCESS CRITERIA: {', '.join(input_goal.success_criteria) if input_goal.success_criteria else 'To be defined'}
            
            Analyze this goal in depth and define the ideal team structure. Focus on:
            
            1. Core capabilities and tasks required
            2. Optimal roles and responsibilities 
            3. Interaction patterns between team members
            4. Technical requirements and dependencies
            
        Output a detailed strategy document that defines the idealized team makeup and 
        specifications for each role, without regard for existing resources.
        """

        response = await self.agent.arun(message=prompt, tools=tools)

        # Parse response into StrategyDocument
        # This is a simplified version - in production, you'd have more sophisticated parsing
        return StrategyDocument(
            ideal_roles=[
                {"role": "analyst", "capabilities": ["analysis"]}
            ],  # Placeholder
            capabilities_required=["problem_solving", "technical_analysis"],
            interaction_patterns={"analyst": ["developer", "architect"]},
            technical_requirements=["python", "agno_framework"],
        )

    async def _delegate_to_talent_scout(
        self, strategy: StrategyDocument, input_goal: InputGoal
    ) -> ScoutingReport:
        """Delegate resource analysis to the Talent Scout."""

        # This would integrate with the agent library scanning
        # For now, return a placeholder structure
        return ScoutingReport(
            matched_agents=[{"name": "existing_analyst", "match_score": 0.8}],
            capability_gaps=[{"role": "specialized_developer", "gap_size": 0.6}],
            reuse_recommendations={"analyst_role": "reuse_existing"},
        )

    async def _delegate_to_agent_developer(
        self, scouting_report: ScoutingReport, strategy: StrategyDocument
    ) -> List[Dict[str, Any]]:
        """Delegate new agent creation to the Agent Developer."""

        from agents.agent_developer import (
            AgentDeveloper,
            ScoutingReport as DevScoutingReport,
            VacantRole,
        )

        logger.info("Initializing Agent Developer...")
        developer = AgentDeveloper()

        # Convert strategy roles to vacant roles for Agent Developer
        vacant_roles = []
        for role_data in strategy.ideal_roles:
            vacant_role = VacantRole(
                role_name=role_data.get("name", "Unnamed Role"),
                title=role_data.get("title", role_data.get("name", "Specialist")),
                core_responsibilities=role_data.get(
                    "responsibilities", ["Handle assigned tasks"]
                ),
                required_capabilities=role_data.get(
                    "capabilities", ["Basic task execution"]
                ),
                interaction_patterns=role_data.get(
                    "interactions", {"team": "collaborative"}
                ),
                success_metrics=role_data.get(
                    "metrics", ["Task completion", "Quality standards"]
                ),
                priority_level=role_data.get("priority", "medium"),
                domain_context=role_data.get("domain", "General"),
                complexity_level=role_data.get("complexity", "medium"),
            )
            vacant_roles.append(vacant_role)

        # Create Agent Developer scouting report
        dev_scouting_report = DevScoutingReport(
            matched_agents=scouting_report.matched_agents,
            vacant_roles=vacant_roles,
            capability_gaps=scouting_report.capability_gaps,
            reuse_analysis=scouting_report.reuse_recommendations,
            priority_recommendations=[role.role_name for role in vacant_roles],
        )

        # Generate agents using Agent Developer
        logger.info(f"Creating {len(vacant_roles)} new agents...")
        generation_result = await developer.develop_agents(
            dev_scouting_report,
            strategy_context={
                "capabilities_required": strategy.capabilities_required,
                "technical_requirements": strategy.technical_requirements,
                "interaction_patterns": strategy.interaction_patterns,
            },
        )

        if generation_result.success:
            logger.info(
                f"Successfully created {len(generation_result.agents_created)} agents"
            )

            # Convert agent specifications to dictionary format
            new_agents = []
            for agent_spec in generation_result.agents_created:
                new_agents.append(
                    {
                        "name": agent_spec.name,
                        "role": agent_spec.role,
                        "title": agent_spec.title,
                        "prompt": agent_spec.system_prompt,
                        "capabilities": agent_spec.tools_required,
                        "instructions": agent_spec.instructions,
                        "success_criteria": agent_spec.success_criteria,
                        "collaboration_patterns": agent_spec.collaboration_patterns,
                        "initialization_code": agent_spec.initialization_code,
                        "example_usage": agent_spec.example_usage,
                        "quality_score": generation_result.quality_score,
                    }
                )

            return new_agents
        else:
            logger.warning(
                f"Agent generation failed: {generation_result.generation_summary}"
            )
            return []

    async def _delegate_to_integration_architect(
        self,
        strategy: StrategyDocument,
        scouting: ScoutingReport,
        new_agents: List[Dict[str, Any]],
        input_goal: InputGoal,
    ) -> TeamPackage:
        """Delegate final team assembly to the Integration Architect."""

        # Import and initialize the Integration Architect
        from agents.integration_architect import (
            IntegrationArchitect,
            StrategyDocument as ArchitectStrategy,
            ScoutingReport as ArchitectScouting,
            NewAgent,
        )

        architect = IntegrationArchitect(
            db_file=f"integration_architect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )

        # Convert internal data structures to architect's expected formats
        architect_strategy = ArchitectStrategy(
            goal_analysis={
                "complexity": input_goal.complexity_level,
                "domain": input_goal.domain,
            },
            team_composition=strategy.ideal_roles,
            team_structure={
                "topology": "hierarchical",
                "coordination": "workflow_based",
            },
            risk_assessment=["Integration complexity", "Communication overhead"],
            resource_requirements={
                "agents": len(strategy.ideal_roles),
                "coordination": "high",
            },
            timeline_estimate={"integration": "1-2 days", "deployment": "immediate"},
        )

        architect_scouting = ArchitectScouting(
            matched_agents=scouting.matched_agents,
            capability_gaps=scouting.capability_gaps,
            reuse_recommendations=scouting.reuse_recommendations,
        )

        architect_new_agents = [
            NewAgent(
                name=agent.get("name", f"Agent_{i}"),
                role=agent.get("role", "specialist"),
                capabilities=agent.get("capabilities", []),
                system_prompt=agent.get(
                    "prompt", f"You are a {agent.get('role', 'specialist')} agent..."
                ),
                dependencies=agent.get("dependencies", []),
            )
            for i, agent in enumerate(new_agents)
        ]

        # Execute integration
        logger.info("Executing Integration Architect analysis...")
        roster_documentation = await architect.integrate_team(
            architect_strategy,
            architect_scouting,
            architect_new_agents,
            input_goal.goal_description,
        )

        # Create the roster documentation file
        doc_path = architect.create_roster_documentation(
            roster_documentation, f"AgentForge Team - {input_goal.domain}", "Roster.md"
        )

        logger.info(f"Roster documentation created: {doc_path}")

        # Build comprehensive team package
        return TeamPackage(
            team_name=f"AgentForge Team - {input_goal.domain}",
            goal_summary=input_goal.goal_description,
            team_members=[
                # Existing agents
                *[
                    {
                        "role": agent["name"],
                        "type": "existing",
                        "capabilities": agent.get("capabilities", []),
                        "source": "agent_library",
                    }
                    for agent in scouting.matched_agents
                ],
                # New agents
                *[
                    {
                        "role": agent.name,
                        "type": "new",
                        "capabilities": agent.capabilities,
                        "source": "agent_developer",
                    }
                    for agent in architect_new_agents
                ],
            ],
            workflow_steps=[
                "1. Initialize team coordination and communication protocols",
                "2. Execute parallel agent workflows according to operational playbook",
                "3. Monitor progress and handle inter-agent coordination",
                "4. Validate deliverables through quality gates",
                "5. Complete goal delivery and generate success metrics",
            ],
            communication_protocols={
                "coordination_hub": "Central coordination through message passing",
                "workflow_handoffs": "Structured handoffs with validation",
                "quality_gates": "Automated quality checks at key milestones",
                "error_handling": "Escalation procedures for failures",
            },
            strategy_document=json.dumps(strategy.model_dump(), indent=2),
            scouting_report=json.dumps(scouting.model_dump(), indent=2),
            roster_documentation=str(roster_documentation),
            new_agents_created=[
                {
                    "name": agent.name,
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                }
                for agent in architect_new_agents
            ],
            existing_agents_used=[
                {"name": agent["name"], "match_score": agent.get("match_score", 0.0)}
                for agent in scouting.matched_agents
            ],
            deployment_instructions=f"Deploy using Agno framework. Roster documentation available at: {doc_path}. Follow deployment checklist in documentation.",
            success_metrics=input_goal.success_criteria
            or [
                "Goal completion",
                "Quality standards",
                "Timeline adherence",
                "Team coordination efficiency",
            ],
        )

    async def _finalize_team_package(
        self, team_package: TeamPackage, input_goal: InputGoal
    ) -> TeamPackage:
        """Finalize and validate the team package before delivery."""

        # Add any final validations, formatting, or enhancements
        logger.info(f"Finalizing team package: {team_package.team_name}")

        return team_package

    def _track_step(self, step_name: str, status: str, metadata: Dict[str, Any] = None):
        """Track workflow step progress."""
        if self.current_workflow:
            step_data = {
                "step": step_name,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            self.current_workflow["steps"].append(step_data)

    def get_workflow_status(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get current or historical workflow status."""
        if workflow_id:
            for workflow in self.workflow_history:
                if workflow["id"] == workflow_id:
                    return workflow
            return {"error": "Workflow not found"}

        return self.current_workflow or {"status": "no_active_workflow"}

    async def run_with_goal(
        self, goal_description: str, domain: str = "general", **kwargs
    ) -> TeamPackage:
        """Convenience method to run orchestrator with a simple goal description."""

        input_goal = InputGoal(
            goal_description=goal_description, domain=domain, **kwargs
        )

        return await self.orchestrate_goal(input_goal)


# Factory function for easy initialization
def create_orchestrator(
    model_id: str = "deepseek/deepseek-v3.1", db_file: str = "agentforge.db", **kwargs
) -> EngineeringManager:
    """Create and configure an Engineering Manager orchestrator."""
    return EngineeringManager(model_id=model_id, db_file=db_file, **kwargs)

