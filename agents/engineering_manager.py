"""
Engineering Manager (Orchestrator) - The central nervous system of AgentForge.
Manages workflow from goal intake to final deliverable.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import asyncio
from .base import AgentForgeBase, AgentForgeInput, AgentForgeOutput


class ComplexityLevel(str, Enum):
    """Goal complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"


class InputGoal(BaseModel):
    """Typed input for Engineering Manager."""
    goal_description: str = Field(..., description="High-level goal description")
    domain: str = Field(..., description="Problem domain (e.g., 'web development', 'data analysis')")
    complexity_level: ComplexityLevel = Field(default=ComplexityLevel.MEDIUM)
    timeline: Optional[str] = Field(None, description="Expected timeline")
    constraints: Optional[List[str]] = Field(default_factory=list)
    success_criteria: Optional[List[str]] = Field(default_factory=list)
    existing_resources: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowStep(str, Enum):
    """Workflow steps in the AgentForge process."""
    GOAL_INTAKE = "goal_intake"
    STRATEGY_ANALYSIS = "strategy_analysis"
    RESOURCE_SCOUTING = "resource_scouting"
    AGENT_DEVELOPMENT = "agent_development"
    TEAM_INTEGRATION = "team_integration"
    FINAL_PACKAGING = "final_packaging"


class TeamPackage(BaseModel):
    """Final output package from Engineering Manager."""
    goal: InputGoal
    strategy_document: str
    scouting_report: str
    new_agents: List[Dict[str, Any]]
    existing_agents: List[Dict[str, Any]]
    roster_documentation: str
    deployment_instructions: str
    created_at: datetime = Field(default_factory=datetime.now)


class EngineeringManager(AgentForgeBase):
    """
    The Engineering Manager (Orchestrator) - Central coordinator of AgentForge.
    
    Responsibilities:
    1. Receive Input Goal
    2. Delegate analysis to Systems Analyst
    3. Receive Strategy Document and delegate to Talent Scout
    4. Receive Scouting Report and delegate to Agent Developer
    5. Delegate final assembly to Integration Architect
    6. Package and deliver final output
    """
    
    def __init__(self):
        super().__init__(
            name="EngineeringManager",
            description="Central orchestrator managing the AgentForge workflow"
        )
        self.workflow_history: List[Dict[str, Any]] = []
        self.current_step: Optional[WorkflowStep] = None
        
    async def process(self, input_data: InputGoal) -> TeamPackage:
        """Main orchestration workflow."""
        
        # Step 1: Goal Intake
        self.current_step = WorkflowStep.GOAL_INTAKE
        await self._log_step("Received input goal", {"goal": input_data.goal_description})
        
        # Step 2: Delegate to Systems Analyst
        self.current_step = WorkflowStep.STRATEGY_ANALYSIS
        strategy_document = await self._delegate_to_systems_analyst(input_data)
        
        # Step 3: Delegate to Talent Scout
        self.current_step = WorkflowStep.RESOURCE_SCOUTING
        scouting_report = await self._delegate_to_talent_scout(strategy_document)
        
        # Step 4: Delegate to Agent Developer (if gaps exist)
        self.current_step = WorkflowStep.AGENT_DEVELOPMENT
        new_agents = await self._delegate_to_agent_developer(scouting_report)
        
        # Step 5: Delegate to Integration Architect
        self.current_step = WorkflowStep.TEAM_INTEGRATION
        roster_docs = await self._delegate_to_integration_architect(
            strategy_document, scouting_report, new_agents
        )
        
        # Step 6: Package Final Output
        self.current_step = WorkflowStep.FINAL_PACKAGING
        team_package = await self._create_final_package(
            input_data, strategy_document, scouting_report, 
            new_agents, roster_docs
        )
        
        await self._log_step("Workflow completed", {"package_created": True})
        return team_package
    
    async def _delegate_to_systems_analyst(self, goal: InputGoal) -> str:
        """Delegate analysis to Systems Analyst."""
        await self._log_step("Delegating to Systems Analyst", {"goal": goal.goal_description})
        
        # TODO: Integrate with actual Systems Analyst agent
        # For now, return a placeholder that would come from the Systems Analyst
        strategy_prompt = f"""
        Analyze the goal: {goal.goal_description}
        Domain: {goal.domain}
        Complexity: {goal.complexity_level}
        Create a comprehensive strategy document defining the ideal team structure.
        """
        
        strategy_response = await self.run_with_mcp(strategy_prompt)
        return strategy_response
    
    async def _delegate_to_talent_scout(self, strategy_document: str) -> str:
        """Delegate resource analysis to Talent Scout."""
        await self._log_step("Delegating to Talent Scout", {"strategy_received": True})
        
        # TODO: Integrate with actual Talent Scout agent
        scout_prompt = f"""
        Analyze this strategy document and identify matches from existing agent library:
        
        {strategy_document}
        
        Cross-reference with agent library at /home/delorenj/code/DeLoDocs/AI/Agents
        Provide scouting report with matches and gaps.
        """
        
        scouting_response = await self.run_with_mcp(scout_prompt)
        return scouting_response
    
    async def _delegate_to_agent_developer(self, scouting_report: str) -> List[Dict[str, Any]]:
        """Delegate agent creation to Agent Developer."""
        await self._log_step("Delegating to Agent Developer", {"scouting_report_received": True})
        
        # TODO: Integrate with actual Agent Developer agent
        developer_prompt = f"""
        Based on this scouting report, create new agents for identified gaps:
        
        {scouting_report}
        
        Create comprehensive agent prompts for each gap identified.
        """
        
        developer_response = await self.run_with_mcp(developer_prompt)
        
        # Parse response into structured format
        # This would be more sophisticated in actual implementation
        new_agents = [{"name": "PlaceholderAgent", "prompt": developer_response, "type": "general"}]
        return new_agents
    
    async def _delegate_to_integration_architect(
        self, strategy_document: str, scouting_report: str, new_agents: List[Dict[str, Any]]
    ) -> str:
        """Delegate final assembly to Integration Architect."""
        await self._log_step("Delegating to Integration Architect", {"components_ready": True})
        
        # TODO: Integrate with actual Integration Architect agent
        architect_prompt = f"""
        Create final roster documentation and operational workflow:
        
        Strategy Document: {strategy_document}
        Scouting Report: {scouting_report}
        New Agents: {len(new_agents)} agents created
        
        Define communication protocols, workflows, and operational procedures.
        """
        
        roster_response = await self.run_with_mcp(architect_prompt)
        return roster_response
    
    async def _create_final_package(
        self, goal: InputGoal, strategy: str, scouting: str, 
        new_agents: List[Dict[str, Any]], roster: str
    ) -> TeamPackage:
        """Package all components into final deliverable."""
        
        deployment_instructions = f"""
        # Deployment Instructions for AgentForge Team
        
        ## Goal
        {goal.goal_description}
        
        ## Team Composition
        - New Agents: {len(new_agents)}
        - Strategy: Complete
        - Roster Documentation: Ready
        
        ## Next Steps
        1. Review roster documentation
        2. Deploy agents according to specifications
        3. Execute team workflow as defined
        """
        
        return TeamPackage(
            goal=goal,
            strategy_document=strategy,
            scouting_report=scouting,
            new_agents=new_agents,
            existing_agents=[],  # TODO: Extract from scouting report
            roster_documentation=roster,
            deployment_instructions=deployment_instructions
        )
    
    async def _log_step(self, action: str, details: Dict[str, Any]):
        """Log workflow step for tracking."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step.value if self.current_step else "unknown",
            "action": action,
            "details": details
        }
        self.workflow_history.append(log_entry)
        
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "current_step": self.current_step.value if self.current_step else None,
            "history_length": len(self.workflow_history),
            "latest_action": self.workflow_history[-1] if self.workflow_history else None
        }


# Example usage and testing
async def main():
    """Example usage of Engineering Manager."""
    
    # Create the orchestrator
    em = EngineeringManager()
    
    # Define a sample goal
    sample_goal = InputGoal(
        goal_description="Create a web application for task management with real-time collaboration",
        domain="web development",
        complexity_level=ComplexityLevel.HIGH,
        timeline="3 months",
        constraints=["Must use React", "Must be mobile-friendly", "Budget: $50k"],
        success_criteria=["User authentication", "Real-time updates", "Task assignment", "Mobile responsive"]
    )
    
    print("🚀 Starting AgentForge Orchestration")
    print(f"Goal: {sample_goal.goal_description}")
    print(f"Domain: {sample_goal.domain}")
    print(f"Complexity: {sample_goal.complexity_level}")
    
    try:
        # Process the goal through the complete workflow
        result = await em.process(sample_goal)
        
        print("\n✅ AgentForge Orchestration Complete!")
        print(f"Created at: {result.created_at}")
        print(f"New agents created: {len(result.new_agents)}")
        print(f"Strategy document: {len(result.strategy_document)} characters")
        print(f"Roster documentation: {len(result.roster_documentation)} characters")
        
        # Show workflow status
        status = em.get_workflow_status()
        print(f"\nWorkflow completed {status['history_length']} steps")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in orchestration: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())