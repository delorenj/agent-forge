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
from .systems_analyst import SystemsAnalyst, StrategyDocument
from .talent_scout import TalentScout, TalentScoutInput, ScoutingReport


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
    scouting_report: ScoutingReport
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
        
        # Initialize specialist agents
        self.systems_analyst = SystemsAnalyst()
        self.talent_scout = TalentScout()
        
        # Configuration for agent libraries (customizable via environment)
        import os
        default_libraries = [
            "/home/delorenj/code/DeLoDocs/AI/Agents",
            "/home/delorenj/code/DeLoDocs/AI/Teams"
        ]
        
        # Allow customization via environment variables
        custom_libraries = os.getenv('AGENT_LIBRARIES')
        if custom_libraries:
            self.agent_libraries = custom_libraries.split(',')
        else:
            self.agent_libraries = default_libraries
        
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
        scouting_report = await self._delegate_to_talent_scout(input_data, strategy_document)
        
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
    
    async def _delegate_to_systems_analyst(self, goal: InputGoal) -> StrategyDocument:
        """Delegate analysis to Systems Analyst."""
        await self._log_step("Delegating to Systems Analyst", {"goal": goal.goal_description})
        
        try:
            # Use actual Systems Analyst agent
            from .systems_analyst import InputGoal as AnalystInputGoal
            
            analyst_input = AnalystInputGoal(
                description=goal.goal_description,
                context=', '.join(goal.constraints) if goal.constraints else None,
                success_criteria=goal.success_criteria,
                domain=goal.domain,
                complexity=goal.complexity_level.value
            )
            
            result = await self.systems_analyst.process(analyst_input)
            
            if result.status == "success":
                await self._log_step("Systems Analyst completed", {"strategy_created": True})
                return result.result
            else:
                raise Exception(f"Systems Analyst failed: {result.result}")
                
        except Exception as e:
            await self._log_step("Systems Analyst fallback", {"error": str(e)})
            
            # Fallback to direct LLM approach
            strategy_prompt = f"""
            Analyze the goal: {goal.goal_description}
            Domain: {goal.domain}
            Complexity: {goal.complexity_level}
            Timeline: {goal.timeline or 'Not specified'}
            Constraints: {', '.join(goal.constraints) if goal.constraints else 'None'}
            Success Criteria: {', '.join(goal.success_criteria) if goal.success_criteria else 'None'}
            
            Create a comprehensive strategy document defining the ideal team structure with specific roles and their requirements.
            """
            
            strategy_response = await self.run_with_mcp(strategy_prompt)
            
            # Create a basic StrategyDocument from the response
            # In practice, you'd parse the response more carefully
            from .talent_scout import StrategyDocument, RoleRequirement
            
            return StrategyDocument(
                title=f"Strategy for {goal.goal_description}",
                goal_description=goal.goal_description,
                domain=goal.domain,
                complexity_level=goal.complexity_level.value,
                roles=[],  # Would need to parse from strategy_response
                timeline=goal.timeline,
                constraints=goal.constraints or []
            )
    
    async def _delegate_to_talent_scout(self, goal: InputGoal, strategy_document: StrategyDocument) -> ScoutingReport:
        """Delegate resource analysis to Enhanced Talent Scout."""
        from .talent_scout import StrategyDocument as TalentScoutStrategyDocument, RoleRequirement

        roles = []
        for role in strategy_document.team_composition:
            roles.append(RoleRequirement(
                role_id=role.name,
                role_name=role.name,
                description=", ".join(role.core_responsibilities),
                required_capabilities=role.required_capabilities,
                domain=strategy_document.goal_analysis.get("domain"),
                complexity_level=strategy_document.goal_analysis.get("complexity_level"),
                priority=role.priority_level,
            ))

        talent_scout_strategy_document = TalentScoutStrategyDocument(
            title=strategy_document.goal_analysis.get("primary_goal"),
            goal_description=strategy_document.goal_analysis.get("primary_goal"),
            domain=strategy_document.goal_analysis.get("domain"),
            complexity_level=strategy_document.goal_analysis.get("complexity_level"),
            roles=roles,
            timeline=strategy_document.timeline_estimate.get("total_duration"),
            constraints=goal.constraints,
        )

        await self._log_step("Delegating to Enhanced Talent Scout", {
            "strategy_title": talent_scout_strategy_document.title,
            "total_roles": len(talent_scout_strategy_document.roles),
            "agent_libraries": len(self.agent_libraries)
        })
        
        try:
            # Initialize Talent Scout if needed
            await self.talent_scout.initialize()
            
            # Create Talent Scout input
            scout_input = TalentScoutInput(
                goal=f"Scout agents for: {goal.goal_description}",
                strategy_document=talent_scout_strategy_document,
                agent_libraries=self.agent_libraries,
                force_reindex=False  # Only reindex if explicitly requested
            )
            
            # Process with Enhanced Talent Scout
            result = await self.talent_scout.process(scout_input)
            
            if result.status == "success":
                scouting_report = result.scouting_report
                
                await self._log_step("Enhanced Talent Scout completed", {
                    "total_roles": scouting_report.total_roles,
                    "filled_roles": scouting_report.filled_roles,
                    "vacant_roles": scouting_report.vacant_roles,
                    "reuse_efficiency": f"{scouting_report.reuse_efficiency:.1%}",
                    "processing_time_ms": scouting_report.processing_time_ms
                })
                
                return scouting_report
            else:
                raise Exception(f"Talent Scout failed: {result.result}")
                
        except Exception as e:
            await self._log_step("Talent Scout error", {"error": str(e)})
            
            # Return empty scouting report on failure
            from .talent_scout import ScoutingReport
            return ScoutingReport(
                strategy_title=strategy_document.title,
                total_roles=len(strategy_document.roles),
                filled_roles=0,
                vacant_roles=len(strategy_document.roles),
                matches=[],
                vacant_positions=[],
                overall_coverage=0.0,
                reuse_efficiency=0.0,
                processing_time_ms=0.0
            )
    
    async def _delegate_to_agent_developer(self, scouting_report: ScoutingReport) -> List[Dict[str, Any]]:
        """Delegate agent creation to Agent Developer."""
        await self._log_step("Delegating to Agent Developer", {
            "vacant_positions": len(scouting_report.vacant_positions),
            "needs_new_agents": len(scouting_report.vacant_positions) > 0
        })
        
        if len(scouting_report.vacant_positions) == 0:
            await self._log_step("No new agents needed", {"all_roles_filled": True})
            return []
        
        try:
            # TODO: Integrate with actual Agent Developer agent when available
            # For now, create placeholder agents based on vacant positions
            
            new_agents = []
            for vacant_position in scouting_report.vacant_positions:
                role = vacant_position.role_requirement
                
                # Create agent specification based on vacant role
                agent_spec = {
                    "id": role.role_id,
                    "name": role.role_name,
                    "role": role.role_name,
                    "description": role.description,
                    "required_capabilities": role.required_capabilities,
                    "preferred_capabilities": role.preferred_capabilities,
                    "domain": role.domain,
                    "complexity_level": role.complexity_level,
                    "priority": role.priority,
                    "gap_analysis": vacant_position.gap_analysis,
                    "creation_recommendations": vacant_position.creation_recommendations,
                    "inspiration_agents": [
                        {
                            "name": match.agent.name,
                            "score": match.overall_score,
                            "capabilities": match.agent.capabilities
                        }
                        for match in vacant_position.closest_matches
                    ]
                }
                
                new_agents.append(agent_spec)
            
            await self._log_step("Agent specifications created", {
                "new_agents_count": len(new_agents),
                "agent_names": [agent["name"] for agent in new_agents]
            })
            
            return new_agents
            
        except Exception as e:
            await self._log_step("Agent Developer error", {"error": str(e)})
            return []
    
    async def _delegate_to_integration_architect(
        self, strategy_document: StrategyDocument, scouting_report: ScoutingReport, new_agents: List[Dict[str, Any]]
    ) -> str:
        """Delegate final assembly to Integration Architect."""
        await self._log_step("Delegating to Integration Architect", {
            "matched_agents": len(scouting_report.matches),
            "new_agents": len(new_agents),
            "total_team_size": len(scouting_report.matches) + len(new_agents)
        })
        
        # TODO: Integrate with actual Integration Architect agent
        # For now, create comprehensive roster documentation
        
        roster_sections = []
        
        # Executive Summary
        roster_sections.append(f"""
# Team Roster Documentation
## {strategy_document.goal_analysis.get("primary_goal")}

### Executive Summary
- **Goal:** {strategy_document.goal_analysis.get("primary_goal")}
- **Domain:** {strategy_document.goal_analysis.get("domain")}
- **Complexity:** {strategy_document.goal_analysis.get("complexity_level")}
- **Total Roles:** {scouting_report.total_roles}
- **Reuse Efficiency:** {scouting_report.reuse_efficiency:.1%}
- **Team Coverage:** {scouting_report.overall_coverage:.1%}
        """)
        
        # Existing Agent Matches
        if scouting_report.matches:
            roster_sections.append("## Existing Agents (Matched)")
            for i, match in enumerate(scouting_report.matches, 1):
                adaptation_note = ""
                if match.adaptation_needed:
                    adaptation_note = f"\n   **Adaptations Required:** {'; '.join(match.adaptation_suggestions)}"
                
                roster_sections.append(f"""
### {i}. {match.role_requirement.role_name}
- **Agent:** {match.agent.name}
- **File:** {match.agent.file_path}
- **Match Score:** {match.overall_score:.3f} ({match.match_confidence} confidence)
- **Reasoning:** {match.match_reasoning}
- **Capabilities:** {', '.join(match.agent.capabilities)}
- **Tools:** {', '.join(match.agent.tools)}{adaptation_note}
                """)
        
        # New Agents to Create
        if new_agents:
            roster_sections.append("## New Agents (To Create)")
            for i, agent in enumerate(new_agents, 1):
                roster_sections.append(f"""
### {i}. {agent['name']}
- **Role:** {agent['role']}
- **Description:** {agent['description']}
- **Required Capabilities:** {', '.join(agent['required_capabilities'])}
- **Preferred Capabilities:** {', '.join(agent.get('preferred_capabilities', []))}
- **Domain:** {agent['domain']}
- **Priority:** {agent['priority']}
- **Gap Analysis:** {agent['gap_analysis']}

**Creation Recommendations:**
{chr(10).join(f'- {rec}' for rec in agent['creation_recommendations'])}
                """)
                
                if agent.get('inspiration_agents'):
                    roster_sections.append("**Inspiration from Similar Agents:**")
                    for insp in agent['inspiration_agents']:
                        roster_sections.append(f"- {insp['name']} (similarity: {insp['score']:.3f})")
        
        # Communication Protocols
        roster_sections.append("""
## Communication Protocols

### Inter-Agent Communication
1. **Primary Communication:** Structured data exchange via typed inputs/outputs
2. **Coordination:** Hierarchical reporting to Engineering Manager
3. **Knowledge Sharing:** Shared knowledge base and documentation
4. **Status Updates:** Regular progress reports and milestone tracking

### Workflow Coordination  
1. **Task Assignment:** Engineering Manager delegates to appropriate agents
2. **Dependency Management:** Agents communicate completion status upstream
3. **Quality Gates:** Each agent validates inputs and outputs
4. **Error Handling:** Escalation procedures for blocked or failed tasks

### Documentation Standards
1. **Input/Output Specifications:** All agents use typed Pydantic models
2. **Process Documentation:** Each agent maintains operation logs
3. **Knowledge Artifacts:** Persistent storage of decisions and learnings
4. **Version Control:** Track changes and evolution of agent capabilities
        """)
        
        # Deployment Instructions
        roster_sections.append(f"""
## Deployment Instructions

### Phase 1: Agent Preparation
1. **Existing Agents:** Verify access to {len(scouting_report.matches)} matched agents
2. **New Agents:** Create {len(new_agents)} new agent specifications
3. **Environment Setup:** Configure agent libraries and dependencies
4. **Testing:** Validate agent communications and capabilities

### Phase 2: Team Assembly
1. **Integration Testing:** Test agent interactions and workflows
2. **Performance Validation:** Ensure agents meet capability requirements
3. **Documentation Review:** Verify all communication protocols
4. **Deployment Validation:** Confirm team readiness for production

### Phase 3: Operational Launch
1. **Goal Execution:** Begin working toward: {strategy_document.goal_analysis.get("primary_goal")}
2. **Progress Monitoring:** Track team performance and goal achievement
3. **Continuous Improvement:** Iterate on agent capabilities and workflows
4. **Success Validation:** Measure against defined success criteria

### Success Criteria
{chr(10).join(f'- {criteria}' for criteria in strategy_document.goal_analysis.get("success_criteria", ['Success criteria not defined']))}
        """)
        
        roster_documentation = "\\n".join(roster_sections)
        
        await self._log_step("Roster documentation created", {
            "sections": len(roster_sections),
            "documentation_length": len(roster_documentation)
        })
        
        return roster_documentation
    
    async def _create_final_package(
        self, goal: InputGoal, strategy: StrategyDocument, scouting: ScoutingReport, 
        new_agents: List[Dict[str, Any]], roster: str
    ) -> TeamPackage:
        """Package all components into final deliverable."""
        
        # Extract existing agents from scouting report
        existing_agents = []
        for match in scouting.matches:
            existing_agents.append({
                "id": match.agent.id,
                "name": match.agent.name,
                "role": match.agent.role,
                "file_path": match.agent.file_path,
                "capabilities": match.agent.capabilities,
                "tools": match.agent.tools,
                "domain": match.agent.domain,
                "match_score": match.overall_score,
                "match_confidence": match.match_confidence,
                "adaptation_needed": match.adaptation_needed,
                "adaptation_suggestions": match.adaptation_suggestions
            })
        
        deployment_instructions = f"""
# AgentForge Deployment Guide

## Project Overview
**Goal:** {goal.goal_description}
**Domain:** {goal.domain}
**Complexity:** {goal.complexity_level.value}
**Timeline:** {goal.timeline or 'Not specified'}

## Team Composition Summary
- **Total Roles Required:** {scouting.total_roles}
- **Existing Agents Matched:** {scouting.filled_roles}
- **New Agents to Create:** {scouting.vacant_roles}
- **Team Coverage:** {scouting.overall_coverage:.1%}
- **Reuse Efficiency:** {scouting.reuse_efficiency:.1%}

## Deployment Checklist

### ✅ Pre-Deployment
- [ ] Review complete roster documentation
- [ ] Verify access to {len(existing_agents)} existing agents
- [ ] Create {len(new_agents)} new agents according to specifications
- [ ] Set up agent communication infrastructure
- [ ] Configure knowledge bases and shared resources

### ✅ Agent Integration
- [ ] Deploy existing agents with any required adaptations
- [ ] Integrate new agents into team workflow
- [ ] Test inter-agent communication protocols
- [ ] Validate capability matching for all roles
- [ ] Configure monitoring and logging systems

### ✅ Operational Readiness
- [ ] Execute integration tests across full agent team
- [ ] Validate against success criteria: {', '.join(goal.success_criteria) if goal.success_criteria else 'None defined'}
- [ ] Set up performance monitoring and metrics
- [ ] Establish escalation procedures for issues
- [ ] Document operational procedures and troubleshooting

### ✅ Launch & Monitoring
- [ ] Begin goal execution with full agent team
- [ ] Monitor progress against timeline: {goal.timeline or 'Flexible'}
- [ ] Track performance metrics and team effectiveness
- [ ] Iterate on agent capabilities based on real-world performance
- [ ] Document lessons learned for future AgentForge deployments

## Support & Troubleshooting
- **Agent Issues:** Check agent logs and capability matching
- **Communication Problems:** Validate typed input/output contracts
- **Performance Issues:** Review resource allocation and agent workload
- **Goal Alignment:** Ensure agents are working toward defined success criteria

## Success Metrics
- Goal completion within timeline constraints
- All success criteria achieved: {chr(10).join(f'  - {c}' for c in (goal.success_criteria or ['Not defined']))}
- Team efficiency and agent reuse optimization
- Knowledge transfer and organizational learning
        """
        
        await self._log_step("Final package created", {
            "existing_agents": len(existing_agents),
            "new_agents": len(new_agents), 
            "total_team_size": len(existing_agents) + len(new_agents),
            "coverage": f"{scouting.overall_coverage:.1%}",
            "reuse_efficiency": f"{scouting.reuse_efficiency:.1%}"
        })
        
        return TeamPackage(
            goal=goal,
            strategy_document=strategy.model_dump_json(indent=2),  # Serialize to JSON string
            scouting_report=scouting,
            new_agents=new_agents,
            existing_agents=existing_agents,
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