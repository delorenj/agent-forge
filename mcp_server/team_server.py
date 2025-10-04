#!/usr/bin/env python3
"""
AgentForge Team MCP Server

Exposes the AgentForge meta-team as MCP tools for use in Claude Code, Cline, and other MCP clients.

Each agent in the meta-team is available as a specialized tool:
- agentforge_create_team: Full workflow (Engineering Manager orchestrates all agents)
- agentforge_analyze_strategy: Strategic analysis (Systems Analyst)
- agentforge_scout_agents: Agent discovery and matching (Talent Scout)
- agentforge_develop_agents: Create new agents (Agent Developer)
- agentforge_integrate_team: Final assembly (Integration Architect)
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel
from agents.systems_analyst import SystemsAnalyst, InputGoal as AnalystInputGoal
from agents.talent_scout import TalentScout, TalentScoutInput, StrategyDocument, RoleRequirement

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server/agentforge_team.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the meta-team (singleton pattern)
em = EngineeringManager()
systems_analyst = SystemsAnalyst()
talent_scout = TalentScout()

# Track initialized state
_initialized = False

async def initialize_services():
    """Initialize all services (Talent Scout needs QDrant setup)"""
    global _initialized
    if not _initialized:
        logger.info("Initializing AgentForge Team MCP Server...")
        await talent_scout.initialize()
        _initialized = True
        logger.info("AgentForge Team MCP Server initialized successfully!")


# Define MCP Tools
TEAM_TOOLS = [
    Tool(
        name="agentforge_create_team",
        description="""🚀 FULL TEAM CREATION WORKFLOW - The complete AgentForge experience!

        Orchestrates the entire meta-team to analyze your goal, scout existing agents,
        create new specialized agents for gaps, and deliver a complete ready-to-deploy team.

        This is the main entry point - Engineering Manager coordinates all other agents.

        Returns:
        - Complete team roster with matched + newly created agents
        - Deployment instructions and operational playbook
        - Quality metrics and reuse efficiency analysis

        Perfect for: "Build me a team to create [X]" requests""",
        inputSchema={
            "type": "object",
            "properties": {
                "goal_description": {
                    "type": "string",
                    "description": "High-level goal description (e.g., 'Build a real-time chat application')"
                },
                "domain": {
                    "type": "string",
                    "description": "Primary domain (e.g., 'web development', 'data science', 'devops')"
                },
                "complexity_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "enterprise"],
                    "default": "medium",
                    "description": "Project complexity level"
                },
                "timeline": {
                    "type": "string",
                    "description": "Expected timeline (e.g., '3 months', '6 weeks')"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any constraints or limitations (e.g., 'Must use React', 'Budget: $50k')"
                },
                "success_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How to measure success (e.g., 'User authentication', '99.9% uptime')"
                }
            },
            "required": ["goal_description", "domain"]
        }
    ),

    Tool(
        name="agentforge_analyze_strategy",
        description="""📊 STRATEGIC ANALYSIS - Systems Analyst specialization

        Deep dive analysis of a goal to determine ideal team composition.
        Uses reasoning to break down complex goals into roles and capabilities.

        Returns detailed strategy document with:
        - Core capability requirements
        - Recommended roles and their specifications
        - Interaction patterns and workflows
        - Timeline estimates and risk analysis

        Perfect for: Planning phase before committing to full team creation""",
        inputSchema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Goal description to analyze"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the goal"
                },
                "domain": {
                    "type": "string",
                    "description": "Primary domain"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "enterprise"],
                    "default": "medium"
                },
                "success_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Success criteria for the goal"
                }
            },
            "required": ["description", "domain"]
        }
    ),

    Tool(
        name="agentforge_scout_agents",
        description="""🔍 AGENT DISCOVERY - Talent Scout specialization

        Semantic search across agent libraries using QDrant vector database.
        Finds existing agents that match role requirements and identifies gaps.

        Returns comprehensive scouting report with:
        - Matched agents with similarity scores and adaptation suggestions
        - Vacant roles requiring new agent creation
        - Reuse efficiency metrics and gap analysis

        Perfect for: Understanding what agents you already have before creating new ones""",
        inputSchema={
            "type": "object",
            "properties": {
                "strategy_title": {
                    "type": "string",
                    "description": "Strategy document title"
                },
                "goal_description": {
                    "type": "string",
                    "description": "Original goal being addressed"
                },
                "domain": {
                    "type": "string",
                    "description": "Primary domain"
                },
                "complexity_level": {
                    "type": "string",
                    "default": "medium"
                },
                "roles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role_id": {"type": "string"},
                            "role_name": {"type": "string"},
                            "description": {"type": "string"},
                            "required_capabilities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "preferred_capabilities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "domain": {"type": "string"},
                            "complexity_level": {"type": "string"},
                            "priority": {"type": "string", "default": "medium"}
                        },
                        "required": ["role_id", "role_name", "description", "required_capabilities", "domain", "complexity_level"]
                    },
                    "description": "Roles to find agents for"
                },
                "agent_libraries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to agent libraries (defaults to standard locations)"
                },
                "force_reindex": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force re-indexing of agent libraries"
                }
            },
            "required": ["strategy_title", "goal_description", "domain", "complexity_level", "roles"]
        }
    ),

    Tool(
        name="agentforge_develop_agents",
        description="""🛠️ AGENT CREATION - Agent Developer specialization

        Creates new specialized Agno agents for identified gaps.
        Master prompt engineering with comprehensive agent specifications.

        For each vacant role, generates:
        - Complete system prompts and instructions
        - Tool configurations and model recommendations
        - Python initialization code (runnable Agno agents!)
        - Test suites and validation checks
        - Full documentation and usage examples

        Perfect for: Creating agents when no suitable matches exist in library""",
        inputSchema={
            "type": "object",
            "properties": {
                "vacant_roles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role_name": {"type": "string"},
                            "title": {"type": "string"},
                            "core_responsibilities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "required_capabilities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "interaction_patterns": {
                                "type": "object",
                                "additionalProperties": {"type": "string"}
                            },
                            "success_metrics": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "priority_level": {"type": "string"},
                            "domain_context": {"type": "string"},
                            "complexity_level": {"type": "string", "default": "medium"}
                        },
                        "required": ["role_name", "title", "core_responsibilities", "required_capabilities", "success_metrics", "priority_level"]
                    },
                    "description": "Vacant roles needing new agents"
                },
                "capability_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Overall capability gaps across the team"
                }
            },
            "required": ["vacant_roles", "capability_gaps"]
        }
    ),

    Tool(
        name="agentforge_quick_agent",
        description="""⚡ QUICK AGENT CREATION - Fast single agent generation

        Simplified interface for creating a single agent without full workflow.
        Great for quick prototyping or one-off agent needs.

        Returns complete agent specification ready to use.

        Perfect for: "I just need a quick [X] agent" requests""",
        inputSchema={
            "type": "object",
            "properties": {
                "role_name": {
                    "type": "string",
                    "description": "Name of the agent role (e.g., 'Data Analyst', 'API Developer')"
                },
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required capabilities for this agent"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium",
                    "description": "Priority level"
                }
            },
            "required": ["role_name", "capabilities"]
        }
    ),

    Tool(
        name="agentforge_get_workflow_status",
        description="""📈 WORKFLOW STATUS - Check Engineering Manager progress

        Get current status of the meta-team workflow including:
        - Current workflow step
        - History of completed steps
        - Latest action details

        Perfect for: Debugging or understanding workflow progress""",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),

    Tool(
        name="agentforge_reindex_libraries",
        description="""🔄 REINDEX AGENT LIBRARIES - Force refresh of agent index

        Force re-indexing of agent libraries in QDrant vector database.
        Use when you've added new agents or updated existing ones.

        Returns indexing statistics.

        Perfect for: After adding new agents to your library""",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_libraries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to agent libraries (defaults to standard locations)"
                }
            }
        }
    )
]


async def handle_create_team(arguments: Dict[str, Any]) -> str:
    """Handle full team creation workflow"""
    logger.info(f"Creating team for goal: {arguments.get('goal_description')}")

    try:
        # Convert complexity level string to enum
        complexity_str = arguments.get('complexity_level', 'medium')
        complexity_level = ComplexityLevel(complexity_str)

        # Create InputGoal
        goal = InputGoal(
            goal_description=arguments['goal_description'],
            domain=arguments['domain'],
            complexity_level=complexity_level,
            timeline=arguments.get('timeline'),
            constraints=arguments.get('constraints', []),
            success_criteria=arguments.get('success_criteria', [])
        )

        # Process through Engineering Manager
        result = await em.process(goal)

        # Format response
        response = {
            "status": "success",
            "team_package": {
                "goal": goal.model_dump(),
                "new_agents_count": len(result.new_agents),
                "existing_agents_count": len(result.existing_agents),
                "total_team_size": len(result.new_agents) + len(result.existing_agents),
                "new_agents": result.new_agents,
                "existing_agents": result.existing_agents,
                "roster_documentation": result.roster_documentation,
                "deployment_instructions": result.deployment_instructions,
                "created_at": result.created_at.isoformat()
            },
            "metrics": {
                "reuse_efficiency": f"{(len(result.existing_agents) / (len(result.new_agents) + len(result.existing_agents)) * 100) if (len(result.new_agents) + len(result.existing_agents)) > 0 else 0:.1f}%",
                "coverage": f"{result.scouting_report.overall_coverage:.1%}",
                "total_roles": result.scouting_report.total_roles,
                "filled_by_existing": result.scouting_report.filled_roles,
                "newly_created": result.scouting_report.vacant_roles
            }
        }

        logger.info(f"Team created successfully: {len(result.new_agents)} new + {len(result.existing_agents)} existing agents")
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error creating team: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_analyze_strategy(arguments: Dict[str, Any]) -> str:
    """Handle strategic analysis"""
    logger.info(f"Analyzing strategy for: {arguments.get('description')}")

    try:
        # Create Systems Analyst input
        analyst_input = AnalystInputGoal(
            description=arguments['description'],
            context=arguments.get('context'),
            domain=arguments['domain'],
            complexity=arguments.get('complexity', 'medium'),
            success_criteria=arguments.get('success_criteria', [])
        )

        # Process with Systems Analyst
        result = await systems_analyst.process(analyst_input)

        if result.status == "success":
            response = {
                "status": "success",
                "strategy_document": result.result.model_dump(),
                "metadata": result.metadata
            }
            logger.info("Strategy analysis completed successfully")
            return json.dumps(response, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": f"Strategy analysis failed: {result.result}"
            }, indent=2)

    except Exception as e:
        logger.error(f"Error in strategy analysis: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_scout_agents(arguments: Dict[str, Any]) -> str:
    """Handle agent scouting"""
    logger.info(f"Scouting agents for: {arguments.get('strategy_title')}")

    try:
        # Build StrategyDocument from arguments
        roles = []
        for role_data in arguments['roles']:
            role = RoleRequirement(
                role_id=role_data['role_id'],
                role_name=role_data['role_name'],
                description=role_data['description'],
                required_capabilities=role_data['required_capabilities'],
                preferred_capabilities=role_data.get('preferred_capabilities', []),
                domain=role_data['domain'],
                complexity_level=role_data['complexity_level'],
                priority=role_data.get('priority', 'medium')
            )
            roles.append(role)

        strategy_doc = StrategyDocument(
            title=arguments['strategy_title'],
            goal_description=arguments['goal_description'],
            domain=arguments['domain'],
            complexity_level=arguments['complexity_level'],
            roles=roles,
            timeline=arguments.get('timeline'),
            constraints=arguments.get('constraints', [])
        )

        # Create Talent Scout input
        scout_input = TalentScoutInput(
            goal=f"Scout agents for: {arguments['goal_description']}",
            strategy_document=strategy_doc,
            agent_libraries=arguments.get('agent_libraries', [
                "/home/delorenj/code/DeLoDocs/AI/Agents",
                "/home/delorenj/code/DeLoDocs/AI/Teams"
            ]),
            force_reindex=arguments.get('force_reindex', False)
        )

        # Process with Talent Scout
        result = await talent_scout.process(scout_input)

        if result.status == "success":
            response = {
                "status": "success",
                "scouting_report": result.scouting_report.model_dump(),
                "indexing_stats": result.indexing_stats,
                "performance_metrics": result.performance_metrics
            }
            logger.info(f"Scouting completed: {result.scouting_report.filled_roles} matches, {result.scouting_report.vacant_roles} gaps")
            return json.dumps(response, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": f"Scouting failed: {result.result}"
            }, indent=2)

    except Exception as e:
        logger.error(f"Error in agent scouting: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_develop_agents(arguments: Dict[str, Any]) -> str:
    """Handle agent development"""
    logger.info(f"Developing agents for {len(arguments.get('vacant_roles', []))} vacant roles")

    try:
        from agents.agent_developer import AgentDeveloper, VacantRole, ScoutingReport

        # Convert vacant_roles to VacantRole objects
        vacant_roles = []
        for role_data in arguments['vacant_roles']:
            vacant_role = VacantRole(
                role_name=role_data['role_name'],
                title=role_data['title'],
                core_responsibilities=role_data['core_responsibilities'],
                required_capabilities=role_data['required_capabilities'],
                interaction_patterns=role_data.get('interaction_patterns', {}),
                success_metrics=role_data['success_metrics'],
                priority_level=role_data['priority_level'],
                domain_context=role_data.get('domain_context'),
                complexity_level=role_data.get('complexity_level', 'medium')
            )
            vacant_roles.append(vacant_role)

        # Create ScoutingReport
        scouting_report = ScoutingReport(
            matched_agents=[],
            vacant_roles=vacant_roles,
            capability_gaps=arguments.get('capability_gaps', []),
            reuse_analysis={},
            priority_recommendations=[role.role_name for role in vacant_roles]
        )

        # Create Agent Developer and develop agents
        developer = AgentDeveloper()
        result = await developer.develop_agents(scouting_report)

        response = {
            "status": "success" if result.success else "error",
            "agents_created": [agent.model_dump() for agent in result.agents_created],
            "generation_summary": result.generation_summary,
            "recommendations": result.recommendations,
            "generated_files": result.generated_files,
            "documentation": result.documentation,
            "integration_notes": result.integration_notes,
            "quality_score": result.quality_score,
            "validation_results": result.validation_results,
            "processing_time": result.processing_time
        }

        logger.info(f"Agent development completed: {len(result.agents_created)} agents created")
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error in agent development: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_quick_agent(arguments: Dict[str, Any]) -> str:
    """Handle quick agent creation"""
    logger.info(f"Quick creating agent: {arguments.get('role_name')}")

    try:
        from agents.agent_developer import AgentDeveloper

        developer = AgentDeveloper()
        agent_spec = await developer.quick_agent_creation(
            role_name=arguments['role_name'],
            capabilities=arguments['capabilities'],
            priority=arguments.get('priority', 'medium')
        )

        response = {
            "status": "success",
            "agent": agent_spec.model_dump(),
            "message": f"Created {agent_spec.name} agent successfully"
        }

        logger.info(f"Quick agent created: {agent_spec.name}")
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error in quick agent creation: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_workflow_status(arguments: Dict[str, Any]) -> str:
    """Get workflow status"""
    try:
        status = em.get_workflow_status()
        response = {
            "status": "success",
            "workflow_status": status
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


async def handle_reindex_libraries(arguments: Dict[str, Any]) -> str:
    """Force reindex agent libraries"""
    logger.info("Reindexing agent libraries")

    try:
        libraries = arguments.get('agent_libraries', [
            "/home/delorenj/code/DeLoDocs/AI/Agents",
            "/home/delorenj/code/DeLoDocs/AI/Teams"
        ])

        indexing_stats = await talent_scout.index_agent_libraries(libraries, force_reindex=True)

        response = {
            "status": "success",
            "indexing_stats": indexing_stats,
            "message": f"Indexed {indexing_stats.get('agents_indexed', 0)} agents from {indexing_stats.get('libraries_scanned', 0)} libraries"
        }

        logger.info(f"Reindexing completed: {indexing_stats}")
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error reindexing libraries: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# Main MCP Server Implementation
async def main():
    """Run the AgentForge Team MCP Server"""

    # Initialize services
    await initialize_services()

    # Create server
    server = Server("agentforge-team")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available AgentForge tools"""
        logger.debug("Listing available tools")
        return TEAM_TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        """Handle tool calls"""
        logger.info(f"Tool called: {name}")
        logger.debug(f"Arguments: {arguments}")

        try:
            # Route to appropriate handler
            if name == "agentforge_create_team":
                result = await handle_create_team(arguments)
            elif name == "agentforge_analyze_strategy":
                result = await handle_analyze_strategy(arguments)
            elif name == "agentforge_scout_agents":
                result = await handle_scout_agents(arguments)
            elif name == "agentforge_develop_agents":
                result = await handle_develop_agents(arguments)
            elif name == "agentforge_quick_agent":
                result = await handle_quick_agent(arguments)
            elif name == "agentforge_get_workflow_status":
                result = await handle_workflow_status(arguments)
            elif name == "agentforge_reindex_libraries":
                result = await handle_reindex_libraries(arguments)
            else:
                result = json.dumps({
                    "status": "error",
                    "error": f"Unknown tool: {name}"
                }, indent=2)

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}", exc_info=True)
            error_response = json.dumps({
                "status": "error",
                "error": str(e),
                "tool": name
            }, indent=2)
            return [TextContent(type="text", text=error_response)]

    # Run server with stdio transport
    logger.info("Starting AgentForge Team MCP Server on stdio...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
