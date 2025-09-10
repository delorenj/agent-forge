# 🔧 AgentForge API Reference

Complete API documentation for AgentForge agents, data models, and interfaces.

## Table of Contents

- [Core Models](#core-models)
- [Engineering Manager API](#engineering-manager-api)
- [Systems Analyst API](#systems-analyst-api)
- [Talent Scout API](#talent-scout-api)
- [Agent Developer API](#agent-developer-api)
- [Integration Architect API](#integration-architect-api)
- [Workflow API](#workflow-api)
- [Utility Functions](#utility-functions)

## Core Models

### InputGoal

The primary input model for AgentForge operations.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ComplexityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    ENTERPRISE = "enterprise"

class InputGoal(BaseModel):
    """Structured input goal with validation requirements for AgentForge."""
    
    goal_description: str = Field(
        ..., 
        description="High-level description of what needs to be achieved",
        min_length=10,
        max_length=1000
    )
    
    domain: str = Field(
        ...,
        description="Primary domain or field (e.g., 'web development', 'data science')",
        min_length=2,
        max_length=100
    )
    
    complexity_level: ComplexityLevel = Field(
        default=ComplexityLevel.MEDIUM,
        description="Complexity level of the goal"
    )
    
    timeline: Optional[str] = Field(
        None,
        description="Expected timeline for completion (e.g., '2 weeks', '1 month')"
    )
    
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints or limitations to consider"
    )
    
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Specific criteria that define success"
    )
    
    existing_resources: List[str] = Field(
        default_factory=list,
        description="Any existing resources or tools that should be leveraged"
    )
```

**Usage Example:**
```python
goal = InputGoal(
    goal_description="Build a real-time analytics dashboard for e-commerce metrics",
    domain="data visualization",
    complexity_level=ComplexityLevel.HIGH,
    timeline="8 weeks",
    constraints=["React frontend", "Python backend", "Real-time updates"],
    success_criteria=["Sub-second data refresh", "Mobile responsive", "99.9% uptime"],
    existing_resources=["PostgreSQL database", "AWS infrastructure", "Existing API endpoints"]
)
```

### TeamPackage

The comprehensive output model containing the complete generated agent team.

```python
class TeamPackage(BaseModel):
    """Final deliverable containing the complete agent team and documentation."""
    
    team_name: str = Field(..., description="Name of the assembled team")
    goal_summary: str = Field(..., description="Summary of the original goal")
    created_at: datetime = Field(default_factory=datetime.now, description="When team was created")
    
    # Team composition
    team_members: List[Dict[str, Any]] = Field(
        ..., 
        description="List of agents with their roles and capabilities"
    )
    
    # Workflow documentation  
    workflow_steps: List[str] = Field(
        ...,
        description="Step-by-step workflow for the team to execute"
    )
    
    communication_protocols: Dict[str, str] = Field(
        ...,
        description="How agents should communicate and handoff tasks"
    )
    
    # Core documentation
    strategy_document: str = Field(..., description="Complete strategy analysis")
    scouting_report: str = Field(..., description="Resource analysis and agent matching")
    roster_documentation: str = Field(..., description="Final team operational playbook")
    
    # Implementation details
    new_agents: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Details of new agents that were created"
    )
    
    existing_agents: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Details of existing agents that were reused"
    )
    
    deployment_instructions: str = Field(
        ...,
        description="Instructions for deploying this team"
    )
    
    success_metrics: List[str] = Field(
        ...,
        description="How to measure the team's success"
    )
    
    # Metadata
    processing_time: float = Field(..., description="Time taken to generate team (seconds)")
    quality_score: float = Field(default=0.0, description="Overall quality score (0-1)")
```

## Engineering Manager API

The central orchestrator for the entire AgentForge workflow.

### Class: EngineeringManager

```python
class EngineeringManager:
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = "agentforge.db",
        mcp_url: str = None,
        knowledge_base_path: str = None
    ):
        """Initialize the Engineering Manager with required components."""
```

**Parameters:**
- `model_id` (str): Language model identifier (default: "deepseek/deepseek-v3.1")  
- `db_file` (str): SQLite database file path (default: "agentforge.db")
- `mcp_url` (str): MCP server URL for enhanced features (optional)
- `knowledge_base_path` (str): Path to knowledge base for embeddings (optional)

### Methods

#### `async process(input_goal: InputGoal) -> TeamPackage`

Primary method to process a goal and generate a complete agent team.

**Parameters:**
- `input_goal` (InputGoal): The goal specification to process

**Returns:**
- `TeamPackage`: Complete team package with all documentation and agents

**Raises:**
- `ValidationError`: If input_goal fails validation
- `ProcessingError`: If workflow fails at any step
- `APIError`: If external API calls fail

**Example:**
```python
em = EngineeringManager()
goal = InputGoal(
    goal_description="Build a customer support chatbot",
    domain="customer service"
)
result = await em.process(goal)
```

#### `async run_with_goal(goal_description: str, domain: str = "general", **kwargs) -> TeamPackage`

Convenience method for quick goal processing.

**Parameters:**
- `goal_description` (str): Simple description of the goal
- `domain` (str): Domain context (default: "general")
- `**kwargs`: Additional InputGoal fields

**Returns:**
- `TeamPackage`: Complete team package

**Example:**
```python
result = await em.run_with_goal(
    "Create a data analysis pipeline", 
    domain="data science",
    complexity_level="high",
    constraints=["Python", "Apache Airflow"]
)
```

#### `get_workflow_status(workflow_id: str = None) -> Dict[str, Any]`

Get current or historical workflow status.

**Parameters:**
- `workflow_id` (str, optional): Specific workflow ID to check

**Returns:**
- `Dict[str, Any]`: Workflow status and metadata

**Example:**
```python
status = em.get_workflow_status()
print(f"Current status: {status['status']}")
print(f"Steps completed: {len(status['steps'])}")
```

## Systems Analyst API

Specializes in goal decomposition and ideal team structure definition.

### Class: SystemsAnalyst

```python
class SystemsAnalyst:
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = "systems_analyst.db",
        knowledge_base_path: str = None
    ):
        """Initialize the Systems Analyst agent."""
```

### Methods

#### `async analyze_goal(input_goal: InputGoal) -> StrategyDocument`

Analyze an input goal and create comprehensive strategy document.

**Parameters:**
- `input_goal` (InputGoal): The goal to analyze

**Returns:**
- `StrategyDocument`: Detailed strategy analysis

**Example:**
```python
analyst = SystemsAnalyst()
strategy = await analyst.analyze_goal(goal)
print(f"Identified {len(strategy.team_composition)} required roles")
```

#### `async quick_analysis(goal_description: str) -> str`

Perform rapid goal analysis for simple requirements.

**Parameters:**
- `goal_description` (str): Simple goal description

**Returns:**
- `str`: Quick analysis summary

#### `create_strategy_document(strategy: StrategyDocument, filename: str) -> str`

Create a markdown strategy document file.

**Parameters:**
- `strategy` (StrategyDocument): Strategy data to document
- `filename` (str): Output filename

**Returns:**
- `str`: Path to created document

### Data Models

#### StrategyDocument

```python
class StrategyDocument(BaseModel):
    """Comprehensive strategy analysis output."""
    
    goal_analysis: Dict[str, Any] = Field(..., description="Detailed goal breakdown")
    team_composition: List[Dict[str, Any]] = Field(..., description="Required agent roles")
    team_structure: Dict[str, Any] = Field(..., description="Team organization patterns") 
    risk_assessment: List[str] = Field(..., description="Identified risks and mitigation")
    resource_requirements: Dict[str, Any] = Field(..., description="Resource needs")
    timeline_estimate: Dict[str, str] = Field(..., description="Timeline projections")
```

## Talent Scout API

Manages agent library analysis and pattern matching.

### Class: TalentScout

```python
class TalentScout:
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = "talent_scout.db",
        agent_library_path: str = None,
        knowledge_base_path: str = None
    ):
        """Initialize the Talent Scout agent."""
```

### Methods

#### `async scout_resources(strategy: StrategyDocument) -> ScoutingReport`

Analyze agent library and match existing agents to requirements.

**Parameters:**
- `strategy` (StrategyDocument): Strategy document with role requirements

**Returns:**
- `ScoutingReport`: Analysis of matches and gaps

#### `async scan_agent_library(library_path: str) -> Dict[str, Any]`

Scan and index an agent library for future matching.

**Parameters:**
- `library_path` (str): Path to agent library directory

**Returns:**
- `Dict[str, Any]`: Library scanning results and metadata

### Data Models

#### ScoutingReport

```python
class ScoutingReport(BaseModel):
    """Resource analysis and agent matching report."""
    
    matched_agents: List[Dict[str, Any]] = Field(..., description="Agents that match requirements")
    capability_gaps: List[Dict[str, Any]] = Field(..., description="Missing capabilities") 
    reuse_recommendations: Dict[str, str] = Field(..., description="Reuse suggestions")
    confidence_scores: Dict[str, float] = Field(..., description="Match confidence levels")
```

## Agent Developer API

Handles creation of new specialized agents for identified gaps.

### Class: AgentDeveloper

```python
class AgentDeveloper:
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1", 
        db_file: str = "agent_developer.db",
        template_library_path: str = None
    ):
        """Initialize the Agent Developer."""
```

### Methods

#### `async develop_agents(scouting_report: ScoutingReport, strategy_context: Dict) -> GenerationResult`

Create new agents for identified capability gaps.

**Parameters:**
- `scouting_report` (ScoutingReport): Gap analysis from Talent Scout
- `strategy_context` (Dict): Strategy context for agent creation

**Returns:**
- `GenerationResult`: Results of agent generation process

#### `async create_agent_specification(role: Dict[str, Any]) -> AgentSpecification`

Create detailed specification for a single agent role.

**Parameters:**
- `role` (Dict[str, Any]): Role requirements and context

**Returns:**
- `AgentSpecification`: Complete agent specification

### Data Models

#### GenerationResult

```python
class GenerationResult(BaseModel):
    """Results from agent generation process."""
    
    success: bool = Field(..., description="Whether generation succeeded")
    agents_created: List[AgentSpecification] = Field(..., description="Generated agent specs")
    generation_summary: str = Field(..., description="Summary of generation process")
    quality_score: float = Field(..., description="Overall quality score")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")
```

#### AgentSpecification

```python
class AgentSpecification(BaseModel):
    """Complete specification for a generated agent."""
    
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    title: str = Field(..., description="Agent title/specialization")
    system_prompt: str = Field(..., description="Complete system prompt")
    tools_required: List[str] = Field(..., description="Required tools")
    instructions: List[str] = Field(..., description="Operating instructions")
    success_criteria: List[str] = Field(..., description="Success metrics")
    collaboration_patterns: Dict[str, Any] = Field(..., description="How agent collaborates")
    initialization_code: str = Field(..., description="Python initialization code")
    example_usage: str = Field(..., description="Usage examples")
```

## Integration Architect API

Handles final team assembly and operational playbook creation.

### Class: IntegrationArchitect

```python
class IntegrationArchitect:
    def __init__(
        self,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = "integration_architect.db"
    ):
        """Initialize the Integration Architect."""
```

### Methods

#### `async integrate_team(strategy: StrategyDocument, scouting: ScoutingReport, new_agents: List[NewAgent], goal: str) -> RosterDocumentation`

Assemble final team and create operational documentation.

**Parameters:**
- `strategy` (StrategyDocument): Original strategy analysis
- `scouting` (ScoutingReport): Resource scouting results  
- `new_agents` (List[NewAgent]): Newly created agents
- `goal` (str): Original goal description

**Returns:**
- `RosterDocumentation`: Complete team documentation

#### `create_roster_documentation(roster: RosterDocumentation, team_name: str, filename: str) -> str`

Create markdown documentation file for the team.

**Parameters:**
- `roster` (RosterDocumentation): Roster data to document
- `team_name` (str): Name of the team
- `filename` (str): Output filename

**Returns:**
- `str`: Path to created documentation file

### Data Models

#### RosterDocumentation

```python
class RosterDocumentation(BaseModel):
    """Complete operational documentation for assembled team."""
    
    team_overview: str = Field(..., description="Team purpose and scope")
    agent_roster: List[Dict[str, Any]] = Field(..., description="Complete agent list")
    workflow_definition: str = Field(..., description="Operational workflow")
    communication_protocols: List[str] = Field(..., description="Inter-agent communication")
    deployment_checklist: List[str] = Field(..., description="Deployment steps")
    monitoring_guidelines: List[str] = Field(..., description="How to monitor team")
    troubleshooting_guide: str = Field(..., description="Common issues and solutions")
```

## Workflow API

Advanced workflow management and orchestration features.

### Class: AgentForgeWorkflow

```python
class AgentForgeWorkflow:
    def __init__(
        self,
        db: Optional[Database] = None,
        session_id: str = None
    ):
        """Initialize workflow with database and session management."""
```

### Methods

#### `async run(input_goal: InputGoal) -> TeamPackage`

Execute the complete AgentForge workflow.

**Parameters:**
- `input_goal` (InputGoal): Goal to process

**Returns:**  
- `TeamPackage`: Complete team package

#### `async get_session_state(session_id: str) -> Dict[str, Any]`

Retrieve workflow session state.

**Parameters:**
- `session_id` (str): Session identifier

**Returns:**
- `Dict[str, Any]`: Session state data

#### `async save_checkpoint(session_id: str, step_name: str, data: Any) -> bool`

Save workflow checkpoint for recovery.

**Parameters:**
- `session_id` (str): Session identifier
- `step_name` (str): Current workflow step
- `data` (Any): Data to checkpoint

**Returns:**
- `bool`: Success indicator

## Utility Functions

### Factory Functions

#### `create_orchestrator(**kwargs) -> EngineeringManager`

Create configured Engineering Manager instance.

**Parameters:**
- `**kwargs`: Configuration options

**Returns:**
- `EngineeringManager`: Configured orchestrator

**Example:**
```python
em = create_orchestrator(
    model_id="anthropic/claude-3-haiku",
    db_file="custom.db"
)
```

### Validation Functions

#### `validate_goal(goal: InputGoal) -> bool`

Validate goal completeness and feasibility.

**Parameters:**
- `goal` (InputGoal): Goal to validate

**Returns:**
- `bool`: Validation result

#### `estimate_complexity(goal_description: str) -> ComplexityLevel`

Auto-estimate complexity level from goal description.

**Parameters:**
- `goal_description` (str): Goal description text

**Returns:**
- `ComplexityLevel`: Estimated complexity

### Configuration Functions

#### `load_config(config_path: str) -> Dict[str, Any]`

Load AgentForge configuration from file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration data

#### `validate_environment() -> Dict[str, bool]`

Validate required environment variables and dependencies.

**Returns:**
- `Dict[str, bool]`: Validation results for each component

## Error Handling

### Custom Exceptions

#### `AgentForgeError`
Base exception for all AgentForge errors.

#### `ValidationError`  
Raised when input validation fails.

#### `ProcessingError`
Raised when workflow processing encounters errors.

#### `APIError`
Raised when external API calls fail.

#### `ConfigurationError`
Raised when configuration is invalid.

### Error Response Format

All API errors return structured error information:

```python
{
    "error": "ProcessingError",
    "message": "Systems Analyst failed to analyze goal", 
    "details": {
        "step": "systems_analyst",
        "timestamp": "2024-01-15T10:30:00Z",
        "context": {"goal_id": "goal_123", "session_id": "sess_456"}
    },
    "recovery_suggestions": [
        "Try simplifying the goal description",
        "Reduce complexity level",
        "Check API key permissions"
    ]
}
```

## Rate Limits and Quotas

### API Usage Guidelines

- **Standard Goals**: ~10-20 API calls per workflow
- **Complex Goals**: ~30-50 API calls per workflow  
- **Enterprise Goals**: ~50-100 API calls per workflow

### Recommended Quotas

- **Development**: 100 requests/hour sufficient for testing
- **Production**: 1000+ requests/hour for regular usage
- **Enterprise**: Custom quotas based on usage patterns

## Authentication

### API Key Management

AgentForge supports multiple authentication methods:

```python
# Environment variables (recommended)
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# Direct configuration  
em = EngineeringManager(api_key="your_key")

# Configuration file
# config.json: {"openai_api_key": "your_key"}
```

### Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for production deployments
3. **Rotate keys regularly** for security
4. **Monitor usage** to detect unauthorized access
5. **Use minimal permissions** for API keys

---

This API reference provides complete documentation for integrating and extending AgentForge. For implementation examples, see the [User Guide](USER_GUIDE.md) and [Developer Guide](DEVELOPER_GUIDE.md).