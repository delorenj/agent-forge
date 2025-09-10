# AgentForge Agents

This directory contains the implementation of the AgentForge meta-team agents.

## Systems Analyst - The Strategist

**Location:** `systems_analyst.py`

### Purpose
Expert in decomposing complex goals into discrete, manageable roles and capabilities. Defines the IDEAL team structure required to solve problems without regard for existing resources.

### Key Features
- **Goal Decomposition**: Breaks down complex goals into component parts and dependencies
- **Role Definition**: Creates detailed agent role specifications with responsibilities and capabilities
- **Team Structure**: Defines optimal coordination patterns and communication protocols
- **Reasoning Integration**: Uses Agno ReasoningTools for systematic analysis
- **Knowledge Integration**: Leverages Agno KnowledgeTools for pattern matching
- **Strategy Documents**: Produces comprehensive strategy documents in markdown format

### Input/Output

**Input:** `InputGoal` (Pydantic model)
- `description`: High-level goal description
- `context`: Additional context or constraints
- `success_criteria`: How success will be measured
- `domain`: Domain/industry context
- `complexity`: Estimated complexity level

**Output:** `StrategyDocument` (Pydantic model)
- `goal_analysis`: Analysis of the input goal
- `team_composition`: Required agent roles with detailed specs
- `team_structure`: Team organization and coordination patterns
- `risk_assessment`: Potential risks and mitigation strategies
- `resource_requirements`: Resource needs assessment
- `timeline_estimate`: Estimated time requirements

### Usage Examples

#### Basic Usage
```python
from agents.systems_analyst import SystemsAnalyst, InputGoal

# Initialize the analyst
analyst = SystemsAnalyst()

# Define a goal
goal = InputGoal(
    description="Build a customer support system with AI chatbots",
    context="Mid-size SaaS company with 10,000+ customers",
    success_criteria=["Reduce response time", "Handle 80% queries automatically"],
    domain="Customer Support",
    complexity="High"
)

# Analyze the goal
strategy = await analyst.analyze_goal(goal)

# Create strategy document
doc_path = analyst.create_strategy_document(strategy, "strategy.md")
```

#### Quick Analysis
```python
# For simple goals
result = await analyst.quick_analysis("Create a blog website with user authentication")
```

### Integration with AgentForge Workflow

1. **Engineering Manager** receives input goal
2. **Systems Analyst** analyzes goal and creates Strategy Document
3. **Talent Scout** uses Strategy Document to match existing agents
4. **Agent Developer** creates new agents for gaps
5. **Integration Architect** assembles final team

### Agno Framework Integration

The Systems Analyst leverages several Agno components:

- **ReasoningTools**: For systematic problem decomposition
  - `think`: Scratchpad for reasoning through problems
  - `analyze`: Evaluating results and determining next steps

- **KnowledgeTools**: For pattern matching and best practices
  - `search`: Finding relevant documentation and patterns
  - `analyze`: Evaluating knowledge relevance

- **OpenRouter Model**: For intelligent analysis and generation
- **Structured Output**: Using Pydantic models for type safety

### Configuration

The agent can be configured with:
- **Knowledge Base Path**: Local documentation to include
- **Model Selection**: Different language models via OpenRouter
- **Reasoning Depth**: How thorough the analysis should be

### Testing

Run the test suite:
```bash
python test_systems_analyst.py
```

Run the demo:
```bash
python demo_systems_analyst.py
```

### Dependencies

- `agno>=2.0.2`: Core framework
- `openrouter>=1.0`: Model access
- `lancedb>=0.16.0`: Vector database for knowledge
- `pydantic>=2.0.0`: Type validation
- `openai`: Embeddings for knowledge search

### Next Steps

The Systems Analyst is designed to work with the other AgentForge agents:

- **Talent Scout**: Will use the Strategy Documents to match existing agents
- **Agent Developer**: Will use role specifications to create new agents
- **Integration Architect**: Will use team structure for final assembly
- **Engineering Manager**: Orchestrates the entire workflow

Each agent follows the same pattern of structured input/output and Agno framework integration.