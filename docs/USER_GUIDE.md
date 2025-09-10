# 📖 AgentForge User Guide

This comprehensive guide will walk you through using AgentForge to create custom agent teams for your specific goals.

## Table of Contents

- [Getting Started](#getting-started)
- [Understanding the Workflow](#understanding-the-workflow)
- [Using AgentForge](#using-agentforge)
- [Configuration Options](#configuration-options)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using AgentForge, ensure you have:

1. **Python 3.12+** installed
2. **API Keys** for your chosen LLM provider:
   - OpenAI API key (recommended)
   - Or OpenRouter API key for multiple model access
3. **Optional**: MCP API key for enhanced features

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agent-forge.git
cd agent-forge

# Install dependencies
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required - Choose one or both
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - Enhanced features
MCP_API_KEY=your_mcp_api_key_here
AGENT_LIBRARY_PATH=/path/to/your/existing/agents
TEAMS_LIBRARY_PATH=/path/to/your/existing/teams

# Database (optional - defaults to SQLite)
DATABASE_URL=sqlite:///agentforge.db
```

### Quick Test

Verify your installation:

```bash
python main.py --example
```

This runs AgentForge with a sample goal to ensure everything is working correctly.

## Understanding the Workflow

AgentForge follows a systematic 5-step workflow:

### 1. Goal Analysis (Systems Analyst)
- **Purpose**: Break down your high-level goal into specific requirements
- **Output**: Strategy Document with detailed role specifications
- **Duration**: ~15-30 seconds

### 2. Resource Scouting (Talent Scout) 
- **Purpose**: Find existing agents that match your requirements
- **Output**: Scouting Report with matches and gaps
- **Duration**: ~10-20 seconds

### 3. Agent Creation (Agent Developer)
- **Purpose**: Create new agents for any identified gaps
- **Output**: Custom agent specifications and code
- **Duration**: ~30-60 seconds (if gaps exist)

### 4. Team Integration (Integration Architect)
- **Purpose**: Assemble final team and create operational playbook
- **Output**: Complete team documentation and workflow
- **Duration**: ~15-30 seconds

### 5. Final Packaging (Engineering Manager)
- **Purpose**: Package everything for delivery
- **Output**: Ready-to-deploy agent team
- **Duration**: ~5 seconds

## Using AgentForge

### Interactive Mode

The easiest way to use AgentForge:

```bash
python main.py
```

You'll be guided through a series of questions:

```
🔥 Welcome to AgentForge - The Meta-Agent System
==================================================
📝 Describe your goal: Build a customer support chatbot system
🏷️  What domain (e.g., 'web development', 'data analysis'): customer service
📊 Complexity levels:
1. Low - Simple, single-purpose tasks
2. Medium - Multi-component projects  
3. High - Complex systems with integration
4. Enterprise - Large-scale, mission-critical systems
Select complexity (1-4) [default: 2]: 3
⏰ Timeline (optional): 6 weeks
🚫 Constraints (comma-separated, optional): Must integrate with Salesforce, 24/7 availability
✅ Success criteria (comma-separated, optional): Handle 80% of queries, Response time under 30 seconds
```

### Programmatic Usage

For integration into your own applications:

```python
import asyncio
from agents.engineering_manager import EngineeringManager, InputGoal

async def create_team():
    # Define your goal
    goal = InputGoal(
        goal_description="Build a comprehensive task management system",
        domain="productivity software",
        complexity_level="high",
        timeline="3 months",
        constraints=[
            "React frontend", 
            "Node.js backend", 
            "Real-time collaboration",
            "Mobile responsive"
        ],
        success_criteria=[
            "User authentication and authorization",
            "Real-time task updates across devices", 
            "Team collaboration features",
            "File attachments and comments",
            "Reporting and analytics dashboard"
        ],
        existing_resources=[
            "AWS cloud infrastructure",
            "Existing user database",
            "Design system components"
        ]
    )
    
    # Create the meta-team orchestrator
    em = EngineeringManager()
    
    # Process the goal
    result = await em.process(goal)
    
    return result

# Run the team creation
team = asyncio.run(create_team())
print(f"Created team with {len(team.team_members)} agents")
```

### Complexity Levels Guide

Choose the appropriate complexity level for your goal:

#### **Simple (Level 1)**
- Single-purpose, straightforward tasks
- 1-2 agent roles typically needed
- Minimal integration requirements
- **Examples**: Simple data analysis, basic content generation, single API integration

#### **Medium (Level 2)**
- Multi-component projects with some integration
- 3-5 agent roles typically needed  
- Moderate coordination requirements
- **Examples**: Web application with authentication, automated reporting system, customer service bot

#### **High (Level 3)**
- Complex systems with significant integration
- 5-8 agent roles typically needed
- Advanced coordination and communication protocols
- **Examples**: E-commerce platform, multi-service architecture, AI-powered analytics platform

#### **Enterprise (Level 4)**
- Mission-critical, large-scale systems
- 8+ agent roles typically needed
- Complex workflows and extensive documentation
- **Examples**: Banking system, healthcare platform, enterprise resource planning

## Configuration Options

### Model Selection

Choose different language models based on your needs:

```python
# Fast and cost-effective
em = EngineeringManager(model_id="anthropic/claude-3-haiku")

# Balanced performance (default)  
em = EngineeringManager(model_id="deepseek/deepseek-v3.1")

# High capability for complex tasks
em = EngineeringManager(model_id="openai/gpt-4o")

# Use OpenRouter for model variety
em = EngineeringManager(model_id="openrouter/auto")  # Auto-select best model
```

### Agent Library Configuration

Point AgentForge to your existing agent libraries:

```python
em = EngineeringManager(
    agent_library_path="/home/user/my-agents",
    knowledge_base_path="/home/user/my-knowledge"
)
```

Or set environment variables:
```bash
export AGENT_LIBRARY_PATH=/home/user/my-agents
export TEAMS_LIBRARY_PATH=/home/user/my-teams
```

### Database Configuration

Use different database backends:

```python
# SQLite (default)
em = EngineeringManager(db_file="my_agentforge.db")

# PostgreSQL
em = EngineeringManager(
    db_url="postgresql://user:password@localhost:5432/agentforge"
)
```

## Interpreting Results

AgentForge returns a comprehensive `TeamPackage` with all the information you need:

### Team Package Structure

```python
class TeamPackage:
    team_name: str                           # Name of your generated team
    goal_summary: str                        # Summary of original goal  
    team_members: List[Dict]                 # List of all agents with roles
    workflow_steps: List[str]                # How the team operates
    communication_protocols: Dict           # Inter-agent communication rules
    strategy_document: str                   # Full strategy analysis (JSON)
    scouting_report: str                     # Agent matching report (JSON)
    roster_documentation: str               # Complete operational playbook  
    new_agents_created: List[Dict]           # Details of new agents
    existing_agents_used: List[Dict]         # Reused agents with match scores
    deployment_instructions: str            # How to deploy this team
    success_metrics: List[str]              # How to measure success
```

### Understanding Team Members

Each team member includes:

```python
{
    "role": "Frontend Developer",           # The role this agent fills
    "type": "new",                         # "new" or "existing"  
    "capabilities": [                      # What this agent can do
        "React development",
        "State management", 
        "Component testing"
    ],
    "source": "agent_developer"           # Where it came from
}
```

### Workflow Steps

The generated workflow typically includes:

1. **Initialize team coordination** - Set up communication protocols
2. **Execute parallel workflows** - Agents work according to their roles
3. **Monitor progress** - Track deliverables and quality gates
4. **Handle coordination** - Manage inter-agent dependencies  
5. **Complete delivery** - Final validation and success metrics

### Communication Protocols

Generated teams include structured communication:

```python
{
    "coordination_hub": "Central message passing system",
    "workflow_handoffs": "Structured handoffs with validation",
    "quality_gates": "Automated quality checks at milestones",
    "error_handling": "Escalation procedures for failures"
}
```

## Best Practices

### Writing Effective Goals

#### ✅ Good Goal Examples

```python
# Specific and actionable
goal = "Build a customer support chatbot that integrates with our existing CRM system, handles common queries automatically, and escalates complex issues to human agents"

# Clear domain and context  
goal = "Create a data pipeline for processing financial transactions in real-time, with fraud detection capabilities and regulatory compliance reporting"

# Measurable success criteria
goal = "Develop a content management system that allows non-technical users to publish articles, manage media assets, and track engagement metrics"
```

#### ❌ Avoid Vague Goals

```python
# Too generic
goal = "Make a website"

# No clear requirements
goal = "Something with AI"  

# Unrealistic scope
goal = "Build the next Facebook but better"
```

### Providing Context

Always include relevant context:

```python
goal = InputGoal(
    goal_description="Build a customer support chatbot",
    domain="e-commerce",  # Specific domain
    constraints=[
        "Must integrate with Shopify",  # Technical constraints
        "Support for 5 languages",     # Functional constraints  
        "Budget: $50K",                 # Budget constraints
        "Go live in 3 months"          # Time constraints
    ],
    success_criteria=[
        "Handle 80% of common queries",     # Quantifiable metrics
        "Average response time < 5 seconds", # Performance targets
        "95% customer satisfaction",        # Quality metrics
        "Reduce support ticket volume by 50%" # Business impact
    ],
    existing_resources=[
        "Shopify store with 10K products",  # What you already have
        "Customer service team of 5",       # Current resources  
        "AWS cloud infrastructure"          # Technical assets
    ]
)
```

### Complexity Level Guidelines

- **Start with Medium** if unsure - you can always adjust
- **Use High for integration-heavy** projects requiring multiple systems
- **Choose Enterprise only for** mission-critical, large-scale systems
- **Simple is best for** proof-of-concepts and single-function tools

### Iterating on Results

If the generated team doesn't match your expectations:

1. **Refine your goal description** with more specific requirements
2. **Adjust complexity level** up or down based on initial results  
3. **Add more constraints** to guide the team composition
4. **Specify success criteria** more precisely to influence role creation
5. **Include existing resources** to encourage reuse over new creation

## Troubleshooting

### Common Issues

#### "No agents created for my goal"
**Cause**: Goal may be too simple or existing agents fulfill all requirements
**Solution**: 
- Increase complexity level
- Add more specific constraints  
- Check if existing agents in library already cover your needs

#### "Generated agents seem irrelevant"
**Cause**: Goal description may be unclear or domain mismatch
**Solution**:
- Provide more specific goal description
- Ensure domain accurately reflects your project
- Add detailed success criteria

#### "Process takes too long"  
**Cause**: Complex goals or slow model responses
**Solution**:
- Try a faster model like `anthropic/claude-3-haiku`
- Reduce complexity level for initial testing
- Check your internet connection and API limits

#### "API errors or timeouts"
**Cause**: API key issues or rate limiting
**Solution**:
- Verify API keys in `.env` file
- Check API key permissions and quotas
- Try switching between OpenAI and OpenRouter

### Error Messages

#### `ImportError: No module named 'agno'`
```bash
pip install agno>=2.0.2
```

#### `ValidationError: field required`  
Check that all required fields are provided in your `InputGoal`:
- `goal_description` (required)
- `domain` (required)

#### `FileNotFoundError: agent library path`
Set the correct path in your environment:
```bash
export AGENT_LIBRARY_PATH=/correct/path/to/agents
```

### Getting Help

1. **Check the logs**: AgentForge provides detailed logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

2. **Run with example**: Test with the built-in example
```bash
python main.py --example
```

3. **Review documentation**: Check other docs in this folder

4. **Open an issue**: [GitHub Issues](https://github.com/your-org/agent-forge/issues)

---

## Next Steps

Once you've mastered the basics:

- **[Developer Guide](DEVELOPER_GUIDE.md)** - Learn to extend and customize AgentForge
- **[API Reference](API_REFERENCE.md)** - Complete API documentation  
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment strategies
- **[Integration Examples](../examples/)** - Real-world usage examples

Ready to create your first agent team? Start with:

```bash
python main.py
```