# =% AgentForge - The Meta-Agent System

**AgentForge** is a sophisticated meta-agent system built on the [Agno framework](https://docs.agno.com) that automatically analyzes high-level goals, devises optimal agent rosters, intelligently reuses existing agents, and creates specialized agents to fill gaps.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Agno](https://img.shields.io/badge/Agno-2.0.2+-green.svg)](https://docs.agno.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## <� What is AgentForge?

AgentForge is a **meta-agent system** - it creates other agent systems. Given a high-level goal, AgentForge:

1. **Analyzes** the goal and determines what kind of team is needed
2. **Scouts** existing agent libraries to find reusable agents
3. **Creates** new specialized agents to fill any gaps
4. **Assembles** a complete, ready-to-deploy agent team
5. **Documents** the team's operational playbook and deployment instructions

Think of it as having a senior engineering manager who can instantly assemble the perfect team for any project.

## <� The Meta-Team Architecture

AgentForge operates as a **5-agent meta-team** that collaborates to build other agent teams:

```mermaid
graph LR
    A[User Goal] --> B[Engineering Manager]
    B --> C[Systems Analyst]
    C --> D[Talent Scout]
    D --> E[Agent Developer]
    E --> F[Integration Architect]
    F --> G[Ready Team]

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

### The Meta-Team Roles

1. **<� Engineering Manager** (Orchestrator)
   - Central coordinator and workflow orchestrator
   - Manages the complete meta-team workflow
   - Packages final deliverables

2. **=� Systems Analyst** (Strategist)
   - Decomposes complex goals into discrete capabilities
   - Defines ideal team structures and roles
   - Creates comprehensive strategy documents

3. **=u Talent Scout** (Librarian)
   - Analyzes existing agent libraries for reusable agents
   - Performs semantic matching of capabilities
   - Identifies gaps that need new agents

4. **=� Agent Developer** (Creator)
   - Creates new specialized agents for identified gaps
   - Masters prompt engineering and agent design
   - Ensures agents follow best practices and standards

5. **<� Integration Architect** (Coordinator)
   - Assembles final teams from existing and new agents
   - Defines operational workflows and communication protocols
   - Creates comprehensive deployment documentation

## =� Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (or other LLM provider)
- QDrant vector database (for Talent Scout)
- Optional: MCP API key for advanced features

### Installation

```bash
# Clone the repository
git clone https://github.com/delorenj/agent-forge.git
cd agent-forge

# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Usage Options

#### 🚀 Option 1: MCP Server (Recommended for Claude Code/Cline)

**Use AgentForge as MCP tools in Claude Code, Cline, or any MCP-compatible client!**

```bash
# One-time setup
./mcp_server/setup.sh

# Add to Claude Code config:
# ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
{
  "mcpServers": {
    "agentforge-team": {
      "command": "python3",
      "args": ["/path/to/agent-forge/mcp_server/team_server.py"],
      "env": {"PYTHONPATH": "/path/to/agent-forge"}
    }
  }
}

# Then in Claude Code:
"Create a team to build a real-time chat application"
```

**Available MCP Tools:**
- `agentforge_create_team` - Full team creation workflow
- `agentforge_analyze_strategy` - Strategic analysis (Systems Analyst)
- `agentforge_scout_agents` - Agent discovery (Talent Scout)
- `agentforge_develop_agents` - Create new agents (Agent Developer)
- `agentforge_quick_agent` - Quick single agent creation
- `agentforge_get_workflow_status` - Workflow monitoring
- `agentforge_reindex_libraries` - Refresh agent index

📚 **[Full MCP Server Guide](docs/MCP_SERVER_GUIDE.md)** | **[Usage Examples](examples/mcp_usage_examples.md)**

#### 📝 Option 2: Interactive Mode

```bash
python main.py
```

#### 💻 Option 3: Programmatic Usage

```python
from agents.engineering_manager import EngineeringManager, InputGoal

# Create a goal
goal = InputGoal(
    goal_description="Build a customer support chatbot with AI capabilities",
    domain="customer service",
    complexity_level="medium",
    constraints=["Must integrate with existing CRM", "24/7 availability"],
    success_criteria=["Handle 80% of queries automatically", "Response time < 30 seconds"]
)

# Create and run the meta-team
em = EngineeringManager()
result = await em.process(goal)

print(f"Created team with {len(result.team_members)} agents")
print(f"Documentation: {result.roster_documentation}")
```

#### Example Goal

```bash
# Run with example goal
python main.py --example
```

## =� Key Features

### ( Intelligent Agent Reuse

- **Semantic Analysis**: Matches existing agents to required capabilities
- **Pattern Recognition**: Identifies reusable patterns across domains
- **Library Integration**: Scans configurable agent libraries
- **Adaptation Recommendations**: Suggests modifications for better fit

### >� Dynamic Agent Creation

- **Gap Analysis**: Identifies missing capabilities in existing roster
- **Specialized Generation**: Creates agents optimized for specific roles
- **Best Practice Integration**: Follows established agent design patterns
- **Quality Validation**: Ensures generated agents meet quality standards

### = Structured Workflows

- **Deterministic Process**: Consistent, reliable team generation
- **Quality Gates**: Validation at each step of the process
- **Session Management**: Tracks progress and enables recovery
- **Type Safety**: Full Pydantic validation for all data structures

### =� Comprehensive Documentation

- **Strategy Documents**: Detailed analysis and team composition
- **Scouting Reports**: Agent matching analysis and gap identification
- **Roster Documentation**: Complete operational playbooks
- **Deployment Guides**: Step-by-step deployment instructions

## =� System Requirements

### Core Dependencies

```toml
agno = ">=2.0.2"           # Core agent framework
fastapi = ">=0.116.1"       # API server (with standard extras)
mcp = ">=1.13.1"           # Model Context Protocol
openai = ">=1.107.0"       # OpenAI models and embeddings
openrouter = ">=1.0"       # Alternative model access
sqlalchemy = ">=2.0.43"    # Database ORM
lancedb = ">=0.16.0"       # Vector database
pydantic = ">=2.0.0"       # Type validation
```

### Environment Configuration

```bash
# Required
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

# Optional - Enhanced Features
MCP_API_KEY=your_mcp_key
AGENT_LIBRARY_PATH=/path/to/your/agents
TEAMS_LIBRARY_PATH=/path/to/your/teams

# Database
DATABASE_URL=sqlite:///agentforge.db
```

## =� Documentation

### Core Documentation

- [**System Architecture**](SYSTEM_ARCHITECTURE.md) - Detailed technical architecture
- [**User Guide**](docs/USER_GUIDE.md) - Step-by-step usage instructions
- [**Developer Guide**](docs/DEVELOPER_GUIDE.md) - Extending and customizing AgentForge
- [**API Reference**](docs/API_REFERENCE.md) - Complete API documentation
- [**Deployment Guide**](docs/DEPLOYMENT_GUIDE.md) - Production deployment instructions

### Agent Documentation

- [**Engineering Manager**](agents/README.md#engineering-manager) - Central orchestrator
- [**Systems Analyst**](agents/README.md#systems-analyst) - Goal analysis and strategy
- [**Talent Scout**](agents/README.md#talent-scout) - Agent library management
- [**Agent Developer**](agents/README.md#agent-developer) - New agent creation
- [**Integration Architect**](agents/README.md#integration-architect) - Team assembly

### Framework Integration

- [**Agno Integration**](docs/agno/) - Using Agno framework features
- [**MCP Integration**](docs/MCP_INTEGRATION.md) - Model Context Protocol features
- [**Knowledge Systems**](docs/KNOWLEDGE_SYSTEMS.md) - Vector databases and search

## <� Usage Examples

### Example 1: Web Development Team

```python
goal = InputGoal(
    goal_description="Create a full-stack e-commerce platform",
    domain="web development",
    complexity_level="high",
    constraints=["React frontend", "Node.js backend", "PostgreSQL"],
    success_criteria=["User authentication", "Payment processing", "Admin dashboard"]
)

team = await em.process(goal)
# Result: Frontend Developer, Backend Developer, Database Architect,
#         Security Specialist, DevOps Engineer, QA Engineer
```

### Example 2: Data Science Pipeline

```python
goal = InputGoal(
    goal_description="Build a customer churn prediction system",
    domain="data science",
    complexity_level="medium",
    constraints=["Python/scikit-learn", "Real-time scoring"],
    success_criteria=["85%+ accuracy", "Sub-100ms predictions", "Automated retraining"]
)

team = await em.process(goal)
# Result: Data Engineer, Feature Engineer, ML Engineer,
#         Model Validator, Deployment Specialist
```

### Example 3: Content Creation System

```python
goal = InputGoal(
    goal_description="Automated content generation and social media management",
    domain="content marketing",
    complexity_level="medium",
    constraints=["Multi-platform", "Brand voice consistency"],
    success_criteria=["Daily content creation", "Engagement tracking", "A/B testing"]
)

team = await em.process(goal)
# Result: Content Strategist, Copy Writer, Social Media Manager,
#         Analytics Specialist, Brand Voice Guardian
```

## =' Advanced Configuration

### Custom Agent Libraries

```python
# Configure custom agent library paths
em = EngineeringManager(
    agent_library_path="/custom/path/to/agents",
    knowledge_base_path="/custom/path/to/knowledge"
)
```

### Model Selection

```python
# Use different language models
em = EngineeringManager(
    model_id="anthropic/claude-3-haiku",  # Faster, cheaper
    # model_id="openai/gpt-4o",          # More capable
    # model_id="deepseek/deepseek-v3.1", # Default balance
)
```

### Database Configuration

```python
# Custom database configuration
em = EngineeringManager(
    db_file="custom_agentforge.db",
    # Or use PostgreSQL
    # db_url="postgresql://user:pass@localhost/agentforge"
)
```

## >� Testing

### Run Tests

```bash
# Run all tests
python -m pytest

# Test individual agents
python test_systems_analyst.py
python test_agent_developer.py

# Integration tests
python test_integration_architect.py
```

### Run Demos

```bash
# Individual agent demos
python demo_systems_analyst.py
python demo_agent_developer.py
python demo_integration_architect.py

# Full workflow demo
python main.py --example
```

## > Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/delorenj/agent-forge.git
cd agent-forge
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 .
black .
```

### Adding New Agents

See [Developer Guide](docs/DEVELOPER_GUIDE.md) for details on:

- Agent design patterns
- Integration with the meta-team workflow
- Testing and validation
- Documentation standards

## =� Performance & Scalability

- **Processing Time**: Typically 30-120 seconds for complete team generation
- **Agent Library**: Scales to thousands of existing agents
- **Concurrent Goals**: Multiple goals can be processed simultaneously
- **Memory Usage**: Efficient vector database indexing and caching

## = Security

- **API Key Management**: Secure credential handling
- **Input Validation**: Comprehensive input sanitization
- **Generated Code Safety**: Validation of all generated agent code
- **Access Control**: Role-based access to agent libraries

## =� Roadmap

### Version 1.1 (Q2 2024)

- [ ] Multi-modal agent support
- [ ] Real-time collaboration features
- [ ] Enhanced agent adaptation algorithms
- [ ] Performance optimization and caching

### Version 1.2 (Q3 2024)

- [ ] Agent marketplace integration
- [ ] Version control for agent libraries
- [ ] A/B testing for generated agents
- [ ] Advanced monitoring and analytics

### Version 2.0 (Q4 2024)

- [ ] Distributed agent execution
- [ ] Advanced learning and adaptation
- [ ] Plugin ecosystem
- [ ] Enterprise features and deployment

## =� License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## <� Support

- **Documentation**: [docs.agentforge.com](https://docs.agentforge.com)
- **Issues**: [GitHub Issues](https://github.com/delorenj/agent-forge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/delorenj/agent-forge/discussions)
- **Discord**: [AgentForge Community](https://discord.gg/agentforge)

## =O Acknowledgments

- Built on the excellent [Agno framework](https://docs.agno.com)
- Inspired by multi-agent systems research
- Community contributions and feedback
- Open source tools and libraries that make this possible

---

**Ready to forge your perfect agent team?** Start with `python main.py` and let AgentForge analyze your goal!
