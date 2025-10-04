# AgentForge MCP Server Guide 🚀

**Expose your AgentForge meta-team as MCP tools in Claude Code, Cline, and any MCP-compatible client!**

## What is This?

The AgentForge MCP Server transforms your meta-team of specialized agents into callable tools that can be used from any MCP-compatible environment. Instead of running agents manually, you can now:

```
User: "Create a team to build a real-time chat application"

Claude Code → agentforge_create_team tool
           → Engineering Manager orchestrates full workflow
           → Returns complete agent team with deployment docs
```

## Quick Start

### 1. Setup (One-time)

```bash
cd /path/to/agent-forge
./mcp_server/setup.sh
```

This will:
- ✅ Verify Python 3.12+
- ✅ Start QDrant vector database (Docker)
- ✅ Install dependencies
- ✅ Create log directories
- ✅ Show you the config to add to Claude Code

### 2. Configure Claude Code

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "agentforge-team": {
      "command": "python3",
      "args": ["/absolute/path/to/agent-forge/mcp_server/team_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/agent-forge"
      }
    }
  }
}
```

**Windows:** `%APPDATA%/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

### 3. Restart Claude Code

After adding the config, restart Claude Code completely.

### 4. Verify It's Working

In Claude Code, try:

```
"List my available MCP tools"
```

You should see:
- ✅ agentforge_create_team
- ✅ agentforge_analyze_strategy
- ✅ agentforge_scout_agents
- ✅ agentforge_develop_agents
- ✅ agentforge_quick_agent
- ✅ agentforge_get_workflow_status
- ✅ agentforge_reindex_libraries

## Available Tools

### 🚀 `agentforge_create_team`
**The full experience!** Orchestrates all meta-team agents to create a complete, ready-to-deploy agent team.

**Example:**
```
"Create a team to build an e-commerce platform with:
- React frontend
- Node.js backend
- PostgreSQL database
- Must support 10k concurrent users
- Timeline: 3 months"
```

**Returns:**
- Complete team roster (matched + new agents)
- Deployment instructions
- Operational playbook
- Quality metrics

---

### 📊 `agentforge_analyze_strategy`
**Strategic analysis phase.** Uses Systems Analyst to break down goals into roles and capabilities.

**Example:**
```
"Analyze the strategy for building a real-time analytics dashboard"
```

**Returns:**
- Recommended roles and responsibilities
- Required capabilities per role
- Interaction patterns
- Timeline estimates
- Risk analysis

---

### 🔍 `agentforge_scout_agents`
**Agent discovery.** Semantic search across your agent libraries using QDrant vector database.

**Example:**
```
"Scout for agents to fill these roles:
- Frontend Developer (React, TypeScript)
- Backend API Developer (Node.js, GraphQL)
- DevOps Engineer (Docker, Kubernetes)"
```

**Returns:**
- Matched agents with similarity scores
- Adaptation suggestions for close matches
- Vacant roles requiring new agents
- Reuse efficiency metrics

---

### 🛠️ `agentforge_develop_agents`
**Agent creation.** Master prompt engineering to create specialized Agno agents.

**Example:**
```
"Create agents for these vacant roles:
- ML Engineer (Python, TensorFlow, model deployment)
- Data Pipeline Engineer (Airflow, Spark, ETL)"
```

**Returns:**
- Complete agent specifications
- System prompts and instructions
- Python initialization code (runnable Agno agents!)
- Test suites
- Documentation

---

### ⚡ `agentforge_quick_agent`
**Fast single agent.** Quick creation without full workflow.

**Example:**
```
"Quick create an API Security Auditor agent with capabilities:
- API security scanning
- Vulnerability detection
- Compliance checking"
```

**Returns:**
- Complete agent spec ready to use

---

### 📈 `agentforge_get_workflow_status`
**Workflow monitoring.** Check Engineering Manager's orchestration progress.

**Example:**
```
"What's the status of my team creation workflow?"
```

**Returns:**
- Current step
- Completed steps history
- Latest action details

---

### 🔄 `agentforge_reindex_libraries`
**Library refresh.** Force re-indexing after adding new agents.

**Example:**
```
"Reindex my agent libraries - I just added 5 new agents"
```

**Returns:**
- Indexing statistics
- Number of agents indexed

## Usage Patterns

### Pattern 1: Full Team Creation
**Best for:** Complete projects requiring multiple specialized agents

```
User: "Create a team to build a SaaS analytics platform with real-time data processing"

→ agentforge_create_team
→ Engineering Manager coordinates:
  1. Systems Analyst analyzes requirements
  2. Talent Scout finds existing agents
  3. Agent Developer creates missing agents
  4. Integration Architect assembles team
→ Returns complete deployment package
```

### Pattern 2: Strategic Planning First
**Best for:** Understanding requirements before committing

```
User: "Analyze what kind of team I need for a mobile app with AI features"

→ agentforge_analyze_strategy
→ Returns strategic breakdown with roles

User: "Now scout for those agents"

→ agentforge_scout_agents
→ Returns matches and gaps

User: "Create the missing agents"

→ agentforge_develop_agents
→ Returns new agent specs
```

### Pattern 3: Quick Agent Creation
**Best for:** One-off agent needs

```
User: "I need a quick Code Reviewer agent with capabilities: Python, security, best practices"

→ agentforge_quick_agent
→ Returns complete agent spec immediately
```

### Pattern 4: Library Management
**Best for:** Maintaining your agent ecosystem

```
User: "I added 10 new agents to my library, reindex them"

→ agentforge_reindex_libraries
→ Updates vector database

User: "Now scout for agents matching: data engineering, ETL, cloud"

→ agentforge_scout_agents
→ Finds matches from newly indexed agents
```

## Real-World Examples

### Example 1: E-Commerce Platform

```
User: "Create a team to build an e-commerce platform with:
- Microservices architecture
- React + Next.js frontend
- Payment processing (Stripe)
- Inventory management
- Must handle 50k daily active users"

Claude Code: [Uses agentforge_create_team]

Result:
✅ Matched 3 existing agents:
   - React Frontend Developer (92% match)
   - DevOps Engineer (88% match)
   - Payment Integration Specialist (85% match)

✅ Created 4 new agents:
   - Microservices Architect
   - Inventory Management Specialist
   - Performance Optimization Engineer
   - E-Commerce Security Specialist

📦 Total Team: 7 agents
📊 Reuse Efficiency: 43%
📚 Documentation: Complete operational playbook included
```

### Example 2: Data Science Pipeline

```
User: "Build me a data science team for predictive analytics:
- Customer churn prediction
- Real-time feature engineering
- Model deployment on AWS
- A/B testing framework"

Claude Code: [Uses agentforge_create_team]

Result:
✅ Matched 2 existing agents:
   - Data Scientist (ML Engineer) (95% match)
   - AWS Cloud Architect (90% match)

✅ Created 3 new agents:
   - Feature Engineering Specialist
   - MLOps Engineer (deployment focus)
   - A/B Testing Framework Developer

📦 Total Team: 5 agents
📊 Reuse Efficiency: 40%
```

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Agent Libraries (where to find existing agents)
AGENT_LIBRARIES=/path/to/agents1,/path/to/agents2

# QDrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=touchmyflappyfoldyholds

# Model Configuration
DEFAULT_MODEL=deepseek/deepseek-v3.1
REASONING_MODEL=anthropic/claude-3-5-sonnet
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
```

### Agent Library Structure

Your agent libraries should contain agent definition files:

```
/home/user/agents/
├── frontend/
│   ├── react_developer.md
│   ├── vue_specialist.md
│   └── ui_designer.md
├── backend/
│   ├── node_api_developer.md
│   ├── python_fastapi_expert.md
│   └── go_microservices.md
└── data/
    ├── data_engineer.md
    ├── ml_engineer.md
    └── analytics_specialist.md
```

Each file should have structured metadata:

```markdown
# Agent Name

Role: Frontend Developer
Description: Expert React developer for building modern web UIs
Capabilities: React, TypeScript, Redux, Testing, Responsive Design
Tools: VSCode, Git, npm, Webpack
Domain: web development
Complexity: high
Tags: frontend, react, typescript, ui
```

## Troubleshooting

### QDrant Not Running

```bash
# Start QDrant with Docker
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Verify
curl http://localhost:6333/collections
```

### Tools Not Showing in Claude Code

1. Check config file location:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Verify absolute paths (no `~` or relative paths)

2. Check logs:
   ```bash
   tail -f mcp_server/agentforge_team.log
   ```

3. Test server manually:
   ```bash
   python3 mcp_server/team_server.py
   # Should show: "Starting AgentForge Team MCP Server..."
   ```

4. Restart Claude Code completely

### Import Errors

```bash
# Verify Python path
export PYTHONPATH=/absolute/path/to/agent-forge

# Reinstall dependencies
pip install -e .
```

### No Agents Found

```bash
# Force reindex libraries
# In Claude Code:
"Reindex my agent libraries from /path/to/agents"

# Or manually:
python3 -c "
from agents.talent_scout import TalentScout
import asyncio

async def reindex():
    scout = TalentScout()
    await scout.initialize()
    stats = await scout.index_agent_libraries(['/path/to/agents'], force_reindex=True)
    print(stats)

asyncio.run(reindex())
"
```

## Advanced Usage

### Custom Agent Templates

Create templates for common agent types:

```python
# custom_templates/api_developer_template.py
TEMPLATE = {
    "role": "API Developer",
    "capabilities": [
        "RESTful API design",
        "GraphQL",
        "Authentication/Authorization",
        "API documentation",
        "Performance optimization"
    ],
    "tools": ["Postman", "Swagger", "Git"],
    "complexity": "medium"
}
```

### Batch Agent Creation

```python
# Create multiple agents at once
roles = [
    {"role_name": "Frontend Dev", "capabilities": ["React", "TypeScript"]},
    {"role_name": "Backend Dev", "capabilities": ["Node.js", "PostgreSQL"]},
    {"role_name": "DevOps", "capabilities": ["Docker", "Kubernetes"]}
]

for role in roles:
    # Use agentforge_quick_agent for each
    ...
```

### Integration with CI/CD

```yaml
# .github/workflows/agent-team-creation.yml
name: Create Agent Team

on:
  workflow_dispatch:
    inputs:
      goal:
        description: 'Team goal'
        required: true

jobs:
  create-team:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -e .
      - name: Create team
        run: |
          python3 -c "
          from agents.engineering_manager import EngineeringManager, InputGoal
          import asyncio

          async def main():
              em = EngineeringManager()
              goal = InputGoal(
                  goal_description='${{ github.event.inputs.goal }}',
                  domain='software development'
              )
              result = await em.process(goal)
              print(result.roster_documentation)

          asyncio.run(main())
          "
```

## Performance Tips

1. **Library Organization**: Keep agent libraries organized by domain for faster indexing
2. **Reuse First**: Always use `agentforge_scout_agents` before creating new agents
3. **Batch Operations**: Create multiple similar agents in one go
4. **Cache Embeddings**: QDrant caches embeddings - reuse existing agents when possible
5. **Monitor Logs**: Check `mcp_server/agentforge_team.log` for performance insights

## API Reference

### Tool Schemas

#### agentforge_create_team
```json
{
  "goal_description": "string (required)",
  "domain": "string (required)",
  "complexity_level": "low|medium|high|enterprise",
  "timeline": "string (optional)",
  "constraints": ["string"],
  "success_criteria": ["string"]
}
```

#### agentforge_analyze_strategy
```json
{
  "description": "string (required)",
  "domain": "string (required)",
  "context": "string (optional)",
  "complexity": "low|medium|high|enterprise",
  "success_criteria": ["string"]
}
```

#### agentforge_scout_agents
```json
{
  "strategy_title": "string (required)",
  "goal_description": "string (required)",
  "domain": "string (required)",
  "complexity_level": "string (required)",
  "roles": [
    {
      "role_id": "string",
      "role_name": "string",
      "description": "string",
      "required_capabilities": ["string"],
      "preferred_capabilities": ["string"],
      "domain": "string",
      "complexity_level": "string",
      "priority": "low|medium|high|critical"
    }
  ],
  "agent_libraries": ["string"],
  "force_reindex": "boolean"
}
```

#### agentforge_develop_agents
```json
{
  "vacant_roles": [
    {
      "role_name": "string",
      "title": "string",
      "core_responsibilities": ["string"],
      "required_capabilities": ["string"],
      "interaction_patterns": {"role": "pattern"},
      "success_metrics": ["string"],
      "priority_level": "string",
      "domain_context": "string",
      "complexity_level": "string"
    }
  ],
  "capability_gaps": ["string"]
}
```

#### agentforge_quick_agent
```json
{
  "role_name": "string (required)",
  "capabilities": ["string (required)"],
  "priority": "low|medium|high|critical"
}
```

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/delorenj/agent-forge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/delorenj/agent-forge/discussions)
- **Documentation**: [Full Docs](https://github.com/delorenj/agent-forge/docs)

## What's Next?

Once your MCP server is running, try:

1. **Create Your First Team**
   ```
   "Create a team to build a task management app"
   ```

2. **Explore Strategic Analysis**
   ```
   "Analyze what I need for a microservices architecture"
   ```

3. **Discover Existing Agents**
   ```
   "Scout for agents with React and Node.js capabilities"
   ```

4. **Build Your Agent Library**
   - Add your own agents to `/path/to/agents`
   - Reindex with `agentforge_reindex_libraries`
   - Maximize reuse efficiency!

---

**Happy Agent Forging! 🔥⚒️**
