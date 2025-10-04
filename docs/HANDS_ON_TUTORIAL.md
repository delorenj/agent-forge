# 🚀 AgentForge Hands-On Tutorial

This tutorial walks you through two real-world scenarios for using AgentForge to build specialized agent teams for complex development projects.

## 📋 Prerequisites

Before starting, ensure you have:
- AgentForge installed and configured
- Environment variables set (OpenAI/OpenRouter API keys)
- QDrant vector database running (for agent matching)
- Sample project documentation ready

```bash
# Verify installation
python main.py --example

# Check environment
cat .env | grep -E "(OPENAI|OPENROUTER|MCP)_API_KEY"
```

## 🎯 Scenario 1: Fine-Grained Control - Custom Agent Team

In this scenario, you want precise control over your team composition. You'll specifically request an **Agno Agent Architect**, **Agentic Engineer**, **Data Scientist**, and **Infrastructure Expert** for a DevDash project.

### Step 1: Prepare Your Project Context

First, let's examine our sample tech stack document:

```bash
# View the tech stack requirements
cat docs/sample_tech_stack.md
```

This DevDash project requires:
- React/TypeScript frontend with real-time dashboards
- Node.js/GraphQL backend with microservices
- Kubernetes infrastructure with monitoring
- Data pipeline with ML/AI components

### Step 2: Launch AgentForge in Interactive Mode

```bash
python main.py
```

### Step 3: Provide Fine-Grained Requirements

When prompted, provide these specific inputs:

**Goal Description:**
```
Create a specialized agent team for DevDash - a developer dashboard with real-time insights. I need exactly 4 agents: an Agno Agent Architect for system design, an Agentic Engineer for implementation, a Data Scientist for analytics pipeline, and an Infrastructure Expert for Kubernetes deployment.
```

**Domain:**
```
full-stack development with data analytics
```

**Complexity Level:**
```
3 (High - Complex systems with integration)
```

**Timeline:**
```
4 months
```

**Constraints:**
```
Must use Agno framework, React/TypeScript frontend, Node.js backend, Kubernetes infrastructure, real-time data processing
```

**Success Criteria:**
```
Agno-based agent architecture, Real-time dashboard functionality, Scalable microservices, ML-powered insights, Production-ready Kubernetes deployment
```

### Step 4: Review the Generated Team

AgentForge will process your request and create:

1. **Agno Agent Architect**
   - Role: System architecture and Agno framework integration
   - Specialization: Multi-agent system design, workflow orchestration
   - Tools: Agno documentation access, architecture modeling

2. **Agentic Engineer** 
   - Role: Implementation of agent-based features
   - Specialization: React/TypeScript with agent integration
   - Tools: Code generation, testing frameworks, Agno SDK

3. **Data Scientist**
   - Role: Analytics pipeline and ML model development
   - Specialization: Real-time data processing, predictive analytics
   - Tools: Python ML stack, Kafka/Spark integration

4. **Infrastructure Expert**
   - Role: Kubernetes deployment and monitoring
   - Specialization: Container orchestration, observability
   - Tools: K8s manifests, Helm charts, monitoring setup

### Step 5: Examine Generated Artifacts

```bash
# View generated agent files
ls -la agents/*_agent.py

# Check documentation
ls -la docs/*_agent.md

# Review test files
ls -la tests/test_*_agent.py
```

### Step 6: Deploy and Test Your Team

```bash
# Test individual agents
python agents/agno_agent_architect_agent.py
python agents/agentic_engineer_agent.py
python agents/data_scientist_agent.py
python agents/infrastructure_expert_agent.py

# Run integration tests
python -m pytest tests/ -v
```

## 🤖 Scenario 2: Automated Optimization - Let AgentForge Decide

In this scenario, you provide your project requirements and let AgentForge automatically optimize the team composition based on your PRD and tech stack.

### Step 1: Prepare Comprehensive Documentation

Create a detailed PRD file:

```bash
# Create PRD for DevDash project
cat > docs/devdash_prd.md << 'EOF'
# DevDash PRD - Developer Dashboard Platform

## Vision
Create a comprehensive developer dashboard that provides real-time insights into development workflows, code quality, and team productivity.

## Core Features
- Real-time code metrics and quality indicators
- Team productivity analytics and reporting  
- Integration with popular dev tools (GitHub, Jira, Slack)
- Customizable dashboard widgets and layouts
- AI-powered insights and recommendations

## Technical Requirements
- Sub-200ms response times for all API calls
- Support for 10,000+ concurrent users
- 99.9% uptime SLA
- Real-time data updates with <1s latency
- Mobile-responsive design

## Success Metrics
- User engagement: 80%+ daily active users
- Performance: <200ms API response times
- Reliability: 99.9% uptime
- User satisfaction: 4.5+ star rating
EOF
```

### Step 2: Launch AgentForge with Document Context

```bash
python main.py
```

### Step 3: Provide High-Level Requirements

**Goal Description:**
```
Analyze the DevDash PRD and tech stack documents to create an optimally balanced agent team. Focus on delivering a production-ready developer dashboard with real-time analytics, high performance, and excellent user experience.
```

**Domain:**
```
enterprise web application development
```

**Complexity Level:**
```
4 (Enterprise - Large-scale, mission-critical systems)
```

**Timeline:**
```
6 months
```

**Constraints:**
```
Must follow tech stack in docs/sample_tech_stack.md, SOC 2 compliance required, budget-conscious approach preferred
```

**Success Criteria:**
```
Production-ready application, All PRD requirements met, Scalable architecture, Comprehensive testing, Security compliance
```

### Step 4: AgentForge Optimization Process

AgentForge will automatically:

1. **Analyze** your PRD and tech stack documents
2. **Identify** required capabilities and skill gaps
3. **Scout** existing agent libraries for reusable components
4. **Optimize** team composition for efficiency and coverage
5. **Create** new specialized agents only where needed

Expected optimized team might include:

- **Full-Stack Architect** (combines system design + implementation)
- **Frontend Performance Specialist** (React + real-time optimization)
- **Backend API Engineer** (GraphQL + microservices)
- **Data Pipeline Engineer** (Kafka + analytics)
- **DevOps Security Specialist** (K8s + compliance)
- **QA Automation Engineer** (testing + quality assurance)

### Step 5: Compare Optimization Results

```bash
# View the optimization report
cat output/scouting_report.md

# Check reused vs new agents
grep -E "(REUSED|NEW)" output/roster_documentation.md

# Review cost analysis
grep -A 10 "Resource Allocation" output/strategy_document.md
```

## 🔍 Key Differences Between Approaches

### Fine-Grained Control (Scenario 1)
- **Pros**: Exact team composition, specific expertise areas
- **Cons**: May miss optimization opportunities, potential skill gaps
- **Best for**: When you know exactly what specialists you need

### Automated Optimization (Scenario 2)  
- **Pros**: Optimal resource allocation, comprehensive coverage, cost-effective
- **Cons**: Less predictable team composition, may create unexpected roles
- **Best for**: Complex projects where you want maximum efficiency

## 🛠️ Advanced Usage Tips

### 1. Hybrid Approach
```bash
# Combine both approaches
python main.py
# Goal: "Create optimized team but ensure we have a dedicated Agno Architect and Data Scientist"
```

### 2. Iterative Refinement
```bash
# Run initial optimization
python main.py --example

# Review results and refine
python main.py
# Goal: "Refine the previous team by adding security specialist and removing redundant roles"
```

### 3. Context-Aware Generation
```bash
# Use existing agent libraries
export AGENT_LIBRARY_PATH="/path/to/existing/agents"
export TEAMS_LIBRARY_PATH="/path/to/existing/teams"
python main.py
```

## 🎯 Next Steps

After completing this tutorial:

1. **Experiment** with different complexity levels and constraints
2. **Customize** generated agents for your specific needs
3. **Integrate** with your existing development workflow
4. **Scale** by building reusable agent libraries
5. **Contribute** successful agent patterns back to the community

## 📚 Additional Resources

- [User Guide](USER_GUIDE.md) - Complete usage documentation
- [API Reference](API_REFERENCE.md) - Programmatic integration
- [Developer Guide](DEVELOPER_GUIDE.md) - Extending AgentForge
- [System Architecture](SYSTEM_ARCHITECTURE.md) - Technical deep dive

## 💡 Practical Examples & Code Snippets

### Example 1: Programmatic Team Creation

```python
# programmatic_team_creation.py
import asyncio
from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel

async def create_devdash_team():
    """Create a DevDash team programmatically."""

    goal = InputGoal(
        goal_description="Create specialized agents for DevDash developer dashboard with real-time analytics",
        domain="full-stack development with data analytics",
        complexity_level=ComplexityLevel.HIGH,
        timeline="4 months",
        constraints=[
            "Must use Agno framework",
            "React/TypeScript frontend",
            "Node.js/GraphQL backend",
            "Kubernetes infrastructure",
            "Real-time data processing"
        ],
        success_criteria=[
            "Agno-based agent architecture",
            "Real-time dashboard functionality",
            "Scalable microservices",
            "ML-powered insights",
            "Production-ready K8s deployment"
        ]
    )

    em = EngineeringManager()
    result = await em.process(goal)

    print(f"✅ Team Created: {len(result.new_agents)} new agents")
    print(f"📋 Strategy: {len(result.strategy_document)} chars")

    return result

# Run the example
if __name__ == "__main__":
    asyncio.run(create_devdash_team())
```

### Example 2: Custom Agent Specifications

```python
# custom_agent_specs.py
from agents.agent_developer import AgentDeveloper, VacantRole

async def create_custom_agno_architect():
    """Create a custom Agno Agent Architect."""

    role = VacantRole(
        role_name="Agno Agent Architect",
        title="Senior Agno Framework Architect",
        core_responsibilities=[
            "Design multi-agent system architectures using Agno",
            "Create agent workflow orchestration patterns",
            "Optimize agent communication and coordination",
            "Ensure scalable and maintainable agent systems"
        ],
        required_capabilities=[
            "Agno framework expertise",
            "System architecture design",
            "Agent workflow patterns",
            "Performance optimization",
            "Documentation and standards"
        ],
        interaction_patterns={
            "engineering_manager": "reports_to",
            "agentic_engineer": "collaborates_with",
            "infrastructure_expert": "coordinates_with"
        },
        success_metrics=[
            "System architecture clarity and completeness",
            "Agent workflow efficiency",
            "Team coordination effectiveness",
            "Documentation quality"
        ],
        priority_level="critical"
    )

    developer = AgentDeveloper()
    spec = await developer.create_agent_from_role(role)

    print(f"✅ Created: {spec.name}")
    print(f"🎯 Role: {spec.role}")
    print(f"🛠️ Tools: {', '.join(spec.tools_required)}")

    return spec
```

### Example 3: Team Integration Testing

```python
# test_team_integration.py
import pytest
import asyncio
from agents.integration_architect import IntegrationArchitect

@pytest.mark.asyncio
async def test_devdash_team_integration():
    """Test DevDash team integration."""

    # Mock team members
    team_members = [
        {"name": "AgnoArchitect", "role": "system_design"},
        {"name": "AgenticEngineer", "role": "implementation"},
        {"name": "DataScientist", "role": "analytics"},
        {"name": "InfraExpert", "role": "deployment"}
    ]

    architect = IntegrationArchitect()

    # Test team assembly
    roster = await architect.assemble_team(team_members)

    assert len(roster.team_members) == 4
    assert "AgnoArchitect" in [m.name for m in roster.team_members]
    assert roster.communication_patterns is not None

    print("✅ Team integration test passed")

# Run tests
if __name__ == "__main__":
    asyncio.run(test_devdash_team_integration())
```

## 🔧 Troubleshooting Common Issues

### Issue 1: Agent Creation Fails
```bash
# Check logs
tail -f logs/agentforge.log

# Verify API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"

# Test with simple goal
python main.py --example
```

### Issue 2: QDrant Connection Issues
```bash
# Check QDrant status
curl http://localhost:6333/health

# Restart QDrant
docker restart qdrant

# Use alternative vector store
export VECTOR_STORE=lancedb
```

### Issue 3: Generated Agents Don't Match Requirements
```bash
# Review strategy document
cat output/strategy_document.md

# Check scouting report
cat output/scouting_report.md

# Refine with more specific constraints
python main.py
# Add more detailed constraints and success criteria
```

Ready to build your first agent team? Start with:

```bash
python main.py
```
