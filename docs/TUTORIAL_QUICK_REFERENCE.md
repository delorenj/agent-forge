# 📚 AgentForge Tutorial Quick Reference

## 🚀 Quick Start Commands

```bash
# Basic interactive mode
python main.py

# Run example scenario
python main.py --example

# Run tutorial examples
python examples/tutorial_examples.py

# Test installation
python -c "from agents.engineering_manager import EngineeringManager; print('✅ AgentForge ready')"
```

## 🎯 Two Main Approaches

### 1. Fine-Grained Control
**When to use**: You know exactly what specialists you need
**Example goal**: "Create exactly 4 agents: Agno Architect, Agentic Engineer, Data Scientist, Infrastructure Expert"

```python
goal = InputGoal(
    goal_description="Create specific agents: [list exact roles]",
    constraints=["Must include X specialist", "Exactly N agents"],
    complexity_level=ComplexityLevel.HIGH
)
```

### 2. Automated Optimization  
**When to use**: You want maximum efficiency and comprehensive coverage
**Example goal**: "Create optimally balanced team for [project description]"

```python
goal = InputGoal(
    goal_description="Create optimized team for [project]",
    constraints=["Budget-conscious", "Follow tech stack"],
    complexity_level=ComplexityLevel.ENTERPRISE
)
```

## 📋 Input Templates

### DevDash Project (Sample)
```
Goal: "Developer dashboard with real-time analytics and team insights"
Domain: "full-stack development with data analytics"
Complexity: High (3) or Enterprise (4)
Timeline: "4-6 months"
Constraints: ["React/TypeScript", "Node.js/GraphQL", "Kubernetes", "Real-time processing"]
Success Criteria: ["Real-time dashboards", "Scalable architecture", "Production-ready"]
```

### E-commerce Platform
```
Goal: "E-commerce platform with payment processing and inventory management"
Domain: "e-commerce web development"
Complexity: High (3)
Timeline: "8 months"
Constraints: ["Microservices", "Payment compliance", "Mobile-first"]
Success Criteria: ["PCI compliance", "99.9% uptime", "Mobile responsive"]
```

### Data Pipeline Project
```
Goal: "Real-time data pipeline with ML model deployment"
Domain: "data engineering and machine learning"
Complexity: Enterprise (4)
Timeline: "5 months"
Constraints: ["Apache Kafka", "Python ML stack", "Cloud deployment"]
Success Criteria: ["Real-time processing", "Model accuracy >90%", "Auto-scaling"]
```

## 🛠️ Common Agent Types

| Agent Type | Best For | Key Capabilities |
|------------|----------|------------------|
| **Agno Agent Architect** | Multi-agent systems | Agno framework, workflow design |
| **Agentic Engineer** | Agent-integrated apps | React/TypeScript + agents |
| **Data Scientist** | Analytics & ML | Python ML, data pipelines |
| **Infrastructure Expert** | Deployment & ops | Kubernetes, monitoring |
| **Full-Stack Architect** | System design | End-to-end architecture |
| **Security Specialist** | Compliance & security | OWASP, SOC 2, penetration testing |
| **QA Engineer** | Testing & quality | Automated testing, CI/CD |

## 🔧 Complexity Levels Guide

| Level | Description | Team Size | Timeline | Example Projects |
|-------|-------------|-----------|----------|------------------|
| **Low (1)** | Simple, single-purpose | 1-2 agents | 1-2 months | Blog, landing page |
| **Medium (2)** | Multi-component | 2-4 agents | 2-4 months | CRM, dashboard |
| **High (3)** | Complex integration | 4-6 agents | 4-8 months | E-commerce, SaaS |
| **Enterprise (4)** | Mission-critical | 6+ agents | 6+ months | Banking, healthcare |

## 📊 Constraint Examples

### Technical Constraints
```
"Must use React/TypeScript frontend"
"Node.js backend required"
"Kubernetes deployment only"
"Real-time data processing <1s latency"
"Mobile-first responsive design"
```

### Business Constraints
```
"Budget-conscious approach"
"3-month timeline maximum"
"SOC 2 compliance required"
"GDPR compliance for EU users"
"99.9% uptime SLA"
```

### Team Constraints
```
"Maximum 4 agents"
"Must include security specialist"
"Reuse existing agents where possible"
"Focus on senior-level expertise"
```

## ✅ Success Criteria Examples

### Technical Success
```
"Sub-200ms API response times"
"Real-time dashboard updates"
"Automated CI/CD pipeline"
"Comprehensive test coverage >90%"
"Production-ready deployment"
```

### Business Success
```
"User engagement >80% daily active"
"Customer satisfaction >4.5 stars"
"Revenue impact measurable"
"Market launch within timeline"
"Scalable to 10,000+ users"
```

### Quality Success
```
"Code quality standards met"
"Security vulnerabilities <5"
"Documentation completeness 100%"
"Team collaboration effectiveness"
"Knowledge transfer completed"
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**Agent creation fails**
```bash
# Check API keys
echo $OPENAI_API_KEY | cut -c1-10
echo $OPENROUTER_API_KEY | cut -c1-10

# Test with simple example
python main.py --example
```

**QDrant connection issues**
```bash
# Check QDrant status
curl http://localhost:6333/health

# Start QDrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

**Generated agents don't match requirements**
```bash
# Review strategy document
cat output/strategy_document.md

# Use more specific constraints
# Add detailed success criteria
# Specify exact agent types needed
```

**Performance issues**
```bash
# Check system resources
htop

# Use lighter models
export MODEL_ID="deepseek/deepseek-v3.1"

# Reduce complexity level
# Use fewer agents
```

## 📁 Output Structure

```
output/
├── strategy_document.md      # Detailed analysis and plan
├── scouting_report.md       # Agent matching results
├── roster_documentation.md  # Final team composition
└── deployment_instructions.md # How to deploy the team

agents/
├── {agent_name}_agent.py    # Generated agent code
└── ...

tests/
├── test_{agent_name}_agent.py # Agent tests
└── ...

docs/
├── {agent_name}_agent.md    # Agent documentation
└── ...
```

## 🎯 Best Practices

1. **Start Simple**: Begin with lower complexity, iterate up
2. **Be Specific**: Detailed constraints = better results
3. **Review Outputs**: Always check strategy and scouting reports
4. **Test Early**: Run generated agents to verify functionality
5. **Iterate**: Refine based on results and feedback

## 📞 Getting Help

- **Documentation**: Check `docs/` folder for detailed guides
- **Examples**: Run `python examples/tutorial_examples.py`
- **Logs**: Check `logs/agentforge.log` for debugging
- **Issues**: Open GitHub issue with error details and context

Ready to start? Choose your approach:

```bash
# Fine-grained control
python main.py
# Goal: "Create exactly [N] agents: [specific roles]"

# Automated optimization  
python main.py
# Goal: "Create optimized team for [project description]"
```
