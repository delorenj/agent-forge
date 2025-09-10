# ❓ AgentForge FAQ & Troubleshooting

Frequently asked questions and troubleshooting guide for AgentForge users and developers.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage & Configuration](#usage--configuration)
- [Performance & Scaling](#performance--scaling)
- [Development & Customization](#development--customization)
- [Deployment & Production](#deployment--production)
- [Troubleshooting](#troubleshooting)
- [Error Messages](#error-messages)

## General Questions

### What is AgentForge?

**Q: What exactly does AgentForge do?**

A: AgentForge is a meta-agent system that automatically creates custom agent teams for specific goals. Given a high-level objective, it:

1. **Analyzes** your goal to understand what's needed
2. **Scouts** existing agent libraries to find reusable agents
3. **Creates** new specialized agents to fill any gaps
4. **Assembles** a complete, ready-to-deploy agent team
5. **Documents** the team's operational playbook

Think of it as having an expert engineering manager who can instantly assemble the perfect team for any project.

**Q: How is this different from using individual AI agents?**

A: Instead of manually figuring out what agents you need, creating them, and coordinating their work, AgentForge does all of this automatically. It's the difference between hiring individual contractors yourself vs. using a staffing agency that understands your needs and assembles the perfect team.

**Q: What kinds of projects can AgentForge help with?**

A: AgentForge can create teams for virtually any domain:
- **Software Development**: Full-stack applications, APIs, mobile apps
- **Data Science**: Analytics pipelines, ML models, data processing
- **Content Creation**: Writing, marketing, social media management
- **Business Operations**: Workflow automation, customer support
- **Research**: Literature reviews, analysis, report generation

### How does the 5-agent meta-team work?

**Q: What do each of the 5 meta-agents do?**

A: The meta-team consists of:

1. **Engineering Manager** - Orchestrates the entire process
2. **Systems Analyst** - Breaks down your goal into specific requirements  
3. **Talent Scout** - Finds existing agents that match your needs
4. **Agent Developer** - Creates new agents for any missing capabilities
5. **Integration Architect** - Assembles everything into a working team

**Q: Do I interact with all 5 agents?**

A: No, you only interact with the Engineering Manager. It coordinates with the other agents automatically and delivers the final team to you.

**Q: How long does the whole process take?**

A: Typically 1-3 minutes for simple goals, 3-10 minutes for complex goals. The time depends on:
- Goal complexity
- Number of new agents needed
- API response times
- Your complexity level setting

## Installation & Setup

### Requirements & Dependencies

**Q: What are the minimum system requirements?**

A: 
- **Python**: 3.12 or higher
- **Memory**: 2GB RAM minimum, 4GB+ recommended
- **Storage**: 1GB free space for installation
- **Network**: Internet connection for API calls
- **OS**: Windows, macOS, or Linux

**Q: Do I need special hardware like GPUs?**

A: No, AgentForge runs the orchestration logic locally but uses cloud-based LLM APIs for the actual AI processing. No special hardware required.

**Q: What API keys do I need?**

A: At minimum, you need ONE of these:
- **OpenAI API key** (recommended for best results)
- **OpenRouter API key** (gives access to multiple models)

Optional but recommended:
- **MCP API key** (for enhanced features)

### Installation Issues

**Q: Installation fails with "No module named 'agno'"**

A: This usually means the Agno framework didn't install correctly. Try:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with specific version
pip install "agno>=2.0.2"

# Or install from scratch
pip uninstall agno
pip install agno
```

**Q: I get permission errors during installation**

A: Use a virtual environment to avoid permission issues:
```bash
python -m venv agentforge-env
source agentforge-env/bin/activate  # Linux/Mac
# agentforge-env\Scripts\activate  # Windows
pip install -e .
```

**Q: Dependencies conflict with my existing packages**

A: AgentForge should be installed in its own virtual environment:
```bash
# Create isolated environment
python -m venv agentforge
cd agentforge
source bin/activate  # Linux/Mac
pip install agent-forge
```

## Usage & Configuration

### Basic Usage

**Q: How do I write a good goal description?**

A: Follow these guidelines:

✅ **Good Goals:**
- Specific and actionable: "Build a customer support chatbot that integrates with Salesforce"
- Include context: "For a mid-size e-commerce company with 10,000+ customers"
- Mention constraints: "Must use React frontend and Node.js backend"
- Define success: "Handle 80% of queries automatically, <5 second response time"

❌ **Avoid:**
- Too vague: "Make a website"
- Too generic: "Something with AI"
- Unrealistic: "Build the next Facebook but better"

**Q: What complexity level should I choose?**

A: 
- **Simple**: Single-purpose tools, 1-2 agents (blog website, data analysis script)
- **Medium**: Multi-component projects, 3-5 agents (web app with auth, reporting system)  
- **High**: Complex systems, 5-8 agents (e-commerce platform, real-time collaboration)
- **Enterprise**: Mission-critical systems, 8+ agents (banking system, healthcare platform)

**Q: The generated team doesn't match my expectations. What should I do?**

A: Try these approaches:
1. **Refine your goal** with more specific requirements
2. **Adjust complexity level** up or down
3. **Add more constraints** to guide team composition
4. **Include existing resources** to influence decisions
5. **Specify success criteria** more precisely

### Configuration

**Q: Which language model should I use?**

A: Recommendations by use case:

- **Development/Testing**: `anthropic/claude-3-haiku` (fast, cost-effective)
- **General Use**: `deepseek/deepseek-v3.1` (good balance, default)
- **Complex Goals**: `openai/gpt-4o` (highest capability)
- **Variety**: `openrouter/auto` (auto-selects best model)

**Q: How do I set up my own agent library?**

A: 
1. Create a directory structure:
```
my-agents/
├── web-development/
│   ├── frontend-developer.md
│   └── backend-developer.md
├── data-science/
│   ├── data-engineer.md
│   └── ml-engineer.md
└── content/
    └── copywriter.md
```

2. Set the environment variable:
```bash
export AGENT_LIBRARY_PATH=/path/to/my-agents
```

3. AgentForge will automatically scan and use these agents

**Q: Can I use different databases besides SQLite?**

A: Yes, AgentForge supports:
- **SQLite** (default, good for development)
- **PostgreSQL** (recommended for production)
- **MySQL** (supported)

Configure with the DATABASE_URL:
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/agentforge

# MySQL  
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/agentforge
```

## Performance & Scaling

### Speed & Efficiency

**Q: Why is AgentForge slow for my goals?**

A: Common causes and solutions:

1. **Complex goals take longer**
   - Solution: Start with Medium complexity, increase if needed

2. **API rate limiting** 
   - Solution: Check your API key limits and upgrade if needed

3. **Large agent library scanning**
   - Solution: Organize library in subdirectories by domain

4. **Slow model selection**
   - Solution: Use faster models like `claude-3-haiku` for testing

**Q: Can I run multiple goals in parallel?**

A: Yes, but with considerations:
- **Development**: 1-3 parallel workflows
- **Production**: Configure `MAX_CONCURRENT_WORKFLOWS` based on API limits
- **Rate limits**: Monitor API usage to avoid hitting limits

**Q: How can I reduce API costs?**

A: Cost optimization strategies:
1. **Use faster/cheaper models** for testing: `claude-3-haiku`, `deepseek-v3.1`
2. **Cache results** - AgentForge has built-in caching
3. **Batch similar goals** - Run related projects together
4. **Set complexity appropriately** - Don't use Enterprise for simple tasks

### Memory & Resources

**Q: AgentForge uses too much memory**

A: Memory optimization:
1. **Reduce concurrent workflows**: Set lower `MAX_CONCURRENT_WORKFLOWS`
2. **Use connection pooling**: Enable database connection pooling
3. **Clear caches**: Regularly clear Redis cache if using
4. **Monitor usage**: Use `docker stats` to track memory usage

**Q: Can AgentForge run on small servers?**

A: Yes, minimum requirements:
- **1 CPU core** (2+ recommended)
- **2GB RAM** (4GB+ for production)
- **1GB disk space**
- Most of the AI processing happens via APIs, so local resource needs are modest

## Development & Customization

### Creating Custom Agents

**Q: How do I create my own custom agent?**

A: Follow this pattern:

1. **Define the agent class:**
```python
from agents.base import BaseAgent
from pydantic import BaseModel, Field

class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="My Custom Agent", **kwargs)
    
    async def process(self, input_data: MyInputSchema) -> MyOutputSchema:
        # Your custom logic here
        pass
```

2. **Register the agent:**
```python
from agents import AgentFactory
AgentFactory.register_agent("my_custom", MyCustomAgent)
```

3. **Add tests and documentation**

See the [Developer Guide](DEVELOPER_GUIDE.md#creating-custom-agents) for complete details.

**Q: Can I modify the existing agents?**

A: Yes, several approaches:
1. **Extend existing agents** - Inherit and override methods
2. **Configure differently** - Pass different parameters
3. **Replace entirely** - Create custom agent with same interface
4. **Add tools** - Extend agents with additional tools

**Q: How do I add new tools to agents?**

A: Create tools following the Agno framework patterns:
```python
class MyCustomTool:
    name = "my_tool"
    description = "What my tool does"
    
    async def execute(self, input_data):
        # Tool logic here
        return result

# Add to agent
agent.tools.append(MyCustomTool())
```

### Integration & Extensions

**Q: Can I integrate AgentForge with my existing system?**

A: Yes, multiple integration approaches:
1. **Python API** - Import and use directly
2. **REST API** - Deploy as microservice
3. **CLI integration** - Call from command line
4. **Webhook integration** - Trigger via webhooks

**Q: Does AgentForge support custom knowledge bases?**

A: Yes, through the Agno framework:
- **Vector databases**: LanceDB, Chroma, Pinecone
- **Document stores**: Upload your own documentation
- **Custom sources**: Integrate with your knowledge systems

**Q: Can I customize the workflow?**

A: Yes, you can:
1. **Add new steps** to the workflow
2. **Skip steps** for specific use cases
3. **Create parallel workflows** for different domains
4. **Add validation gates** for quality control

## Deployment & Production

### Production Deployment

**Q: Is AgentForge production-ready?**

A: Yes, with proper configuration:
- **Stateless design** - Scales horizontally
- **Database persistence** - Session and result storage
- **Health checks** - Built-in monitoring endpoints
- **Security features** - API key management, input validation
- **Docker support** - Container-ready deployment

**Q: What's the recommended production architecture?**

A: For production:
- **Load balancer** (nginx, AWS ALB)
- **Multiple app instances** (3+ replicas)
- **PostgreSQL database** (managed service recommended)
- **Redis cache** (for session storage)
- **Container orchestration** (Kubernetes, ECS)
- **Monitoring** (Prometheus, CloudWatch)

**Q: How do I handle secrets in production?**

A: Use proper secret management:
- **AWS**: Secrets Manager or Parameter Store
- **Kubernetes**: Secret resources
- **Azure**: Key Vault
- **GCP**: Secret Manager
- **Docker**: Docker secrets

Never put API keys in environment files or code.

### Scaling & Performance

**Q: How many requests can AgentForge handle?**

A: Depends on configuration:
- **Single instance**: 10-50 concurrent workflows
- **Horizontal scaling**: 100s-1000s of workflows
- **Bottlenecks**: Usually API rate limits, not AgentForge itself

**Q: How do I scale AgentForge horizontally?**

A: AgentForge is designed for horizontal scaling:
1. **Stateless design** - No shared state between instances
2. **External database** - PostgreSQL for shared data
3. **Session storage** - Redis for session management
4. **Load balancing** - Distribute requests across instances

Example Kubernetes scaling:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentforge-hpa
spec:
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

## Troubleshooting

### Common Issues

**Q: "No agents were created for my goal"**

A: This usually means:
1. **Goal too simple** - Existing agents already cover everything
2. **Complexity level too low** - Try increasing complexity
3. **Very specific constraints** - May have filtered out all possibilities

**Solution**: Try a slightly more complex goal or reduce constraints.

**Q: "Generated agents seem irrelevant"**

A: Common causes:
1. **Unclear goal description** - Be more specific
2. **Wrong domain** - Make sure domain matches your goal
3. **Missing context** - Add more background information

**Solution**: Refine your goal description with more specific requirements.

**Q: "API errors or timeouts"**

A: Troubleshooting steps:
1. **Check API keys** - Verify they're correct and have permissions
2. **Check rate limits** - You may have exceeded API quotas
3. **Network issues** - Test basic internet connectivity
4. **Service outages** - Check provider status pages

**Q: "Database connection errors"**

A: Common fixes:
1. **Check DATABASE_URL** - Ensure it's correct
2. **Database running** - Verify database server is running
3. **Permissions** - Check user has necessary permissions
4. **Network access** - Ensure AgentForge can reach database

**Q: "Memory usage keeps growing"**

A: Likely causes:
1. **Memory leaks** - Check for unclosed connections
2. **Large datasets** - Process data in chunks
3. **Cache growth** - Implement cache eviction policies
4. **Connection pools** - Configure connection pool limits

### Debugging Tips

**Q: How do I debug AgentForge issues?**

A: Debugging strategies:
1. **Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Test individual agents**:
```python
from agents.systems_analyst import SystemsAnalyst
analyst = SystemsAnalyst()
# Test with simple input
```

3. **Check component health**:
```bash
curl http://localhost:8000/health
```

4. **Monitor metrics**:
```bash
curl http://localhost:8000/metrics
```

**Q: How do I get more detailed error information?**

A: Enable verbose error reporting:
1. Set `LOG_LEVEL=DEBUG` in environment
2. Check application logs: `docker-compose logs -f agentforge`
3. Look at specific agent logs
4. Enable database query logging if needed

**Q: The workflow gets stuck at a specific step**

A: Debugging workflow issues:
1. **Check agent-specific logs** for the stuck step
2. **Verify input data** going into that step
3. **Test the specific agent** in isolation
4. **Check for API timeouts** or rate limiting
5. **Verify all required environment variables** are set

## Error Messages

### Installation Errors

**Error**: `ModuleNotFoundError: No module named 'agno'`
**Solution**: 
```bash
pip install --upgrade pip
pip install "agno>=2.0.2"
```

**Error**: `Permission denied` during installation
**Solution**: Use virtual environment or add `--user` flag
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Runtime Errors

**Error**: `ValidationError: field required`
**Solution**: Check that all required fields are provided in InputGoal:
```python
goal = InputGoal(
    goal_description="Your goal here",  # Required
    domain="your domain",               # Required
    # Optional fields...
)
```

**Error**: `ConnectionError: Could not connect to database`
**Solution**: Check DATABASE_URL and database status:
```bash
# Test connection
python -c "
from sqlalchemy import create_engine
engine = create_engine('your-database-url')
engine.connect()
print('Connection successful')
"
```

**Error**: `RateLimitError: API rate limit exceeded`
**Solution**: 
1. Check your API key limits
2. Implement backoff/retry logic
3. Reduce concurrent workflows
4. Upgrade your API plan

**Error**: `TimeoutError: Workflow timed out`
**Solution**: 
1. Increase `WORKFLOW_TIMEOUT` setting
2. Reduce goal complexity
3. Check for API response delays
4. Verify network connectivity

### API Errors

**Error**: `AuthenticationError: Invalid API key`
**Solution**:
1. Verify API key is correct
2. Check key permissions
3. Ensure key isn't expired
4. Test key with provider's API directly

**Error**: `InvalidRequestError: Model not found`
**Solution**:
1. Check model ID is correct
2. Verify you have access to the model
3. Try a different model
4. Check provider documentation

## Getting Additional Help

### Community Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/agent-forge/issues)
- **GitHub Discussions**: [Ask questions and get community help](https://github.com/your-org/agent-forge/discussions)
- **Discord**: [Real-time chat with community](https://discord.gg/agentforge)

### Professional Support

For production deployments or custom development needs:
- **Enterprise Support**: Contact sales@agentforge.com
- **Consulting Services**: Available for complex implementations
- **Training**: Team training sessions available

### Documentation

- **Complete Documentation**: [docs.agentforge.com](https://docs.agentforge.com)
- **API Reference**: [Complete API documentation](API_REFERENCE.md)
- **Developer Guide**: [Extension and customization guide](DEVELOPER_GUIDE.md)

---

**Still need help?** Don't hesitate to reach out to the community or create an issue. We're here to help you succeed with AgentForge!