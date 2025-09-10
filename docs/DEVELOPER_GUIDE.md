# 🛠️ AgentForge Developer Guide

Complete guide for extending, customizing, and contributing to AgentForge.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Extending AgentForge](#extending-agentforge)
- [Creating Custom Agents](#creating-custom-agents)
- [Adding New Tools](#adding-new-tools)
- [Testing Guidelines](#testing-guidelines)
- [Contributing](#contributing)
- [Advanced Customization](#advanced-customization)

## Development Environment Setup

### Prerequisites

- **Python 3.12+**
- **Git** for version control
- **IDE** with Python support (VS Code, PyCharm, etc.)
- **API Keys** for testing

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-org/agent-forge.git
cd agent-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "flake8>=6.0.0", 
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "pytest-cov>=4.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.0.0"
]
```

### Environment Configuration

Create a development `.env` file:

```bash
# Development environment
ENVIRONMENT=development

# API Keys for testing
OPENAI_API_KEY=your_development_key
OPENROUTER_API_KEY=your_development_key

# Test database
DATABASE_URL=sqlite:///test_agentforge.db

# Agent libraries for testing
AGENT_LIBRARY_PATH=./test_agents
TEAMS_LIBRARY_PATH=./test_teams

# Logging
LOG_LEVEL=DEBUG
```

## Architecture Deep Dive

### Core Components

AgentForge follows a modular architecture with clear separation of concerns:

```
agent-forge/
├── agents/                 # Core agent implementations
│   ├── base.py            # Base agent class
│   ├── engineering_manager.py
│   ├── systems_analyst.py
│   ├── talent_scout.py
│   ├── agent_developer.py
│   └── integration_architect.py
├── orchestrator.py        # Workflow orchestration
├── models/                # Data models and schemas
├── tools/                 # Custom tools and utilities
├── db/                   # Database management
├── docs/                 # Documentation
└── tests/               # Test suite
```

### Design Patterns

#### 1. Agent Pattern

All agents follow a consistent interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class BaseAgent(ABC):
    """Base class for all AgentForge agents."""
    
    def __init__(
        self,
        name: str,
        model_id: str = "deepseek/deepseek-v3.1",
        db_file: str = None,
        tools: List[Any] = None
    ):
        self.name = name
        self.model_id = model_id
        self.db_file = db_file or f"{name.lower().replace(' ', '_')}.db"
        self.tools = tools or []
        
        # Initialize Agno agent
        self._initialize_agent()
    
    @abstractmethod
    async def process(self, input_data: BaseModel) -> BaseModel:
        """Process input and return structured output."""
        pass
    
    @abstractmethod
    def _initialize_agent(self) -> None:
        """Initialize the underlying Agno agent."""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data structure."""
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and health information."""
        return {"status": "ready", "name": self.name}
```

#### 2. Workflow Pattern

Workflows follow a step-based pattern with checkpointing:

```python
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    name: str
    function: Callable
    input_schema: type
    output_schema: type
    required: bool = True
    retry_count: int = 3

class WorkflowManager:
    def __init__(self, steps: List[WorkflowStep]):
        self.steps = steps
        self.session_state = {}
        
    async def execute(self, initial_input: Any) -> Any:
        """Execute workflow with error handling and checkpointing."""
        current_input = initial_input
        
        for step in self.steps:
            try:
                # Validate input
                validated_input = step.input_schema(**current_input)
                
                # Execute step
                result = await step.function(validated_input)
                
                # Validate output
                validated_output = step.output_schema(**result)
                
                # Save checkpoint
                await self._save_checkpoint(step.name, validated_output)
                
                current_input = validated_output.model_dump()
                
            except Exception as e:
                if step.required:
                    raise WorkflowError(f"Required step {step.name} failed: {e}")
                else:
                    # Skip optional step and continue
                    continue
                    
        return current_input
```

#### 3. Factory Pattern

Factories provide flexible object creation:

```python
from typing import Type, Dict, Any

class AgentFactory:
    """Factory for creating agent instances."""
    
    _agent_registry: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]):
        """Register new agent type."""
        cls._agent_registry[name] = agent_class
    
    @classmethod
    def create_agent(cls, agent_type: str, **kwargs) -> BaseAgent:
        """Create agent instance by type."""
        if agent_type not in cls._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent_class = cls._agent_registry[agent_type]
        return agent_class(**kwargs)
    
    @classmethod
    def list_agent_types(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._agent_registry.keys())

# Register agents
AgentFactory.register_agent("engineering_manager", EngineeringManager)
AgentFactory.register_agent("systems_analyst", SystemsAnalyst)
# ... etc
```

## Extending AgentForge

### Adding New Agent Types

1. **Create the Agent Class**

```python
# agents/custom_agent.py
from .base import BaseAgent
from pydantic import BaseModel, Field
from typing import List

class CustomAgentInput(BaseModel):
    """Input schema for custom agent."""
    request: str = Field(..., description="What to process")
    context: List[str] = Field(default_factory=list)

class CustomAgentOutput(BaseModel):
    """Output schema for custom agent."""
    result: str = Field(..., description="Processing result")
    confidence: float = Field(..., description="Confidence score")

class CustomAgent(BaseAgent):
    """Custom agent for specialized processing."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Custom Agent", **kwargs)
        
    async def process(self, input_data: CustomAgentInput) -> CustomAgentOutput:
        """Process custom requests."""
        # Your custom logic here
        result = await self.agent.arun(
            message=f"Process this request: {input_data.request}",
            context={"background": input_data.context}
        )
        
        return CustomAgentOutput(
            result=result.content,
            confidence=0.95  # Example confidence score
        )
    
    def _initialize_agent(self):
        """Initialize with custom tools and configuration."""
        from agno.agent import Agent
        from agno.models.openrouter import OpenRouter
        
        self.agent = Agent(
            name=self.name,
            model=OpenRouter(id=self.model_id),
            instructions=[
                "You are a custom agent with specialized capabilities",
                "Focus on delivering high-quality, accurate results",
                "Always provide confidence scores for your outputs"
            ]
        )
```

2. **Register the Agent**

```python
# agents/__init__.py
from .custom_agent import CustomAgent
from .base import AgentFactory

# Register the new agent
AgentFactory.register_agent("custom", CustomAgent)
```

3. **Add Tests**

```python
# tests/test_custom_agent.py
import pytest
from agents.custom_agent import CustomAgent, CustomAgentInput

@pytest.mark.asyncio
async def test_custom_agent():
    agent = CustomAgent()
    
    input_data = CustomAgentInput(
        request="Test request",
        context=["Some context"]
    )
    
    result = await agent.process(input_data)
    
    assert result.result is not None
    assert 0 <= result.confidence <= 1
```

### Extending the Workflow

Add new steps to the core workflow:

```python
# orchestrator.py - Extended workflow
class ExtendedOrchestrator(EngineeringManager):
    """Extended orchestrator with additional workflow steps."""
    
    async def process(self, input_goal: InputGoal) -> TeamPackage:
        """Extended processing with additional steps."""
        
        # Standard workflow steps
        result = await super().process(input_goal)
        
        # Additional custom steps
        enhanced_result = await self._enhance_team_package(result)
        validated_result = await self._validate_team_quality(enhanced_result)
        
        return validated_result
    
    async def _enhance_team_package(self, team: TeamPackage) -> TeamPackage:
        """Add custom enhancements to the team package."""
        # Custom enhancement logic
        return team
    
    async def _validate_team_quality(self, team: TeamPackage) -> TeamPackage:
        """Perform additional quality validation."""
        # Custom validation logic
        return team
```

### Adding Custom Tools

Create tools that agents can use:

```python
# tools/custom_tools.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class CustomTool:
    """Custom tool for agent use."""
    
    name: str = "custom_processor"
    description: str = "Processes data with custom algorithms"
    
    class InputSchema(BaseModel):
        data: str = Field(..., description="Data to process")
        options: Dict[str, Any] = Field(default_factory=dict)
    
    class OutputSchema(BaseModel):
        processed_data: str = Field(..., description="Processed result")
        metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    
    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Execute the custom tool."""
        # Custom processing logic
        processed = f"Processed: {input_data.data}"
        
        return self.OutputSchema(
            processed_data=processed,
            metadata={
                "processing_time": 0.1,
                "algorithm": "custom_v1.0"
            }
        )

# Integration with agents
def add_custom_tools(agent):
    """Add custom tools to an agent."""
    agent.tools.append(CustomTool())
    return agent
```

## Creating Custom Agents

### Step-by-Step Guide

1. **Define the Agent's Purpose**
   - What specific capability does it provide?
   - How does it fit into the AgentForge workflow?
   - What are its inputs and outputs?

2. **Create Data Models**

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class MyAgentInput(BaseModel):
    """Clear, specific input schema."""
    primary_data: str = Field(..., description="Main data to process")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = Field(None, description="Additional context")

class MyAgentOutput(BaseModel):
    """Comprehensive output schema."""
    result: str = Field(..., description="Primary result")
    confidence_score: float = Field(..., description="Confidence (0-1)")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    recommendations: List[str] = Field(default_factory=list)
```

3. **Implement Core Logic**

```python
class MyCustomAgent(BaseAgent):
    """My custom agent implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(name="My Custom Agent", **kwargs)
        
    async def process(self, input_data: MyAgentInput) -> MyAgentOutput:
        """Main processing method."""
        
        # 1. Validate and prepare input
        self._validate_input(input_data)
        
        # 2. Execute core logic
        result = await self._execute_core_logic(input_data)
        
        # 3. Post-process and validate output
        output = self._prepare_output(result, input_data)
        
        return output
    
    async def _execute_core_logic(self, input_data: MyAgentInput) -> str:
        """Core agent logic."""
        # Use the underlying Agno agent
        response = await self.agent.arun(
            message=self._create_prompt(input_data),
            instructions=self._get_dynamic_instructions(input_data)
        )
        
        return response.content
    
    def _create_prompt(self, input_data: MyAgentInput) -> str:
        """Create dynamic prompt based on input."""
        return f"Process this data: {input_data.primary_data}"
    
    def _get_dynamic_instructions(self, input_data: MyAgentInput) -> List[str]:
        """Get context-specific instructions."""
        base_instructions = [
            "You are a specialized processing agent",
            "Focus on accuracy and detail",
            "Provide confidence scores for your outputs"
        ]
        
        # Add dynamic instructions based on input
        if input_data.context:
            base_instructions.append(f"Consider this context: {input_data.context}")
            
        return base_instructions
```

4. **Add Error Handling**

```python
from typing import Union
import logging

class AgentError(Exception):
    """Custom agent error."""
    pass

class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="My Custom Agent", **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process(self, input_data: MyAgentInput) -> MyAgentOutput:
        """Process with comprehensive error handling."""
        try:
            # Validation
            validation_result = self.validate_input(input_data)
            if not validation_result:
                raise AgentError("Input validation failed")
            
            # Processing
            result = await self._execute_with_retry(input_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}")
            
            # Return error response or re-raise based on strategy
            return self._handle_processing_error(e, input_data)
    
    async def _execute_with_retry(
        self, 
        input_data: MyAgentInput, 
        max_retries: int = 3
    ) -> MyAgentOutput:
        """Execute with automatic retry on failure."""
        
        for attempt in range(max_retries):
            try:
                return await self._execute_core_logic(input_data)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
```

### Best Practices for Custom Agents

#### ✅ Do:

1. **Use Type Hints**: Full typing for all methods and parameters
2. **Validate Inputs**: Always validate input data structure and content
3. **Handle Errors Gracefully**: Provide meaningful error messages and recovery
4. **Log Important Events**: Use structured logging for debugging
5. **Follow Naming Conventions**: Clear, descriptive names for classes and methods
6. **Write Tests**: Comprehensive test coverage for all functionality
7. **Document Thoroughly**: Clear docstrings and usage examples

#### ❌ Avoid:

1. **Hardcoded Values**: Use configuration and parameters instead
2. **Silent Failures**: Always handle and report errors appropriately
3. **Blocking Calls**: Use async/await for all I/O operations
4. **Memory Leaks**: Properly clean up resources and close connections
5. **Inconsistent Interfaces**: Follow the established patterns and schemas

## Adding New Tools

### Tool Architecture

Tools in AgentForge follow the Agno framework patterns:

```python
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class ToolBase:
    """Base class for all AgentForge tools."""
    
    name: str
    description: str
    
    class InputSchema(BaseModel):
        """Define tool input structure."""
        pass
    
    class OutputSchema(BaseModel):
        """Define tool output structure."""
        pass
    
    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Execute tool functionality."""
        raise NotImplementedError
    
    def validate_prerequisites(self) -> bool:
        """Check if tool can execute (API keys, dependencies, etc.)."""
        return True
```

### Example: Database Query Tool

```python
# tools/database_tools.py
import asyncio
import aiosqlite
from typing import List, Dict, Any

class DatabaseQueryTool(ToolBase):
    """Tool for executing database queries safely."""
    
    name = "database_query"
    description = "Execute read-only database queries"
    
    class InputSchema(BaseModel):
        query: str = Field(..., description="SQL query to execute")
        parameters: List[Any] = Field(default_factory=list, description="Query parameters")
        database_path: str = Field(..., description="Path to SQLite database")
    
    class OutputSchema(BaseModel):
        results: List[Dict[str, Any]] = Field(..., description="Query results")
        row_count: int = Field(..., description="Number of rows returned")
        execution_time: float = Field(..., description="Execution time in seconds")
    
    def __init__(self):
        self.allowed_operations = ["SELECT", "WITH"]  # Read-only operations
    
    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Execute database query safely."""
        
        # Validate query safety
        self._validate_query_safety(input_data.query)
        
        start_time = time.time()
        
        async with aiosqlite.connect(input_data.database_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute(input_data.query, input_data.parameters)
            rows = await cursor.fetchall()
            
            # Convert to dict format
            results = [dict(row) for row in rows]
        
        execution_time = time.time() - start_time
        
        return self.OutputSchema(
            results=results,
            row_count=len(results),
            execution_time=execution_time
        )
    
    def _validate_query_safety(self, query: str) -> None:
        """Ensure query is safe (read-only)."""
        query_upper = query.strip().upper()
        
        # Check if query starts with allowed operations
        if not any(query_upper.startswith(op) for op in self.allowed_operations):
            raise ValueError(f"Only {self.allowed_operations} operations allowed")
        
        # Check for dangerous keywords
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        if any(keyword in query_upper for keyword in dangerous_keywords):
            raise ValueError("Potentially dangerous SQL operation detected")
```

### Example: Web Scraping Tool

```python
# tools/web_tools.py
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

class WebScrapingTool(ToolBase):
    """Tool for safe web scraping operations."""
    
    name = "web_scraper"
    description = "Extract structured data from web pages"
    
    class InputSchema(BaseModel):
        url: str = Field(..., description="URL to scrape")
        selectors: Dict[str, str] = Field(..., description="CSS selectors to extract")
        timeout: int = Field(default=30, description="Request timeout in seconds")
        headers: Dict[str, str] = Field(default_factory=dict)
    
    class OutputSchema(BaseModel):
        extracted_data: Dict[str, Any] = Field(..., description="Extracted data")
        page_title: str = Field(..., description="Page title")
        status_code: int = Field(..., description="HTTP status code")
        response_time: float = Field(..., description="Response time in seconds")
    
    async def execute(self, input_data: InputSchema) -> OutputSchema:
        """Scrape web page and extract data."""
        
        start_time = time.time()
        
        # Set default headers
        headers = {
            "User-Agent": "AgentForge Web Scraper 1.0",
            **input_data.headers
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(input_data.timeout)) as session:
            async with session.get(input_data.url, headers=headers) as response:
                status_code = response.status
                content = await response.text()
        
        # Parse content
        soup = BeautifulSoup(content, 'html.parser')
        page_title = soup.title.string if soup.title else "No title"
        
        # Extract data using selectors
        extracted_data = {}
        for key, selector in input_data.selectors.items():
            elements = soup.select(selector)
            extracted_data[key] = [elem.get_text(strip=True) for elem in elements]
        
        response_time = time.time() - start_time
        
        return self.OutputSchema(
            extracted_data=extracted_data,
            page_title=page_title,
            status_code=status_code,
            response_time=response_time
        )
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_agents/        # Agent-specific tests
│   ├── test_tools/         # Tool tests
│   └── test_models/        # Data model tests
├── integration/            # Integration tests
│   ├── test_workflows/     # End-to-end workflow tests
│   └── test_orchestrator/  # Orchestrator integration tests
├── fixtures/               # Test data and fixtures
│   ├── sample_goals.json   # Sample input goals
│   ├── mock_responses/     # Mock API responses
│   └── test_agents/        # Test agent library
└── conftest.py            # Pytest configuration
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_agents/test_systems_analyst.py
import pytest
from unittest.mock import AsyncMock, patch
from agents.systems_analyst import SystemsAnalyst, InputGoal

class TestSystemsAnalyst:
    """Test suite for Systems Analyst agent."""
    
    @pytest.fixture
    def analyst(self):
        """Create test analyst instance."""
        return SystemsAnalyst(db_file=":memory:")  # In-memory database
    
    @pytest.fixture
    def sample_goal(self):
        """Create sample input goal."""
        return InputGoal(
            goal_description="Build a web application",
            domain="web development",
            complexity_level="medium"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_goal_success(self, analyst, sample_goal):
        """Test successful goal analysis."""
        
        result = await analyst.analyze_goal(sample_goal)
        
        assert result is not None
        assert len(result.team_composition) > 0
        assert result.goal_analysis is not None
    
    @pytest.mark.asyncio
    async def test_analyze_goal_with_constraints(self, analyst):
        """Test goal analysis with constraints."""
        
        goal = InputGoal(
            goal_description="Build an e-commerce platform",
            domain="web development", 
            complexity_level="high",
            constraints=["React frontend", "Node.js backend"]
        )
        
        result = await analyst.analyze_goal(goal)
        
        # Verify constraints are considered
        assert any("React" in str(result.team_composition))
        assert any("Node.js" in str(result.team_composition))
    
    @pytest.mark.asyncio
    async def test_analyze_invalid_goal(self, analyst):
        """Test handling of invalid goals."""
        
        with pytest.raises(ValueError):
            invalid_goal = InputGoal(
                goal_description="",  # Empty description
                domain="test"
            )
            await analyst.analyze_goal(invalid_goal)
    
    @patch('agents.systems_analyst.SystemsAnalyst.agent.arun')
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_arun, analyst, sample_goal):
        """Test error handling when underlying agent fails."""
        
        # Mock agent failure
        mock_arun.side_effect = Exception("API error")
        
        with pytest.raises(Exception):
            await analyst.analyze_goal(sample_goal)
```

#### Integration Tests

```python
# tests/integration/test_workflows/test_full_workflow.py
import pytest
from agents.engineering_manager import EngineeringManager, InputGoal

class TestFullWorkflow:
    """Integration tests for complete AgentForge workflow."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator."""
        return EngineeringManager(db_file=":memory:")
    
    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark slow tests
    async def test_simple_goal_workflow(self, orchestrator):
        """Test complete workflow with simple goal."""
        
        goal = InputGoal(
            goal_description="Create a simple blog website",
            domain="web development",
            complexity_level="low"
        )
        
        result = await orchestrator.process(goal)
        
        # Verify result structure
        assert result.team_name is not None
        assert len(result.team_members) > 0
        assert result.strategy_document is not None
        assert result.roster_documentation is not None
        
        # Verify team composition makes sense
        roles = [member.get("role", "") for member in result.team_members]
        expected_roles = ["Frontend Developer", "Backend Developer", "Content Manager"]
        assert any(role in str(roles) for role in expected_roles)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complex_goal_workflow(self, orchestrator):
        """Test workflow with complex goal requiring multiple agents."""
        
        goal = InputGoal(
            goal_description="Build a real-time collaborative document editing platform",
            domain="productivity software",
            complexity_level="enterprise",
            constraints=["Real-time sync", "Conflict resolution", "Version control"],
            success_criteria=["Sub-100ms sync", "99.9% uptime", "Support 1000+ concurrent users"]
        )
        
        result = await orchestrator.process(goal)
        
        # Verify complex goal generates appropriate team
        assert len(result.team_members) >= 6  # Enterprise complexity should have many agents
        assert result.quality_score > 0.7    # Should have high quality score
        
        # Check for expected specialized roles
        role_descriptions = str(result.roster_documentation).lower()
        expected_keywords = ["real-time", "sync", "conflict", "scalability"]
        for keyword in expected_keywords:
            assert keyword in role_descriptions
```

#### Mock and Fixtures

```python
# tests/conftest.py
import pytest
import json
from pathlib import Path

@pytest.fixture
def sample_goals():
    """Load sample goals from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_goals.json"
    with open(fixture_path) as f:
        return json.load(f)

@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "openai_chat": {
            "choices": [{"message": {"content": "Mock response"}}],
            "usage": {"total_tokens": 100}
        }
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # Reduce log noise in tests
```

### Test Configuration

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    -ra
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    asyncio: marks async tests
asyncio_mode = auto
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit -m "not slow"

# Run with coverage
pytest --cov=agents --cov=orchestrator

# Run specific test file
pytest tests/unit/test_agents/test_systems_analyst.py

# Run integration tests (slower)
pytest tests/integration -m "slow"

# Generate HTML coverage report
pytest --cov=agents --cov-report=html
```

## Contributing

### Code Style

AgentForge follows Python best practices:

- **PEP 8** for code formatting (enforced by Black)
- **Type hints** for all function parameters and return values
- **Docstrings** in Google style for all public methods
- **Import organization** using isort

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Run** the full test suite
5. **Update** documentation if needed
6. **Submit** a pull request with clear description

### Commit Message Format

```
feat: add new custom agent type support

- Add base class for custom agents
- Implement agent registration system
- Update documentation with examples
- Add comprehensive tests

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Advanced Customization

### Custom Model Integration

Add support for new language model providers:

```python
# models/custom_provider.py
from agno.models.base import Model
from typing import Dict, Any

class CustomModelProvider(Model):
    """Integration with custom model provider."""
    
    def __init__(
        self,
        id: str,
        api_key: str,
        base_url: str,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
    
    async def agenerate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using custom provider."""
        
        # Custom API call logic
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.id,
                "messages": messages,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
```

### Custom Database Backend

Replace SQLite with other database systems:

```python
# db/postgresql.py
from agno.db.base import Database
import asyncpg
from typing import Any, Dict, List, Optional

class PostgreSQLDatabase(Database):
    """PostgreSQL database backend for AgentForge."""
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)
    
    async def disconnect(self):
        """Close database connections."""
        if self.pool:
            await self.pool.close()
    
    async def create_tables(self):
        """Create required tables."""
        async with self.pool.acquire() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id UUID PRIMARY KEY,
                    goal_data JSONB,
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
    
    async def save_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Save workflow data."""
        async with self.pool.acquire() as connection:
            return await connection.fetchval(
                "INSERT INTO workflows (goal_data, status) VALUES ($1, $2) RETURNING id",
                workflow_data,
                "active"
            )
```

### Custom Knowledge Integration

Add specialized knowledge sources:

```python
# knowledge/custom_knowledge.py
from agno.knowledge.base import KnowledgeBase
from typing import List, Dict, Any

class CustomKnowledgeBase(KnowledgeBase):
    """Custom knowledge integration."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        super().__init__()
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search custom knowledge base."""
        
        async with aiohttp.ClientSession() as session:
            params = {
                "query": query,
                "limit": limit
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with session.get(
                f"{self.api_endpoint}/search",
                params=params,
                headers=headers
            ) as response:
                data = await response.json()
                return data["results"]
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add document to knowledge base."""
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "content": content,
                "metadata": metadata
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                f"{self.api_endpoint}/documents",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                return data["document_id"]
```

---

This developer guide provides comprehensive information for extending and contributing to AgentForge. For additional help:

- **Issues**: [GitHub Issues](https://github.com/your-org/agent-forge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/agent-forge/discussions)
- **Documentation**: [Complete Documentation](../README.md)

Happy coding! 🚀