"""
Test configuration and fixtures for AgentForge test suite.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.engineering_manager import InputGoal, ComplexityLevel
from agents.systems_analyst import SystemsAnalyst
from agents.agent_developer import AgentDeveloper
from agents.integration_architect import IntegrationArchitect


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_input_goal():
    """Sample InputGoal for testing."""
    return InputGoal(
        goal_description="Create a comprehensive task management system with real-time collaboration",
        domain="web development",
        complexity_level=ComplexityLevel.HIGH,
        timeline="3 months",
        constraints=["React frontend", "Node.js backend", "Mobile responsive"],
        success_criteria=[
            "User authentication and authorization",
            "Real-time task updates using WebSocket",
            "Team collaboration features",
            "Mobile-friendly responsive design",
            "Task prioritization and categorization"
        ]
    )


@pytest.fixture
def simple_input_goal():
    """Simple InputGoal for testing."""
    return InputGoal(
        goal_description="Build a basic blog website with user authentication",
        domain="web development",
        complexity_level=ComplexityLevel.MEDIUM,
        timeline="1 month",
        constraints=["Simple design", "Fast loading"],
        success_criteria=["User registration", "Blog posts", "Comments"]
    )


@pytest.fixture
def complex_input_goal():
    """Complex InputGoal for testing."""
    return InputGoal(
        goal_description="Build an AI-powered e-commerce platform with personalized recommendations, real-time inventory management, multi-vendor support, and advanced analytics dashboard",
        domain="e-commerce",
        complexity_level=ComplexityLevel.ENTERPRISE,
        timeline="8 months",
        constraints=[
            "Microservices architecture",
            "Cloud-native deployment",
            "PCI DSS compliance",
            "Multi-region support"
        ],
        success_criteria=[
            "Handle 50K concurrent users",
            "Process 1M+ products from 1000+ vendors",
            "AI recommendations improve conversion by 25%",
            "Real-time inventory sync across all channels",
            "Comprehensive analytics and reporting",
            "Mobile-first responsive design",
            "99.99% uptime SLA"
        ]
    )


@pytest.fixture
def mock_claude_flow_hooks():
    """Mock Claude Flow hooks for testing."""
    hooks = Mock()
    hooks.pre_task = AsyncMock(return_value="pre-task executed")
    hooks.post_edit = AsyncMock(return_value="post-edit executed") 
    hooks.notify = AsyncMock(return_value="notification sent")
    hooks.post_task = AsyncMock(return_value="post-task executed")
    return hooks


@pytest.fixture
def mock_agno_agent():
    """Mock Agno Agent for testing."""
    agent = Mock()
    agent.arun = AsyncMock(return_value="Mock agent response")
    agent.aprint_response = AsyncMock(return_value="Mock agent response")
    agent.name = "MockAgent"
    agent.model = Mock()
    agent.tools = []
    return agent


@pytest.fixture
def mock_strategy_document():
    """Mock strategy document for testing."""
    return """
    # Strategy Document

    ## Goal Analysis
    The goal requires a comprehensive task management system with the following key components:
    - User authentication and session management
    - Real-time collaboration capabilities  
    - Task management and organization
    - Mobile-responsive interface
    - Database for persistent storage

    ## Team Composition
    Based on the analysis, the following agent roles are required:

    ### 1. Authentication Specialist
    - **Responsibilities**: User registration, login, session management, security
    - **Capabilities**: OAuth integration, JWT tokens, security best practices
    - **Priority**: High

    ### 2. Frontend Developer  
    - **Responsibilities**: React UI development, responsive design, UX optimization
    - **Capabilities**: React, CSS frameworks, responsive design, accessibility
    - **Priority**: High

    ### 3. Backend API Developer
    - **Responsibilities**: REST API development, business logic, data validation
    - **Capabilities**: Node.js, Express, API design, database integration
    - **Priority**: High

    ### 4. Real-time Systems Engineer
    - **Responsibilities**: WebSocket implementation, real-time updates, performance
    - **Capabilities**: WebSocket, Socket.io, real-time architecture, scaling
    - **Priority**: Medium

    ### 5. Database Architect
    - **Responsibilities**: Database design, optimization, data modeling
    - **Capabilities**: SQL/NoSQL, database performance, data integrity
    - **Priority**: Medium

    ## Team Structure
    - **Topology**: Hierarchical with Frontend/Backend separation
    - **Coordination**: API-first approach with clear interfaces
    - **Decision Making**: Technical leads for frontend/backend domains
    """


@pytest.fixture 
def mock_scouting_report():
    """Mock scouting report for testing."""
    return """
    # Scouting Report

    ## Matches Found
    - **existing_auth_agent**: 85% match for Authentication Specialist role
    - **existing_ui_agent**: 70% match for Frontend Developer role  

    ## Gaps Identified
    The following roles need new agents:
    - **Backend API Developer**: No existing match found
    - **Real-time Systems Engineer**: No suitable existing agent
    - **Database Architect**: Partial match but insufficient capabilities

    ## Reuse Analysis
    - 40% of required capabilities can be fulfilled by existing agents
    - 60% require new agent development
    - High-priority roles (Authentication, Frontend) have good existing options
    """


@pytest.fixture
def mock_agent_specifications():
    """Mock agent specifications for testing."""
    return [
        {
            "name": "BackendAPIDeveloper",
            "role": "Backend API Developer", 
            "system_prompt": "You are a Backend API Developer specialized in Node.js...",
            "capabilities": ["REST API design", "Node.js", "Express", "Database integration"],
            "tools_required": ["ReasoningTools", "KnowledgeTools"],
            "priority": "high"
        },
        {
            "name": "RealtimeSystemsEngineer", 
            "role": "Real-time Systems Engineer",
            "system_prompt": "You are a Real-time Systems Engineer specialized in WebSocket...",
            "capabilities": ["WebSocket", "Socket.io", "Real-time architecture", "Performance optimization"],
            "tools_required": ["ReasoningTools", "PerformanceTools"],
            "priority": "medium"  
        }
    ]


@pytest.fixture
def mock_roster_documentation():
    """Mock roster documentation for testing."""
    return """
    # Team Roster Documentation

    ## Team Purpose
    Develop a comprehensive task management system with real-time collaboration capabilities.

    ## Team Members

    ### Existing Agents (Reused)
    - **AuthenticationSpecialist**: Handle user auth and security
    - **FrontendDeveloper**: React UI and responsive design  

    ### New Agents (Created)
    - **BackendAPIDeveloper**: REST API and business logic
    - **RealtimeSystemsEngineer**: WebSocket and real-time features
    - **DatabaseArchitect**: Database design and optimization

    ## Workflow Process
    1. **Requirements Analysis**: All agents review requirements
    2. **Architecture Design**: Database and API architects collaborate  
    3. **Implementation Phase**: Frontend and backend development in parallel
    4. **Integration Phase**: All components integrated and tested
    5. **Deployment Phase**: System deployed and monitored

    ## Communication Protocols
    - Daily standups between all agents
    - API contracts defined by Backend developer
    - Real-time integration coordinated between Frontend and Real-time engineers
    - Database schema approved by all stakeholders
    """


# Pytest configuration
pytest_plugins = ['asyncio']