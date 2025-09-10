"""
Unit tests for Systems Analyst - The Strategist.

Tests the expert in decomposing complex goals into discrete, manageable roles 
and capabilities, defining ideal team structures without regard for existing resources.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agents.systems_analyst import (
    SystemsAnalyst,
    InputGoal,
    AgentRole, 
    TeamStructure,
    StrategyDocument
)


class TestInputGoalModel:
    """Test InputGoal Pydantic model validation."""
    
    def test_input_goal_creation_minimal(self):
        """Test creating InputGoal with minimal required fields."""
        goal = InputGoal(description="Build a website")
        
        assert goal.description == "Build a website"
        assert goal.context is None
        assert goal.success_criteria is None
        assert goal.domain is None
        assert goal.complexity is None
    
    def test_input_goal_creation_complete(self):
        """Test creating InputGoal with all fields."""
        goal = InputGoal(
            description="Build an e-commerce platform",
            context="For a startup with limited budget",
            success_criteria=["Handle 1000 users", "Payment processing", "Mobile responsive"],
            domain="E-commerce",
            complexity="High"
        )
        
        assert goal.description == "Build an e-commerce platform"
        assert goal.context == "For a startup with limited budget"
        assert len(goal.success_criteria) == 3
        assert goal.domain == "E-commerce"
        assert goal.complexity == "High"
    
    def test_input_goal_validation(self):
        """Test InputGoal field validation."""
        # Test required description field
        with pytest.raises(ValueError):
            InputGoal()
        
        # Test empty description
        with pytest.raises(ValueError):
            InputGoal(description="")


class TestAgentRoleModel:
    """Test AgentRole Pydantic model validation."""
    
    def test_agent_role_creation(self):
        """Test creating valid AgentRole."""
        role = AgentRole(
            name="Frontend Developer",
            title="Senior Frontend Developer",
            core_responsibilities=["UI development", "User experience design"],
            required_capabilities=["React", "CSS", "JavaScript"],
            interaction_patterns={"Backend Developer": "API integration"},
            success_metrics=["User satisfaction", "Performance metrics"],
            priority_level="high"
        )
        
        assert role.name == "Frontend Developer"
        assert role.title == "Senior Frontend Developer"
        assert len(role.core_responsibilities) == 2
        assert len(role.required_capabilities) == 3
        assert role.priority_level == "high"
    
    def test_agent_role_required_fields(self):
        """Test AgentRole required field validation."""
        with pytest.raises(ValueError):
            AgentRole(name="Test", title="Test Title")  # Missing required fields


class TestTeamStructureModel:
    """Test TeamStructure Pydantic model validation."""
    
    def test_team_structure_creation(self):
        """Test creating valid TeamStructure."""
        structure = TeamStructure(
            topology="hierarchical",
            coordination_mechanism="daily standups",
            decision_making_process="consensus",
            communication_protocols=["Slack", "Email", "Video calls"],
            workflow_stages=["Planning", "Development", "Testing", "Deployment"]
        )
        
        assert structure.topology == "hierarchical"
        assert structure.coordination_mechanism == "daily standups"
        assert len(structure.communication_protocols) == 3
        assert len(structure.workflow_stages) == 4


class TestSystemsAnalystInitialization:
    """Test Systems Analyst initialization and setup."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        analyst = SystemsAnalyst()
        
        assert analyst.agent is not None
        assert analyst.agent.name == "SystemsAnalyst"
        assert analyst.agent.markdown is True
        assert analyst.agent.add_history_to_context is True
    
    def test_initialization_with_knowledge_base(self, temp_directory):
        """Test initialization with knowledge base path."""
        # Create a test knowledge file
        test_file = os.path.join(temp_directory, "test_knowledge.md")
        with open(test_file, 'w') as f:
            f.write("# Test Knowledge\nSample Agno patterns and documentation.")
        
        analyst = SystemsAnalyst(knowledge_base_path=temp_directory)
        
        assert analyst.agent is not None
        # Knowledge base should be initialized (we can't easily test the internal state)
        assert hasattr(analyst.agent, 'tools')
        assert len(analyst.agent.tools) >= 2  # ReasoningTools and KnowledgeTools
    
    def test_agent_tools_configuration(self):
        """Test that agent has proper tools configured."""
        analyst = SystemsAnalyst()
        
        tool_names = [tool.__class__.__name__ for tool in analyst.agent.tools]
        assert 'ReasoningTools' in tool_names
        assert 'KnowledgeTools' in tool_names
    
    def test_agent_instructions(self):
        """Test agent instructions content."""
        analyst = SystemsAnalyst()
        
        instructions = analyst.agent.instructions
        assert "Systems Analyst" in instructions
        assert "Strategist" in instructions
        assert "decomposing complex goals" in instructions.lower()
        assert "ideal team structures" in instructions.lower()
        assert "reasoning tools" in instructions.lower()


class TestSystemsAnalystGoalAnalysis:
    """Test goal analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_goal_simple(self):
        """Test analyzing a simple goal."""
        analyst = SystemsAnalyst()
        
        # Mock the agent response
        mock_response = """
        # Strategy Document
        
        ## Goal Analysis
        Build a basic blog website with authentication and commenting.
        
        ## Team Composition
        - Frontend Developer: Handle UI and user experience
        - Backend Developer: API and database management
        - Authentication Specialist: User management and security
        
        ## Team Structure
        Small team with direct communication and agile methodology.
        """
        analyst.agent.arun = AsyncMock(return_value=mock_response)
        
        goal = InputGoal(
            description="Build a blog website with user authentication and commenting",
            domain="Web Development",
            complexity="Medium"
        )
        
        result = await analyst.analyze_goal(goal)
        
        assert result == mock_response
        analyst.agent.arun.assert_called_once()
        
        # Verify the prompt includes goal details
        call_args = analyst.agent.arun.call_args[0][0]
        assert goal.description in call_args
        assert goal.domain in call_args
        assert goal.complexity in call_args
    
    @pytest.mark.asyncio
    async def test_analyze_goal_complex(self):
        """Test analyzing a complex goal with full details."""
        analyst = SystemsAnalyst()
        
        mock_response = """
        # Comprehensive Strategy Document
        
        ## Goal Analysis
        Complex e-commerce platform requiring multiple specialized roles.
        
        ## Team Composition
        - Solution Architect: Overall system design
        - Frontend Developers (2): React UI development  
        - Backend Developers (3): Microservices and APIs
        - DevOps Engineer: Infrastructure and deployment
        - QA Engineers (2): Testing and quality assurance
        - Security Specialist: Security implementation
        
        ## Team Structure
        Hierarchical structure with technical leads and cross-functional collaboration.
        """
        analyst.agent.arun = AsyncMock(return_value=mock_response)
        
        complex_goal = InputGoal(
            description="Build an AI-powered e-commerce platform with personalized recommendations",
            context="Startup with $2M funding, targeting 100K+ users",
            success_criteria=[
                "Handle 10K concurrent users",
                "AI-powered recommendations",
                "Mobile-first design",
                "Real-time inventory management"
            ],
            domain="E-commerce / AI",
            complexity="Very High"
        )
        
        result = await analyst.analyze_goal(complex_goal)
        
        assert result == mock_response
        analyst.agent.arun.assert_called_once()
        
        # Verify complex goal details are included in prompt
        call_args = analyst.agent.arun.call_args[0][0]
        assert complex_goal.description in call_args
        assert complex_goal.context in call_args
        assert "Handle 10K concurrent users" in call_args
        assert complex_goal.domain in call_args
    
    @pytest.mark.asyncio
    async def test_analyze_goal_with_empty_optional_fields(self):
        """Test analyzing goal with None/empty optional fields."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value="Strategy response")
        
        goal = InputGoal(
            description="Simple goal",
            context=None,
            success_criteria=None,
            domain=None,
            complexity=None
        )
        
        result = await analyst.analyze_goal(goal)
        
        assert result == "Strategy response"
        
        # Verify prompt handles None values gracefully
        call_args = analyst.agent.arun.call_args[0][0]
        assert "Not specified" in call_args  # Should replace None with "Not specified"
    
    @pytest.mark.asyncio
    async def test_quick_analysis(self):
        """Test quick analysis method."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value="Quick analysis result")
        
        result = await analyst.quick_analysis("Create a task management app")
        
        assert result == "Quick analysis result"
        analyst.agent.arun.assert_called_once()
        
        # Verify it creates an InputGoal internally
        call_args = analyst.agent.arun.call_args[0][0]
        assert "Create a task management app" in call_args


class TestSystemsAnalystDocumentGeneration:
    """Test strategy document generation functionality."""
    
    def test_create_strategy_document_default_path(self, temp_directory):
        """Test creating strategy document with default path."""
        analyst = SystemsAnalyst()
        
        analysis_result = """
        ## Team Analysis
        Based on the goal, we need:
        - 3 Frontend developers
        - 2 Backend developers
        - 1 DevOps engineer
        
        ## Architecture
        Microservices with React frontend.
        """
        
        # Change to temp directory to avoid creating files in project
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_directory)
            doc_path = analyst.create_strategy_document(analysis_result)
            
            assert doc_path == "agent-strategy.md"
            assert os.path.exists(doc_path)
            
            # Check file content
            with open(doc_path, 'r') as f:
                content = f.read()
            
            assert "Agent Strategy Document" in content
            assert "Systems Analyst (AgentForge)" in content
            assert analysis_result in content
            assert "Generated by:" in content
            assert "Date:" in content
        finally:
            os.chdir(original_cwd)
    
    def test_create_strategy_document_custom_path(self, temp_directory):
        """Test creating strategy document with custom path."""
        analyst = SystemsAnalyst()
        
        analysis_result = "Test analysis result"
        custom_path = os.path.join(temp_directory, "custom-strategy.md")
        
        doc_path = analyst.create_strategy_document(analysis_result, custom_path)
        
        assert doc_path == custom_path
        assert os.path.exists(custom_path)
        
        with open(custom_path, 'r') as f:
            content = f.read()
        
        assert analysis_result in content
        assert "Systems Analyst (AgentForge)" in content
    
    def test_create_strategy_document_formatting(self, temp_directory):
        """Test strategy document formatting and structure."""
        analyst = SystemsAnalyst()
        
        analysis_result = "Detailed strategy analysis"
        doc_path = os.path.join(temp_directory, "test-strategy.md")
        
        result_path = analyst.create_strategy_document(analysis_result, doc_path)
        
        with open(result_path, 'r') as f:
            content = f.read()
        
        # Check document structure
        assert content.startswith("# Agent Strategy Document")
        assert "**Generated by:**" in content
        assert "**Date:**" in content
        assert "**Document Type:**" in content
        assert "---" in content  # Section separators
        assert content.endswith("against existing agent resources.\n")


class TestSystemsAnalystErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test handling of agent execution errors."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(side_effect=Exception("Agent execution failed"))
        
        goal = InputGoal(description="Test goal")
        
        with pytest.raises(Exception, match="Agent execution failed"):
            await analyst.analyze_goal(goal)
    
    def test_file_creation_error_handling(self, temp_directory):
        """Test handling of file creation errors."""
        analyst = SystemsAnalyst()
        
        # Try to write to a directory that doesn't exist
        invalid_path = "/nonexistent/directory/strategy.md"
        
        with pytest.raises(FileNotFoundError):
            analyst.create_strategy_document("test content", invalid_path)
    
    @pytest.mark.asyncio
    async def test_empty_goal_handling(self):
        """Test handling of empty or minimal goals."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value="Minimal strategy")
        
        # Test with minimal description
        goal = InputGoal(description="X")  # Single character
        result = await analyst.analyze_goal(goal)
        
        assert result == "Minimal strategy"
        analyst.agent.arun.assert_called_once()


class TestSystemsAnalystIntegration:
    """Integration tests with Agno framework components."""
    
    def test_knowledge_tools_integration(self):
        """Test integration with knowledge tools."""
        analyst = SystemsAnalyst()
        
        # Verify KnowledgeTools is configured correctly
        knowledge_tools = None
        for tool in analyst.agent.tools:
            if tool.__class__.__name__ == 'KnowledgeTools':
                knowledge_tools = tool
                break
        
        assert knowledge_tools is not None
        assert knowledge_tools.think is True
        assert knowledge_tools.search is True
        assert knowledge_tools.analyze is True
        assert knowledge_tools.add_few_shot is True
    
    def test_reasoning_tools_integration(self):
        """Test integration with reasoning tools."""
        analyst = SystemsAnalyst()
        
        # Verify ReasoningTools is configured correctly
        reasoning_tools = None
        for tool in analyst.agent.tools:
            if tool.__class__.__name__ == 'ReasoningTools':
                reasoning_tools = tool
                break
        
        assert reasoning_tools is not None
        assert reasoning_tools.think is True
        assert reasoning_tools.analyze is True
        assert reasoning_tools.add_instructions is True
        assert reasoning_tools.add_few_shot is True
    
    def test_model_configuration(self):
        """Test model configuration."""
        analyst = SystemsAnalyst()
        
        # Should use OpenRouter with deepseek model
        assert analyst.agent.model is not None
        # Can't easily test model details without making actual calls


class TestSystemsAnalystPerformance:
    """Performance tests for Systems Analyst."""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self):
        """Test concurrent analysis of multiple goals."""
        analyst = SystemsAnalyst()
        
        # Mock fast responses
        analyst.agent.arun = AsyncMock(return_value="Fast analysis")
        
        goals = [
            InputGoal(description=f"Goal {i}", domain=f"Domain {i}")
            for i in range(5)
        ]
        
        # Analyze all goals concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[analyst.analyze_goal(goal) for goal in goals])
        end_time = asyncio.get_event_loop().time()
        
        # Verify all succeeded
        assert len(results) == 5
        assert all(result == "Fast analysis" for result in results)
        
        # Verify reasonable performance
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should complete quickly with mocked responses
    
    def test_large_document_generation(self, temp_directory):
        """Test generation of large strategy documents."""
        analyst = SystemsAnalyst()
        
        # Create a large analysis result
        large_analysis = "# Large Analysis\n" + "Analysis content. " * 1000
        doc_path = os.path.join(temp_directory, "large-strategy.md")
        
        result_path = analyst.create_strategy_document(large_analysis, doc_path)
        
        assert os.path.exists(result_path)
        
        # Verify file was written correctly
        with open(result_path, 'r') as f:
            content = f.read()
        
        assert large_analysis in content
        assert len(content) > len(large_analysis)  # Should include formatting


class TestSystemsAnalystSpecialization:
    """Test Systems Analyst specialization capabilities."""
    
    @pytest.mark.asyncio
    async def test_domain_specific_analysis(self):
        """Test analysis adapts to different domains."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value="Domain-specific analysis")
        
        domains = ["E-commerce", "Healthcare", "Finance", "Education", "Gaming"]
        
        for domain in domains:
            goal = InputGoal(
                description=f"Build a {domain.lower()} application",
                domain=domain
            )
            
            result = await analyst.analyze_goal(goal)
            assert result == "Domain-specific analysis"
            
            # Verify domain is included in the analysis prompt
            call_args = analyst.agent.arun.call_args[0][0]
            assert domain in call_args
    
    @pytest.mark.asyncio 
    async def test_complexity_aware_analysis(self):
        """Test analysis adapts to different complexity levels."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value="Complexity-aware analysis")
        
        complexity_levels = ["Low", "Medium", "High", "Very High", "Enterprise"]
        
        for complexity in complexity_levels:
            goal = InputGoal(
                description="Test application",
                complexity=complexity
            )
            
            result = await analyst.analyze_goal(goal)
            assert result == "Complexity-aware analysis"
            
            # Verify complexity is considered in analysis
            call_args = analyst.agent.arun.call_args[0][0]
            assert complexity in call_args


# Fixtures for specialized testing
@pytest.fixture
def sample_team_analysis():
    """Sample team analysis result for testing."""
    return """
    # Strategy Document

    ## Goal Analysis
    The application requires a modern web development approach with the following key components:
    - User interface and experience design
    - Backend API and business logic
    - Database design and management
    - Authentication and security
    - DevOps and deployment pipeline

    ## Team Composition

    ### Frontend Developer
    - **Role**: User Interface Development
    - **Responsibilities**: React development, responsive design, user experience
    - **Capabilities**: React, CSS, JavaScript, UI/UX design principles
    - **Priority**: High

    ### Backend Developer  
    - **Role**: API and Business Logic
    - **Responsibilities**: REST API design, business logic, database integration
    - **Capabilities**: Node.js, Express, API design, database management
    - **Priority**: High

    ### DevOps Engineer
    - **Role**: Infrastructure and Deployment
    - **Responsibilities**: CI/CD pipeline, cloud infrastructure, monitoring
    - **Capabilities**: Docker, Kubernetes, AWS/Azure, monitoring tools
    - **Priority**: Medium

    ## Team Structure
    - **Topology**: Small cross-functional team
    - **Coordination**: Daily standups and sprint planning
    - **Communication**: Slack for async, video calls for sync
    """


class TestSystemsAnalystOutputQuality:
    """Test quality and completeness of Systems Analyst outputs."""
    
    @pytest.mark.asyncio
    async def test_strategy_document_completeness(self, sample_team_analysis):
        """Test that strategy documents are complete and well-structured."""
        analyst = SystemsAnalyst()
        analyst.agent.arun = AsyncMock(return_value=sample_team_analysis)
        
        goal = InputGoal(
            description="Build a task management application",
            domain="Web Development",
            complexity="High"
        )
        
        result = await analyst.analyze_goal(goal)
        
        # Verify key sections are present
        assert "Goal Analysis" in result
        assert "Team Composition" in result  
        assert "Team Structure" in result
        assert "Frontend Developer" in result
        assert "Backend Developer" in result
        assert "Priority" in result
        assert "Capabilities" in result
    
    def test_document_format_validation(self, temp_directory, sample_team_analysis):
        """Test that generated documents follow proper format."""
        analyst = SystemsAnalyst()
        
        doc_path = os.path.join(temp_directory, "format-test.md")
        result_path = analyst.create_strategy_document(sample_team_analysis, doc_path)
        
        with open(result_path, 'r') as f:
            content = f.read()
        
        # Check document structure and formatting
        lines = content.split('\n')
        assert lines[0] == "# Agent Strategy Document"
        assert any(line.startswith("**Generated by:**") for line in lines)
        assert any(line.startswith("**Date:**") for line in lines)
        assert any(line.startswith("**Document Type:**") for line in lines)
        assert "---" in content
        assert content.endswith("against existing agent resources.\n")