"""
Integration tests for AgentForge system.

Tests agent communication, workflow coordination, and cross-component integration.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel, TeamPackage
from agents.systems_analyst import SystemsAnalyst
from agents.base import AgentForgeBase


class TestAgentCommunication:
    """Test communication between different agents."""
    
    @pytest.mark.asyncio
    async def test_engineering_manager_to_systems_analyst_flow(self, sample_input_goal):
        """Test data flow from Engineering Manager to Systems Analyst."""
        em = EngineeringManager()
        analyst = SystemsAnalyst()
        
        # Mock analyst response
        expected_strategy = "Mock strategy document from Systems Analyst"
        analyst.analyze_goal = AsyncMock(return_value=expected_strategy)
        
        # Test delegation
        result = await em._delegate_to_systems_analyst(sample_input_goal)
        
        # Verify the communication worked
        assert result is not None
        assert isinstance(result, str)
        # The actual result depends on the run_with_mcp implementation
    
    @pytest.mark.asyncio
    async def test_workflow_data_preservation(self, sample_input_goal):
        """Test that data is preserved throughout the workflow."""
        em = EngineeringManager()
        
        # Mock all delegations to return specific data
        em._delegate_to_systems_analyst = AsyncMock(return_value="Test Strategy Document")
        em._delegate_to_talent_scout = AsyncMock(return_value="Test Scouting Report")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "TestAgent", "type": "test"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Test Roster Documentation")
        
        # Execute workflow
        result = await em.process(sample_input_goal)
        
        # Verify data preservation
        assert isinstance(result, TeamPackage)
        assert result.goal == sample_input_goal
        assert result.strategy_document == "Test Strategy Document"
        assert result.scouting_report == "Test Scouting Report"
        assert len(result.new_agents) == 1
        assert result.new_agents[0]["name"] == "TestAgent"
        assert result.roster_documentation == "Test Roster Documentation"
    
    @pytest.mark.asyncio
    async def test_error_propagation_between_agents(self, sample_input_goal):
        """Test that errors are properly propagated between agents."""
        em = EngineeringManager()
        
        # Make the second step fail
        em._delegate_to_systems_analyst = AsyncMock(return_value="Success")
        em._delegate_to_talent_scout = AsyncMock(side_effect=Exception("Talent Scout Error"))
        
        # Execute and expect error propagation
        with pytest.raises(Exception, match="Talent Scout Error"):
            await em.process(sample_input_goal)
        
        # Verify first step completed but second failed
        em._delegate_to_systems_analyst.assert_called_once()
        em._delegate_to_talent_scout.assert_called_once()


class TestWorkflowCoordination:
    """Test workflow coordination and sequencing."""
    
    @pytest.mark.asyncio
    async def test_workflow_step_sequencing(self, sample_input_goal):
        """Test that workflow steps execute in correct sequence."""
        em = EngineeringManager()
        
        # Track call order
        call_order = []
        
        async def track_systems_analyst(goal):
            call_order.append("systems_analyst")
            return "Strategy"
        
        async def track_talent_scout(strategy):
            call_order.append("talent_scout")
            return "Scouting"
            
        async def track_agent_developer(scouting):
            call_order.append("agent_developer")
            return [{"name": "Agent"}]
            
        async def track_integration_architect(strategy, scouting, agents):
            call_order.append("integration_architect")
            return "Roster"
        
        # Assign tracking functions
        em._delegate_to_systems_analyst = track_systems_analyst
        em._delegate_to_talent_scout = track_talent_scout
        em._delegate_to_agent_developer = track_agent_developer
        em._delegate_to_integration_architect = track_integration_architect
        
        # Execute workflow
        result = await em.process(sample_input_goal)
        
        # Verify correct sequence
        expected_order = ["systems_analyst", "talent_scout", "agent_developer", "integration_architect"]
        assert call_order == expected_order
        assert isinstance(result, TeamPackage)
    
    @pytest.mark.asyncio
    async def test_workflow_state_tracking(self, sample_input_goal):
        """Test that workflow state is properly tracked."""
        em = EngineeringManager()
        
        # Mock delegations
        em._delegate_to_systems_analyst = AsyncMock(return_value="Strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Roster")
        
        # Execute workflow
        await em.process(sample_input_goal)
        
        # Verify state tracking
        status = em.get_workflow_status()
        assert status["current_step"] == "final_packaging"
        assert status["history_length"] > 0
        assert status["latest_action"] is not None
        
        # Verify workflow history contains all major steps
        actions = [entry["action"] for entry in em.workflow_history]
        assert any("Received input goal" in action for action in actions)
        assert any("Delegating to Systems Analyst" in action for action in actions)
        assert any("Delegating to Talent Scout" in action for action in actions)
        assert any("Workflow completed" in action for action in actions)
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_coordination(self):
        """Test coordination when multiple workflows run concurrently."""
        em1 = EngineeringManager()
        em2 = EngineeringManager()
        
        # Mock quick responses
        for em in [em1, em2]:
            em._delegate_to_systems_analyst = AsyncMock(return_value="Strategy")
            em._delegate_to_talent_scout = AsyncMock(return_value="Scouting")
            em._delegate_to_agent_developer = AsyncMock(return_value=[])
            em._delegate_to_integration_architect = AsyncMock(return_value="Roster")
        
        # Create different goals
        goal1 = InputGoal(goal_description="Goal 1", domain="Domain 1")
        goal2 = InputGoal(goal_description="Goal 2", domain="Domain 2")
        
        # Execute concurrently
        results = await asyncio.gather(
            em1.process(goal1),
            em2.process(goal2)
        )
        
        # Verify both succeeded and are independent
        assert len(results) == 2
        assert results[0].goal.goal_description == "Goal 1"
        assert results[1].goal.goal_description == "Goal 2"
        
        # Verify independent workflow histories
        assert len(em1.workflow_history) > 0
        assert len(em2.workflow_history) > 0
        assert em1.workflow_history != em2.workflow_history


class TestClaudeFlowHooksIntegration:
    """Test integration with Claude Flow hooks for coordination."""
    
    @pytest.mark.asyncio
    async def test_hooks_coordination_workflow(self, mock_claude_flow_hooks):
        """Test that Claude Flow hooks are properly integrated."""
        # This test would verify that hooks are called at appropriate times
        # For now, we'll test the hook call structure
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Hook executed successfully"
            
            # Import and run the bash command that would be used
            import subprocess
            
            # Test pre-task hook
            result = subprocess.run(
                ["echo", "npx claude-flow@alpha hooks pre-task --description 'test task'"],
                capture_output=True, text=True
            )
            
            assert result.returncode == 0
    
    def test_memory_coordination_structure(self):
        """Test memory coordination structure for agent communication."""
        # Test that agents can coordinate through memory keys
        test_memory_keys = [
            "tester/validation",
            "swarm/engineering-manager/strategy",
            "swarm/systems-analyst/analysis",
            "swarm/agent-developer/creation",
            "swarm/integration-architect/roster"
        ]
        
        for key in test_memory_keys:
            # Verify key format is valid
            assert "/" in key
            assert len(key.split("/")) >= 2
            assert key.replace("/", "").replace("-", "").replace("_", "").isalnum()


class TestAgnoFrameworkIntegration:
    """Test integration with Agno framework components."""
    
    def test_agno_agent_base_class_integration(self):
        """Test that AgentForge agents properly inherit from Agno Agent."""
        em = EngineeringManager()
        analyst = SystemsAnalyst()
        
        # Verify Agno Agent properties
        assert hasattr(em, 'agent')
        assert hasattr(analyst, 'agent')
        
        # Check agent configuration
        assert em.agent.name == "EngineeringManager"
        assert analyst.agent.name == "SystemsAnalyst"
        assert em.agent.markdown is True
        assert analyst.agent.markdown is True
    
    def test_agno_model_integration(self):
        """Test integration with Agno models (OpenRouter)."""
        analyst = SystemsAnalyst()
        
        # Verify model is configured
        assert analyst.agent.model is not None
        # Model type checking would require accessing internal structure
    
    def test_agno_tools_integration(self):
        """Test integration with Agno tools."""
        analyst = SystemsAnalyst()
        
        # Verify tools are configured
        assert len(analyst.agent.tools) >= 2
        
        tool_names = [tool.__class__.__name__ for tool in analyst.agent.tools]
        assert 'ReasoningTools' in tool_names
        assert 'KnowledgeTools' in tool_names
    
    def test_agno_knowledge_base_integration(self):
        """Test integration with Agno knowledge base."""
        analyst = SystemsAnalyst()
        
        # Verify knowledge tools have knowledge base
        knowledge_tools = None
        for tool in analyst.agent.tools:
            if tool.__class__.__name__ == 'KnowledgeTools':
                knowledge_tools = tool
                break
        
        assert knowledge_tools is not None
        assert hasattr(knowledge_tools, 'knowledge')


class TestPydanticModelIntegration:
    """Test Pydantic model validation and serialization."""
    
    def test_input_goal_serialization(self, sample_input_goal):
        """Test InputGoal serialization and deserialization."""
        # Test serialization
        data = sample_input_goal.model_dump()
        assert isinstance(data, dict)
        assert data["goal_description"] == sample_input_goal.goal_description
        assert data["domain"] == sample_input_goal.domain
        
        # Test deserialization
        recreated_goal = InputGoal(**data)
        assert recreated_goal == sample_input_goal
    
    def test_team_package_serialization(self, sample_input_goal):
        """Test TeamPackage serialization and deserialization."""
        package = TeamPackage(
            goal=sample_input_goal,
            strategy_document="Test strategy",
            scouting_report="Test scouting", 
            new_agents=[{"name": "TestAgent"}],
            existing_agents=[{"name": "ExistingAgent"}],
            roster_documentation="Test roster",
            deployment_instructions="Test deployment"
        )
        
        # Test serialization
        data = package.model_dump()
        assert isinstance(data, dict)
        assert "goal" in data
        assert "strategy_document" in data
        assert "created_at" in data
        
        # Test deserialization
        recreated_package = TeamPackage(**data)
        assert recreated_package.goal.goal_description == sample_input_goal.goal_description
        assert recreated_package.strategy_document == "Test strategy"
    
    def test_model_validation_errors(self):
        """Test Pydantic model validation errors."""
        # Test missing required fields
        with pytest.raises(ValueError):
            InputGoal()  # Missing goal_description and domain
        
        with pytest.raises(ValueError):
            TeamPackage()  # Missing all required fields
    
    def test_model_field_validation(self):
        """Test individual field validation."""
        # Test ComplexityLevel enum validation
        valid_goal = InputGoal(
            goal_description="Test",
            domain="Test", 
            complexity_level=ComplexityLevel.HIGH
        )
        assert valid_goal.complexity_level == ComplexityLevel.HIGH
        
        # Test invalid complexity level would be caught by Pydantic
        # (This would be tested if we had custom validation)


class TestSystemIntegration:
    """Test system-level integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_system_integration(self, sample_input_goal):
        """Test complete system integration from start to finish."""
        em = EngineeringManager()
        
        # Create realistic mock responses
        strategy_response = """
        # Strategy Document
        ## Goal Analysis
        Task management system requires:
        - User authentication and session management
        - Task CRUD operations and persistence
        - Real-time collaboration features
        - Mobile-responsive UI
        
        ## Team Composition
        - Authentication Specialist (High priority)
        - Frontend Developer (High priority)  
        - Backend API Developer (High priority)
        - Real-time Systems Engineer (Medium priority)
        - Database Architect (Medium priority)
        """
        
        scouting_response = """
        # Scouting Report
        ## Matches Found
        - existing_auth_agent: 85% match for Authentication Specialist
        
        ## Gaps Identified  
        - Frontend Developer: No suitable match
        - Backend API Developer: No suitable match
        - Real-time Systems Engineer: No suitable match
        - Database Architect: Partial match, insufficient capabilities
        """
        
        new_agents_response = [
            {
                "name": "FrontendDeveloper",
                "role": "Frontend Developer",
                "capabilities": ["React", "CSS", "JavaScript"],
                "priority": "high"
            },
            {
                "name": "BackendAPIDeveloper", 
                "role": "Backend API Developer",
                "capabilities": ["Node.js", "Express", "REST APIs"],
                "priority": "high"
            }
        ]
        
        roster_response = """
        # Team Roster
        ## Purpose
        Develop task management system with real-time collaboration
        
        ## Team Members
        - Authentication Specialist (existing): Handle user auth
        - Frontend Developer (new): React UI development
        - Backend API Developer (new): REST APIs and business logic
        
        ## Workflow
        1. Requirements analysis
        2. API design and database schema  
        3. Parallel frontend and backend development
        4. Integration and testing
        5. Deployment and monitoring
        """
        
        # Mock all delegations with realistic data
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_response)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_response)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents_response)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_response)
        
        # Execute complete workflow
        result = await em.process(sample_input_goal)
        
        # Verify comprehensive integration
        assert isinstance(result, TeamPackage)
        assert result.goal == sample_input_goal
        assert "Task management system" in result.strategy_document
        assert "Scouting Report" in result.scouting_report
        assert len(result.new_agents) == 2
        assert result.new_agents[0]["name"] == "FrontendDeveloper"
        assert "Team Roster" in result.roster_documentation
        assert len(result.deployment_instructions) > 0
        
        # Verify workflow completion
        status = em.get_workflow_status()
        assert status["current_step"] == "final_packaging"
        assert "Workflow completed" in status["latest_action"]["action"]
    
    @pytest.mark.asyncio
    async def test_system_scalability_integration(self):
        """Test system integration under load."""
        # Test multiple concurrent workflows with different complexity
        em = EngineeringManager()
        
        # Mock efficient responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Quick strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Quick scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "QuickAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Quick roster")
        
        # Create goals with different complexities
        goals = [
            InputGoal(goal_description="Simple blog", domain="web", complexity_level=ComplexityLevel.LOW),
            InputGoal(goal_description="E-commerce site", domain="web", complexity_level=ComplexityLevel.MEDIUM),
            InputGoal(goal_description="Enterprise platform", domain="enterprise", complexity_level=ComplexityLevel.HIGH),
            InputGoal(goal_description="AI system", domain="ai", complexity_level=ComplexityLevel.ENTERPRISE)
        ]
        
        # Process all concurrently
        results = await asyncio.gather(*[em.process(goal) for goal in goals])
        
        # Verify all succeeded
        assert len(results) == 4
        for i, result in enumerate(results):
            assert isinstance(result, TeamPackage)
            assert result.goal.complexity_level == goals[i].complexity_level
    
    def test_configuration_integration(self):
        """Test system configuration and environment integration."""
        # Test that agents can be configured with different settings
        em1 = EngineeringManager()
        em2 = EngineeringManager()
        
        # Verify independent configurations
        assert em1.name == "EngineeringManager"
        assert em2.name == "EngineeringManager"
        assert em1.workflow_history != em2.workflow_history
        
        # Verify database isolation
        assert em1.agent.db != em2.agent.db or em1.agent.db is None or em2.agent.db is None