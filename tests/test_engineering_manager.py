"""
Unit tests for Engineering Manager (Orchestrator).

Tests the central nervous system of AgentForge that manages workflow 
from goal intake to final deliverable.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from agents.engineering_manager import (
    EngineeringManager, 
    InputGoal, 
    ComplexityLevel, 
    WorkflowStep,
    TeamPackage
)


class TestEngineeringManagerInputValidation:
    """Test input validation and data models."""
    
    def test_input_goal_creation_valid(self, sample_input_goal):
        """Test valid InputGoal creation."""
        assert sample_input_goal.goal_description == "Create a comprehensive task management system with real-time collaboration"
        assert sample_input_goal.domain == "web development"
        assert sample_input_goal.complexity_level == ComplexityLevel.HIGH
        assert sample_input_goal.timeline == "3 months"
        assert len(sample_input_goal.constraints) == 3
        assert len(sample_input_goal.success_criteria) == 5
    
    def test_input_goal_defaults(self):
        """Test InputGoal with minimal required fields."""
        goal = InputGoal(
            goal_description="Simple goal",
            domain="test domain"
        )
        assert goal.complexity_level == ComplexityLevel.MEDIUM
        assert goal.timeline is None
        assert goal.constraints == []
        assert goal.success_criteria == []
        assert goal.existing_resources == {}
    
    def test_complexity_levels(self):
        """Test all complexity level options."""
        levels = [ComplexityLevel.LOW, ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.ENTERPRISE]
        for level in levels:
            goal = InputGoal(goal_description="test", domain="test", complexity_level=level)
            assert goal.complexity_level == level
    
    def test_workflow_steps_enum(self):
        """Test workflow step enumeration."""
        steps = [
            WorkflowStep.GOAL_INTAKE,
            WorkflowStep.STRATEGY_ANALYSIS,
            WorkflowStep.RESOURCE_SCOUTING,
            WorkflowStep.AGENT_DEVELOPMENT,
            WorkflowStep.TEAM_INTEGRATION,
            WorkflowStep.FINAL_PACKAGING
        ]
        for step in steps:
            assert isinstance(step.value, str)


class TestEngineeringManagerInitialization:
    """Test Engineering Manager initialization and setup."""
    
    def test_initialization(self):
        """Test proper initialization of Engineering Manager."""
        em = EngineeringManager()
        
        assert em.name == "EngineeringManager"
        assert "orchestrator" in em.description.lower()
        assert em.workflow_history == []
        assert em.current_step is None
        assert em.agent is not None
    
    def test_workflow_status_initial(self):
        """Test initial workflow status."""
        em = EngineeringManager()
        status = em.get_workflow_status()
        
        assert status["current_step"] is None
        assert status["history_length"] == 0
        assert status["latest_action"] is None


class TestEngineeringManagerWorkflowTracking:
    """Test workflow tracking and logging functionality."""
    
    @pytest.mark.asyncio
    async def test_log_step_functionality(self):
        """Test workflow step logging."""
        em = EngineeringManager()
        em.current_step = WorkflowStep.GOAL_INTAKE
        
        await em._log_step("Test action", {"key": "value"})
        
        assert len(em.workflow_history) == 1
        log_entry = em.workflow_history[0]
        assert log_entry["step"] == WorkflowStep.GOAL_INTAKE.value
        assert log_entry["action"] == "Test action"
        assert log_entry["details"]["key"] == "value"
        assert "timestamp" in log_entry
    
    def test_workflow_status_with_history(self):
        """Test workflow status with history."""
        em = EngineeringManager()
        em.current_step = WorkflowStep.STRATEGY_ANALYSIS
        em.workflow_history = [
            {"timestamp": "2024-01-01", "step": "goal_intake", "action": "test", "details": {}}
        ]
        
        status = em.get_workflow_status()
        assert status["current_step"] == WorkflowStep.STRATEGY_ANALYSIS.value
        assert status["history_length"] == 1
        assert status["latest_action"]["action"] == "test"


class TestEngineeringManagerDelegation:
    """Test delegation methods to other agents."""
    
    @pytest.mark.asyncio
    async def test_delegate_to_systems_analyst(self, sample_input_goal):
        """Test delegation to Systems Analyst."""
        em = EngineeringManager()
        
        # Mock the run_with_mcp method
        em.run_with_mcp = AsyncMock(return_value="Mock strategy document")
        
        result = await em._delegate_to_systems_analyst(sample_input_goal)
        
        assert result == "Mock strategy document"
        em.run_with_mcp.assert_called_once()
        
        # Check that the call includes goal information
        call_args = em.run_with_mcp.call_args[0][0]
        assert sample_input_goal.goal_description in call_args
        assert sample_input_goal.domain in call_args
    
    @pytest.mark.asyncio
    async def test_delegate_to_talent_scout(self):
        """Test delegation to Talent Scout."""
        em = EngineeringManager()
        em.run_with_mcp = AsyncMock(return_value="Mock scouting report")
        
        strategy_doc = "Test strategy document"
        result = await em._delegate_to_talent_scout(strategy_doc)
        
        assert result == "Mock scouting report"
        em.run_with_mcp.assert_called_once()
        
        # Check that strategy document is included in the call
        call_args = em.run_with_mcp.call_args[0][0]
        assert strategy_doc in call_args
        assert "/home/delorenj/code/DeLoDocs/AI/Agents" in call_args
    
    @pytest.mark.asyncio
    async def test_delegate_to_agent_developer(self):
        """Test delegation to Agent Developer."""
        em = EngineeringManager()
        em.run_with_mcp = AsyncMock(return_value="Mock new agents response")
        
        scouting_report = "Test scouting report with gaps"
        result = await em._delegate_to_agent_developer(scouting_report)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["name"] == "PlaceholderAgent"
        assert result[0]["type"] == "general"
        
        em.run_with_mcp.assert_called_once()
        call_args = em.run_with_mcp.call_args[0][0]
        assert scouting_report in call_args
    
    @pytest.mark.asyncio
    async def test_delegate_to_integration_architect(self):
        """Test delegation to Integration Architect."""
        em = EngineeringManager()
        em.run_with_mcp = AsyncMock(return_value="Mock roster documentation")
        
        strategy = "Test strategy"
        scouting = "Test scouting"
        new_agents = [{"name": "TestAgent", "type": "test"}]
        
        result = await em._delegate_to_integration_architect(strategy, scouting, new_agents)
        
        assert result == "Mock roster documentation"
        em.run_with_mcp.assert_called_once()
        
        call_args = em.run_with_mcp.call_args[0][0]
        assert strategy in call_args
        assert scouting in call_args
        assert str(len(new_agents)) in call_args


class TestEngineeringManagerTeamPackaging:
    """Test final team package creation."""
    
    @pytest.mark.asyncio
    async def test_create_final_package(self, sample_input_goal):
        """Test creation of final team package."""
        em = EngineeringManager()
        
        strategy = "Test strategy document"
        scouting = "Test scouting report"
        new_agents = [{"name": "TestAgent", "prompt": "Test prompt", "type": "test"}]
        roster = "Test roster documentation"
        
        package = await em._create_final_package(
            sample_input_goal, strategy, scouting, new_agents, roster
        )
        
        assert isinstance(package, TeamPackage)
        assert package.goal == sample_input_goal
        assert package.strategy_document == strategy
        assert package.scouting_report == scouting
        assert package.new_agents == new_agents
        assert package.roster_documentation == roster
        assert len(package.deployment_instructions) > 0
        assert isinstance(package.created_at, datetime)
        
        # Check deployment instructions content
        instructions = package.deployment_instructions
        assert sample_input_goal.goal_description in instructions
        assert str(len(new_agents)) in instructions
    
    def test_team_package_model_validation(self, sample_input_goal):
        """Test TeamPackage model validation."""
        # Test valid package creation
        package = TeamPackage(
            goal=sample_input_goal,
            strategy_document="test strategy",
            scouting_report="test scouting",
            new_agents=[{"name": "test"}],
            existing_agents=[{"name": "existing"}],
            roster_documentation="test roster",
            deployment_instructions="test instructions"
        )
        
        assert package.goal == sample_input_goal
        assert len(package.new_agents) == 1
        assert len(package.existing_agents) == 1


class TestEngineeringManagerCompleteWorkflow:
    """Test complete workflow orchestration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, sample_input_goal):
        """Test successful complete workflow execution."""
        em = EngineeringManager()
        
        # Mock all delegation methods
        em._delegate_to_systems_analyst = AsyncMock(return_value="Mock strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Mock scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "MockAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Mock roster")
        
        result = await em.process(sample_input_goal)
        
        # Verify result
        assert isinstance(result, TeamPackage)
        assert result.goal == sample_input_goal
        assert result.strategy_document == "Mock strategy"
        assert result.scouting_report == "Mock scouting"
        assert len(result.new_agents) == 1
        
        # Verify all delegations were called
        em._delegate_to_systems_analyst.assert_called_once_with(sample_input_goal)
        em._delegate_to_talent_scout.assert_called_once_with("Mock strategy")
        em._delegate_to_agent_developer.assert_called_once_with("Mock scouting")
        em._delegate_to_integration_architect.assert_called_once()
        
        # Verify workflow tracking
        assert len(em.workflow_history) > 0
        assert em.current_step == WorkflowStep.FINAL_PACKAGING
        
        # Verify workflow steps were executed in order
        steps_executed = [entry["step"] for entry in em.workflow_history if entry["action"].startswith("Delegating") or entry["action"] == "Received input goal"]
        expected_workflow = [
            WorkflowStep.GOAL_INTAKE.value,
            WorkflowStep.STRATEGY_ANALYSIS.value, 
            WorkflowStep.RESOURCE_SCOUTING.value,
            WorkflowStep.AGENT_DEVELOPMENT.value,
            WorkflowStep.TEAM_INTEGRATION.value
        ]
        
        for expected_step in expected_workflow:
            assert any(expected_step in step for step in steps_executed)
    
    @pytest.mark.asyncio
    async def test_workflow_with_different_complexity_levels(self):
        """Test workflow with different complexity levels."""
        em = EngineeringManager()
        
        # Mock delegations
        em._delegate_to_systems_analyst = AsyncMock(return_value="Strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Scouting")  
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Roster")
        
        complexity_levels = [ComplexityLevel.LOW, ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.ENTERPRISE]
        
        for complexity in complexity_levels:
            goal = InputGoal(
                goal_description=f"Test goal with {complexity.value} complexity",
                domain="test",
                complexity_level=complexity
            )
            
            result = await em.process(goal)
            assert isinstance(result, TeamPackage)
            assert result.goal.complexity_level == complexity


class TestEngineeringManagerErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_delegation_error_handling(self, sample_input_goal):
        """Test handling of delegation errors."""
        em = EngineeringManager()
        
        # Make systems analyst delegation fail
        em._delegate_to_systems_analyst = AsyncMock(side_effect=Exception("Systems Analyst failed"))
        
        with pytest.raises(Exception, match="Systems Analyst failed"):
            await em.process(sample_input_goal)
        
        # Verify workflow tracking still works
        assert len(em.workflow_history) > 0
        assert em.current_step == WorkflowStep.STRATEGY_ANALYSIS
    
    @pytest.mark.asyncio  
    async def test_empty_goal_handling(self):
        """Test handling of minimal/empty goals."""
        em = EngineeringManager()
        
        # Mock delegations to return empty results
        em._delegate_to_systems_analyst = AsyncMock(return_value="")
        em._delegate_to_talent_scout = AsyncMock(return_value="")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="")
        
        minimal_goal = InputGoal(goal_description="", domain="")
        
        result = await em.process(minimal_goal)
        
        assert isinstance(result, TeamPackage)
        assert result.goal == minimal_goal
        assert len(result.new_agents) == 0


class TestEngineeringManagerIntegration:
    """Integration tests with actual agent components."""
    
    @pytest.mark.asyncio
    async def test_with_real_base_agent(self, sample_input_goal):
        """Test integration with actual base agent functionality."""
        em = EngineeringManager()
        
        # This tests that the base agent methods work correctly
        # The run_with_mcp method should handle both MCP and non-MCP scenarios
        
        # Test message handling
        test_message = "Test orchestration message"
        
        # Mock the agent response to avoid actual LLM calls
        em.agent.aprint_response = AsyncMock(return_value="Mock response")
        
        response = await em.run_with_mcp(test_message)
        assert response == "Mock response"
        em.agent.aprint_response.assert_called_once()


# Performance and stress tests
class TestEngineeringManagerPerformance:
    """Performance tests for the Engineering Manager."""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_processing(self):
        """Test processing multiple workflows concurrently."""
        em = EngineeringManager()
        
        # Mock fast delegations
        em._delegate_to_systems_analyst = AsyncMock(return_value="Fast strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Fast scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "FastAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Fast roster")
        
        # Create multiple goals
        goals = [
            InputGoal(goal_description=f"Concurrent goal {i}", domain=f"domain{i}")
            for i in range(5)
        ]
        
        # Process all goals concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[em.process(goal) for goal in goals])
        end_time = asyncio.get_event_loop().time()
        
        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert isinstance(result, TeamPackage)
        
        # Verify reasonable performance (should be much faster than sequential)
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_workflow_history_performance(self, sample_input_goal):
        """Test workflow history doesn't cause performance issues."""
        em = EngineeringManager()
        
        # Mock delegations
        em._delegate_to_systems_analyst = AsyncMock(return_value="Strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Roster")
        
        # Process goal and verify history is reasonable
        result = await em.process(sample_input_goal)
        
        assert isinstance(result, TeamPackage)
        assert len(em.workflow_history) < 20  # Should not have excessive logging
        assert em.workflow_history[-1]["action"] == "Workflow completed"