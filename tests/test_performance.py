"""
Performance and scalability tests for AgentForge system.

Tests system performance, memory usage, and scalability under various loads.
"""

import pytest
import asyncio
import time
import threading
import psutil
import os
from unittest.mock import AsyncMock
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel


class TestSystemPerformance:
    """Test system performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_single_workflow_performance(self, sample_input_goal):
        """Test performance of single workflow execution."""
        em = EngineeringManager()
        
        # Setup fast mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Fast strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Fast scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "FastAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Fast roster")
        
        # Measure execution time
        start_time = time.perf_counter()
        result = await em.process(sample_input_goal)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 1.0  # Should complete within 1 second with mocks
        assert isinstance(result, TeamPackage)
        
        print(f"Single workflow execution time: {execution_time:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self):
        """Test performance with concurrent workflow executions."""
        em = EngineeringManager()
        
        # Setup fast mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Concurrent strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Concurrent scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "ConcurrentAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Concurrent roster")
        
        # Create multiple goals
        goals = [
            InputGoal(goal_description=f"Concurrent goal {i}", domain=f"domain{i}")
            for i in range(10)
        ]
        
        # Measure concurrent execution time
        start_time = time.perf_counter()
        results = await asyncio.gather(*[em.process(goal) for goal in goals])
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        throughput = len(goals) / execution_time
        
        # Performance assertions
        assert len(results) == 10
        assert all(isinstance(result, TeamPackage) for result in results)
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert throughput > 5.0  # Should achieve at least 5 goals per second
        
        print(f"Concurrent workflow throughput: {throughput:.1f} goals/second")
        print(f"Total execution time: {execution_time:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_scaling_performance_by_complexity(self):
        """Test how performance scales with goal complexity."""
        em = EngineeringManager()
        
        # Setup mock responses proportional to complexity
        complexity_responses = {
            ComplexityLevel.LOW: ("Simple strategy", "Simple scouting", [{"name": "SimpleAgent"}], "Simple roster"),
            ComplexityLevel.MEDIUM: ("Medium strategy " * 5, "Medium scouting " * 5, [{"name": f"MedAgent{i}"} for i in range(3)], "Medium roster " * 5),
            ComplexityLevel.HIGH: ("High strategy " * 10, "High scouting " * 10, [{"name": f"HighAgent{i}"} for i in range(6)], "High roster " * 10),
            ComplexityLevel.ENTERPRISE: ("Enterprise strategy " * 20, "Enterprise scouting " * 20, [{"name": f"EntAgent{i}"} for i in range(12)], "Enterprise roster " * 20)
        }
        
        performance_results = {}
        
        for complexity in ComplexityLevel:
            strategy, scouting, agents, roster = complexity_responses[complexity]
            
            em._delegate_to_systems_analyst = AsyncMock(return_value=strategy)
            em._delegate_to_talent_scout = AsyncMock(return_value=scouting)
            em._delegate_to_agent_developer = AsyncMock(return_value=agents)
            em._delegate_to_integration_architect = AsyncMock(return_value=roster)
            
            goal = InputGoal(
                goal_description=f"Test {complexity.value} complexity goal",
                domain="test",
                complexity_level=complexity
            )
            
            # Measure execution time
            start_time = time.perf_counter()
            result = await em.process(goal)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            performance_results[complexity] = execution_time
            
            # Verify result
            assert isinstance(result, TeamPackage)
            assert len(result.new_agents) == len(agents)
        
        # Verify performance scales reasonably
        assert performance_results[ComplexityLevel.LOW] < 1.0
        assert performance_results[ComplexityLevel.MEDIUM] < 2.0
        assert performance_results[ComplexityLevel.HIGH] < 3.0
        assert performance_results[ComplexityLevel.ENTERPRISE] < 5.0
        
        # Performance should generally increase with complexity
        assert performance_results[ComplexityLevel.LOW] <= performance_results[ComplexityLevel.MEDIUM] * 2
        assert performance_results[ComplexityLevel.MEDIUM] <= performance_results[ComplexityLevel.HIGH] * 2
        
        for complexity, exec_time in performance_results.items():
            print(f"{complexity.value}: {exec_time:.3f} seconds")
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_execution(self, sample_input_goal):
        """Test memory usage during workflow execution."""
        em = EngineeringManager()
        
        # Setup mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Memory test strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Memory test scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "MemoryAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Memory test roster")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute workflow
        result = await em.process(sample_input_goal)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory assertions
        assert isinstance(result, TeamPackage)
        assert memory_increase < 50  # Should not increase memory by more than 50MB
        
        print(f"Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB (Δ{memory_increase:.1f} MB)")
    
    @pytest.mark.asyncio
    async def test_repeated_execution_memory_stability(self):
        """Test memory stability across repeated executions."""
        em = EngineeringManager()
        
        # Setup minimal mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Stable")
        em._delegate_to_talent_scout = AsyncMock(return_value="Stable")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Stable")
        
        process = psutil.Process(os.getpid())
        memory_readings = []
        
        # Execute multiple workflows
        for i in range(10):
            goal = InputGoal(goal_description=f"Stability test {i}", domain="test")
            result = await em.process(goal)
            
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(memory_usage)
            
            assert isinstance(result, TeamPackage)
        
        # Check memory stability
        max_memory = max(memory_readings)
        min_memory = min(memory_readings)
        memory_variation = max_memory - min_memory
        
        # Memory should remain relatively stable
        assert memory_variation < 20  # Less than 20MB variation
        
        print(f"Memory stability: {min_memory:.1f} - {max_memory:.1f} MB (variation: {memory_variation:.1f} MB)")


class TestScalabilityLimits:
    """Test system scalability limits and bottlenecks."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_limit(self):
        """Test system behavior at high concurrency levels."""
        em = EngineeringManager()
        
        # Setup very fast mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Fast")
        em._delegate_to_talent_scout = AsyncMock(return_value="Fast")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Fast")
        
        # Create many concurrent goals
        num_goals = 50  # High concurrency test
        goals = [
            InputGoal(goal_description=f"Scale test {i}", domain="test")
            for i in range(num_goals)
        ]
        
        # Execute with high concurrency
        start_time = time.perf_counter()
        results = await asyncio.gather(*[em.process(goal) for goal in goals])
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        throughput = len(goals) / execution_time
        
        # Scalability assertions
        assert len(results) == num_goals
        assert all(isinstance(result, TeamPackage) for result in results)
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert throughput > 3.0  # Should maintain reasonable throughput
        
        print(f"High concurrency test: {num_goals} goals in {execution_time:.2f}s ({throughput:.1f} goals/sec)")
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self):
        """Test system handling of large data payloads."""
        em = EngineeringManager()
        
        # Create large mock responses
        large_strategy = "Large strategy document. " * 1000  # ~25KB
        large_scouting = "Large scouting report. " * 1000   # ~25KB
        large_agents = [{"name": f"Agent{i}", "description": "Large description. " * 100} for i in range(20)]
        large_roster = "Large roster documentation. " * 1000  # ~30KB
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=large_strategy)
        em._delegate_to_talent_scout = AsyncMock(return_value=large_scouting)
        em._delegate_to_agent_developer = AsyncMock(return_value=large_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=large_roster)
        
        # Create goal with large constraints and criteria
        goal = InputGoal(
            goal_description="Large data handling test with extensive requirements and detailed specifications",
            domain="test",
            constraints=["Constraint " + str(i) for i in range(50)],  # 50 constraints
            success_criteria=["Success criterion " + str(i) for i in range(100)]  # 100 criteria
        )
        
        # Execute with large data
        start_time = time.perf_counter()
        result = await em.process(goal)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Large data handling assertions
        assert isinstance(result, TeamPackage)
        assert len(result.strategy_document) > 20000  # Should handle large documents
        assert len(result.new_agents) == 20
        assert len(result.goal.constraints) == 50
        assert len(result.goal.success_criteria) == 100
        assert execution_time < 10.0  # Should still complete reasonably fast
        
        print(f"Large data handling: {execution_time:.3f}s for {len(large_strategy) + len(large_scouting) + len(large_roster)}+ chars")
    
    @pytest.mark.asyncio
    async def test_workflow_history_scalability(self, sample_input_goal):
        """Test workflow history scalability with many operations."""
        em = EngineeringManager()
        
        # Setup mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="History test")
        em._delegate_to_talent_scout = AsyncMock(return_value="History test")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="History test")
        
        # Execute workflow multiple times to build history
        for i in range(20):
            goal = InputGoal(goal_description=f"History test {i}", domain="test")
            result = await em.process(goal)
            assert isinstance(result, TeamPackage)
        
        # Check history scaling
        assert len(em.workflow_history) > 0
        assert len(em.workflow_history) < 200  # Should not grow excessively
        
        # Recent operations should still be fast
        start_time = time.perf_counter()
        final_result = await em.process(sample_input_goal)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        assert execution_time < 2.0  # Should not be slowed by history
        assert isinstance(final_result, TeamPackage)
        
        print(f"Workflow history: {len(em.workflow_history)} entries, latest execution: {execution_time:.3f}s")


class TestStressTests:
    """Stress tests for system limits and breaking points."""
    
    @pytest.mark.asyncio
    async def test_rapid_fire_requests(self):
        """Test system under rapid-fire request load."""
        em = EngineeringManager()
        
        # Setup instant mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Instant")
        em._delegate_to_talent_scout = AsyncMock(return_value="Instant")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Instant")
        
        # Execute many requests as fast as possible
        num_requests = 100
        
        async def rapid_request(i):
            goal = InputGoal(goal_description=f"Rapid {i}", domain="test")
            return await em.process(goal)
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[rapid_request(i) for i in range(num_requests)])
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        throughput = num_requests / execution_time
        
        # Stress test assertions
        assert len(results) == num_requests
        assert all(isinstance(result, TeamPackage) for result in results)
        assert throughput > 20  # Should handle at least 20 requests per second
        
        print(f"Rapid-fire stress test: {num_requests} requests in {execution_time:.2f}s ({throughput:.1f} req/sec)")
    
    @pytest.mark.asyncio
    async def test_extreme_complexity_stress(self):
        """Test system with extremely complex goals."""
        em = EngineeringManager()
        
        # Create extremely complex mock responses
        extreme_strategy = "Extreme complexity analysis. " * 5000  # ~150KB
        extreme_scouting = "Extreme scouting analysis. " * 5000   # ~150KB
        extreme_agents = [
            {
                "name": f"ExtremeAgent{i}",
                "role": f"Specialized Role {i}",
                "capabilities": [f"Capability {j}" for j in range(50)],
                "description": "Extremely detailed agent description. " * 100
            }
            for i in range(100)
        ]
        extreme_roster = "Extreme roster documentation. " * 5000  # ~150KB
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=extreme_strategy)
        em._delegate_to_talent_scout = AsyncMock(return_value=extreme_scouting)
        em._delegate_to_agent_developer = AsyncMock(return_value=extreme_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=extreme_roster)
        
        # Create extremely complex goal
        extreme_goal = InputGoal(
            goal_description="Build an extremely complex enterprise system with AI, blockchain, IoT, AR/VR, quantum computing, edge computing, and multi-dimensional requirements across 20 different domains with 1000+ microservices and advanced ML capabilities",
            domain="enterprise, AI, blockchain, IoT, AR/VR, quantum, edge, cloud, mobile, web, data, security, compliance, analytics, automation, integration, performance, scalability, reliability, maintenance",
            complexity_level=ComplexityLevel.ENTERPRISE,
            timeline="5 years",
            constraints=[f"Extreme constraint {i}" for i in range(200)],
            success_criteria=[f"Extreme success criterion {i}" for i in range(500)]
        )
        
        # Execute extreme complexity test
        start_time = time.perf_counter()
        result = await em.process(extreme_goal)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Extreme complexity assertions
        assert isinstance(result, TeamPackage)
        assert len(result.new_agents) == 100
        assert len(result.goal.constraints) == 200
        assert len(result.goal.success_criteria) == 500
        assert execution_time < 60.0  # Should complete within 1 minute even for extreme complexity
        
        total_data_size = (
            len(result.strategy_document) + 
            len(result.scouting_report) + 
            len(result.roster_documentation)
        )
        
        print(f"Extreme complexity: {execution_time:.2f}s, {total_data_size} chars, {len(result.new_agents)} agents")
    
    @pytest.mark.asyncio
    async def test_memory_pressure_stress(self):
        """Test system behavior under memory pressure."""
        em = EngineeringManager()
        
        # Create memory-intensive mock responses
        memory_intensive_data = "Memory intensive data. " * 10000  # ~250KB per response
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=memory_intensive_data)
        em._delegate_to_talent_scout = AsyncMock(return_value=memory_intensive_data)
        em._delegate_to_agent_developer = AsyncMock(return_value=[
            {"name": f"MemAgent{i}", "data": "Large data. " * 1000} for i in range(50)
        ])
        em._delegate_to_integration_architect = AsyncMock(return_value=memory_intensive_data)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute multiple memory-intensive workflows
        results = []
        for i in range(10):
            goal = InputGoal(goal_description=f"Memory pressure test {i}", domain="test")
            result = await em.process(goal)
            results.append(result)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory pressure assertions
        assert len(results) == 10
        assert all(isinstance(result, TeamPackage) for result in results)
        assert memory_increase < 200  # Should not consume more than 200MB additional memory
        
        print(f"Memory pressure test: {memory_increase:.1f} MB increase with 10 intensive workflows")


class TestPerformanceRegression:
    """Performance regression tests to detect performance degradations."""
    
    @pytest.mark.asyncio
    async def test_baseline_performance_benchmark(self, sample_input_goal):
        """Establish baseline performance benchmark."""
        em = EngineeringManager()
        
        # Standard mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Benchmark strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Benchmark scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "BenchmarkAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Benchmark roster")
        
        # Run multiple iterations for statistical significance
        execution_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = await em.process(sample_input_goal)
            end_time = time.perf_counter()
            
            execution_times.append(end_time - start_time)
            assert isinstance(result, TeamPackage)
        
        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        
        # Performance regression assertions
        assert avg_time < 0.5  # Average should be under 500ms
        assert max_time < 1.0   # Maximum should be under 1s
        assert min_time > 0.001  # Minimum should be reasonable (not zero)
        
        performance_stats = {
            "avg_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "iterations": len(execution_times)
        }
        
        print(f"Baseline performance: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
        return performance_stats
    
    @pytest.mark.asyncio
    async def test_concurrency_performance_regression(self):
        """Test for concurrency performance regressions."""
        em = EngineeringManager()
        
        # Setup mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Concurrency test")
        em._delegate_to_talent_scout = AsyncMock(return_value="Concurrency test")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Concurrency test")
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        performance_data = {}
        
        for concurrency in concurrency_levels:
            goals = [
                InputGoal(goal_description=f"Concurrency {concurrency} - {i}", domain="test")
                for i in range(concurrency)
            ]
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*[em.process(goal) for goal in goals])
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            throughput = len(goals) / execution_time
            
            performance_data[concurrency] = {
                "execution_time": execution_time,
                "throughput": throughput
            }
            
            assert len(results) == concurrency
            assert all(isinstance(result, TeamPackage) for result in results)
        
        # Verify throughput scales reasonably with concurrency
        single_throughput = performance_data[1]["throughput"]
        concurrent_throughput = performance_data[10]["throughput"]
        
        # Throughput should improve with concurrency (not necessarily linear)
        assert concurrent_throughput >= single_throughput * 2  # At least 2x improvement
        
        for concurrency, perf in performance_data.items():
            print(f"Concurrency {concurrency}: {perf['throughput']:.1f} goals/sec")
        
        return performance_data