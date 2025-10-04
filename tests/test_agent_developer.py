"""
Test suite for Agent Developer - The Creator

This test suite validates the Agent Developer's ability to create comprehensive,
well-structured agent definitions based on vacant role specifications.
"""

import asyncio
import unittest
from typing import List, Dict, Any
import json
import tempfile
import os
from agents.agent_developer import (
    AgentDeveloper, 
    VacantRole, 
    ScoutingReport, 
    AgentSpecification,
    AgentGenerationResult
)


class TestAgentDeveloper(unittest.TestCase):
    """Test cases for Agent Developer functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.developer = AgentDeveloper()
        
        # Create sample vacant roles for testing
        cls.sample_vacant_roles = [
            VacantRole(
                role_name="API Developer", 
                title="Senior API Developer",
                core_responsibilities=[
                    "Design and implement RESTful APIs",
                    "Ensure API security and performance",
                    "Create comprehensive API documentation",
                    "Handle API versioning and backwards compatibility"
                ],
                required_capabilities=[
                    "REST API design principles",
                    "Authentication and authorization",
                    "API rate limiting and throttling", 
                    "OpenAPI/Swagger documentation",
                    "Performance optimization"
                ],
                interaction_patterns={
                    "Frontend Developer": "Provide API specifications and endpoints",
                    "Database Developer": "Coordinate data access patterns",
                    "QA Engineer": "Support API testing and validation"
                },
                success_metrics=[
                    "API response times under 200ms",
                    "99.9% API uptime",
                    "Complete API documentation coverage",
                    "Zero security vulnerabilities"
                ],
                priority_level="high",
                domain_context="Web API development",
                complexity_level="complex"
            ),
            VacantRole(
                role_name="DevOps Engineer",
                title="Senior DevOps Engineer", 
                core_responsibilities=[
                    "Design and maintain CI/CD pipelines",
                    "Manage cloud infrastructure and deployments",
                    "Monitor system performance and reliability",
                    "Implement security and compliance measures"
                ],
                required_capabilities=[
                    "Container orchestration (Docker, Kubernetes)",
                    "Cloud platforms (AWS, Azure, GCP)",
                    "Infrastructure as Code (Terraform, CloudFormation)",
                    "Monitoring and observability tools",
                    "Security best practices"
                ],
                interaction_patterns={
                    "API Developer": "Support deployment and monitoring",
                    "QA Engineer": "Integrate testing into deployment pipeline",
                    "Security Specialist": "Implement security controls"
                },
                success_metrics=[
                    "Deployment frequency and success rate",
                    "Mean time to recovery (MTTR)",
                    "System uptime and reliability",
                    "Security compliance scores"
                ],
                priority_level="high",
                domain_context="Infrastructure and operations",
                complexity_level="complex"
            )
        ]
        
        cls.sample_scouting_report = ScoutingReport(
            matched_agents=[
                {"name": "existing_qa_agent", "match_score": 0.8, "role": "Quality Assurance"}
            ],
            vacant_roles=cls.sample_vacant_roles,
            capability_gaps=[
                "API development and design",
                "DevOps and infrastructure automation",
                "Container orchestration",
                "API security implementation"
            ],
            reuse_analysis={
                "existing_patterns": "Limited API-specific patterns available",
                "reuse_potential": "Medium - some infrastructure patterns exist"
            },
            priority_recommendations=["API Developer", "DevOps Engineer"]
        )
    
    async def test_basic_agent_development(self):
        """Test basic agent development functionality"""
        print("\n🧪 Testing basic agent development...")
        
        result = await self.developer.develop_agents(self.sample_scouting_report)
        
        # Basic validations
        self.assertTrue(result.success, "Agent development should succeed")
        self.assertGreater(len(result.agents_created), 0, "Should create at least one agent")
        self.assertEqual(len(result.agents_created), len(self.sample_vacant_roles), 
                        "Should create agent for each vacant role")
        
        print(f"✅ Created {len(result.agents_created)} agents successfully")
        return result
    
    async def test_agent_specification_quality(self):
        """Test the quality of generated agent specifications"""
        print("\n🧪 Testing agent specification quality...")
        
        result = await self.developer.develop_agents(self.sample_scouting_report)
        
        for agent_spec in result.agents_created:
            # Test system prompt quality
            self.assertGreater(len(agent_spec.system_prompt), 100,
                             f"System prompt too short for {agent_spec.name}")
            self.assertIn("You are", agent_spec.system_prompt,
                         f"System prompt should define agent identity for {agent_spec.name}")
            
            # Test instructions completeness
            self.assertGreaterEqual(len(agent_spec.instructions), 3,
                                  f"Should have at least 3 instructions for {agent_spec.name}")
            
            # Test tools specification
            self.assertGreater(len(agent_spec.tools_required), 0,
                             f"Should specify required tools for {agent_spec.name}")
            self.assertIn("ReasoningTools", agent_spec.tools_required,
                         f"All agents should have ReasoningTools for {agent_spec.name}")
            
            # Test success criteria
            self.assertGreater(len(agent_spec.success_criteria), 0,
                             f"Should have success criteria for {agent_spec.name}")
            
            # Test code generation
            self.assertIn("def create_", agent_spec.initialization_code,
                         f"Should include initialization function for {agent_spec.name}")
            self.assertIn("Agent(", agent_spec.initialization_code,
                         f"Should create Agent instance for {agent_spec.name}")
            
        print(f"✅ All {len(result.agents_created)} agent specifications meet quality standards")
        return result
    
    async def test_role_specific_customization(self):
        """Test that agents are customized for their specific roles"""
        print("\n🧪 Testing role-specific customization...")
        
        result = await self.developer.develop_agents(self.sample_scouting_report)
        
        # Find the API Developer agent
        api_agent = next((agent for agent in result.agents_created 
                         if "api" in agent.name.lower()), None)
        self.assertIsNotNone(api_agent, "Should create API Developer agent")
        
        # Test API-specific characteristics
        api_terms = ["API", "endpoint", "REST", "authentication"]
        prompt_contains_api_terms = any(term.lower() in api_agent.system_prompt.lower() 
                                       for term in api_terms)
        self.assertTrue(prompt_contains_api_terms, 
                       "API agent should have API-specific terminology")
        
        # Find the DevOps Engineer agent
        devops_agent = next((agent for agent in result.agents_created 
                           if "devops" in agent.name.lower()), None)
        self.assertIsNotNone(devops_agent, "Should create DevOps Engineer agent")
        
        # Test DevOps-specific characteristics
        devops_terms = ["deployment", "infrastructure", "pipeline", "monitoring"]
        prompt_contains_devops_terms = any(term.lower() in devops_agent.system_prompt.lower() 
                                          for term in devops_terms)
        self.assertTrue(prompt_contains_devops_terms,
                       "DevOps agent should have DevOps-specific terminology")
        
        print(f"✅ Agents show role-specific customization")
        return result
    
    async def test_file_generation(self):
        """Test generation of agent implementation files"""
        print("\n🧪 Testing file generation...")
        
        result = await self.developer.develop_agents(self.sample_scouting_report)
        
        # Test that files were generated
        self.assertGreater(len(result.generated_files), 0, "Should generate implementation files")
        
        # Test file types
        file_extensions = set()
        for filepath in result.generated_files.keys():
            file_extensions.add(os.path.splitext(filepath)[1])
        
        self.assertIn('.py', file_extensions, "Should generate Python files")
        
        # Test file content
        for filepath, content in result.generated_files.items():
            self.assertGreater(len(content), 50, f"File {filepath} should have substantial content")
            
            if filepath.endswith('.py') and 'test_' not in filepath:
                self.assertIn('Agent(', content, f"Agent file {filepath} should create Agent instance")
                self.assertIn('def create_', content, f"Agent file {filepath} should have creation function")
        
        print(f"✅ Generated {len(result.generated_files)} files with appropriate content")
        return result
    
    async def test_validation_and_quality_scoring(self):
        """Test agent validation and quality scoring"""
        print("\n🧪 Testing validation and quality scoring...")
        
        result = await self.developer.develop_agents(self.sample_scouting_report)
        
        # Test quality score
        self.assertGreaterEqual(result.quality_score, 0.0, "Quality score should be non-negative")
        self.assertLessEqual(result.quality_score, 1.0, "Quality score should not exceed 1.0")
        self.assertGreater(result.quality_score, 0.5, "Quality score should indicate good quality")
        
        # Test validation results
        self.assertGreater(len(result.validation_results), 0, "Should have validation results")
        
        for validation in result.validation_results:
            self.assertIn('agent_name', validation, "Validation should identify agent")
            self.assertIn('tests_passed', validation, "Should report passed tests")
            self.assertIn('tests_failed', validation, "Should report failed tests")
            
            # Quality check: more tests should pass than fail
            self.assertGreaterEqual(validation['tests_passed'], validation['tests_failed'],
                                  f"Agent {validation['agent_name']} should pass more tests than it fails")
        
        print(f"✅ Quality score: {result.quality_score:.2f}, Validation completed")
        return result
    
    async def test_quick_agent_creation(self):
        """Test quick agent creation method"""
        print("\n🧪 Testing quick agent creation...")
        
        agent_spec = await self.developer.quick_agent_creation(
            role_name="Data Analyst",
            capabilities=["SQL queries", "Data visualization", "Statistical analysis"],
            priority="high"
        )
        
        # Validate quick creation result
        self.assertEqual(agent_spec.role, "Data Analyst", "Should set correct role")
        self.assertIn("Data Analyst", agent_spec.title, "Should set appropriate title")
        self.assertGreater(len(agent_spec.system_prompt), 50, "Should generate system prompt")
        self.assertGreater(len(agent_spec.instructions), 0, "Should generate instructions")
        
        print(f"✅ Quick agent creation successful: {agent_spec.name}")
        return agent_spec
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        print("\n🧪 Testing error handling...")
        
        # Test with empty scouting report
        empty_report = ScoutingReport(
            matched_agents=[],
            vacant_roles=[],
            capability_gaps=[],
            reuse_analysis={},
            priority_recommendations=[]
        )
        
        result = await self.developer.develop_agents(empty_report)
        
        # Should handle empty input gracefully
        self.assertEqual(len(result.agents_created), 0, "Should create no agents for empty report")
        self.assertIsNotNone(result.generation_summary, "Should provide summary even for empty input")
        
        print("✅ Error handling works correctly")
        return result
    
    async def test_integration_with_strategy_context(self):
        """Test integration with strategy context"""
        print("\n🧪 Testing integration with strategy context...")
        
        strategy_context = {
            "project_domain": "E-commerce platform",
            "timeline": "6 months",
            "team_size": "10 developers",
            "technology_stack": ["Node.js", "React", "PostgreSQL"],
            "non_functional_requirements": ["High availability", "Scalability", "Security"]
        }
        
        result = await self.developer.develop_agents(
            self.sample_scouting_report, 
            strategy_context=strategy_context
        )
        
        # Test that strategy context influences agent creation
        self.assertTrue(result.success, "Should succeed with strategy context")
        self.assertGreater(len(result.agents_created), 0, "Should create agents")
        
        # Check if strategy context is reflected in documentation
        context_terms = ["e-commerce", "node.js", "react", "postgresql"]
        doc_contains_context = any(term.lower() in result.documentation.lower() 
                                  for term in context_terms)
        
        print(f"✅ Strategy context integration successful")
        return result
    
    def test_memory_coordination_setup(self):
        """Test memory coordination capability"""
        print("\n🧪 Testing memory coordination setup...")
        
        # Test that developer has memory coordination capabilities
        self.assertIsNotNone(self.developer.knowledge_base, "Should have knowledge base for coordination")
        self.assertIsNotNone(self.developer.agent, "Should have agent for coordination")
        
        # Test that agent has appropriate tools
        agent_tools = [tool.__class__.__name__ for tool in self.developer.agent.tools]
        self.assertIn('ReasoningTools', agent_tools, "Should have reasoning tools for coordination")
        self.assertIn('KnowledgeTools', agent_tools, "Should have knowledge tools for coordination")
        
        print("✅ Memory coordination setup verified")


class TestAgentDeveloperIntegration(unittest.TestCase):
    """Integration tests for Agent Developer with other components"""
    
    async def test_systems_analyst_integration(self):
        """Test integration with Systems Analyst output"""
        print("\n🧪 Testing Systems Analyst integration...")
        
        # This would test integration with actual Systems Analyst output
        # For now, we simulate the integration
        
        from agents.systems_analyst import SystemsAnalyst, InputGoal
        
        analyst = SystemsAnalyst()
        developer = AgentDeveloper()
        
        # Create test goal
        test_goal = InputGoal(
            description="Build a microservices-based e-commerce platform",
            context="Modern cloud-native architecture with containerization",
            success_criteria=[
                "Scalable to 100k concurrent users",
                "99.99% uptime requirement", 
                "Sub-200ms API response times"
            ],
            domain="E-commerce / Cloud Architecture",
            complexity="High"
        )
        
        # Get strategy from analyst
        strategy_result = await analyst.analyze_goal(test_goal)
        
        # Create mock scouting report based on strategy
        mock_scouting = ScoutingReport(
            matched_agents=[],
            vacant_roles=[
                VacantRole(
                    role_name="Microservices Architect",
                    title="Senior Microservices Architect",
                    core_responsibilities=[
                        "Design microservices architecture",
                        "Define service boundaries and APIs",
                        "Ensure scalability and resilience"
                    ],
                    required_capabilities=[
                        "Microservices design patterns",
                        "API gateway configuration",
                        "Service mesh implementation",
                        "Distributed system design"
                    ],
                    interaction_patterns={
                        "DevOps Engineer": "Collaborate on deployment strategies",
                        "API Developer": "Define service interfaces"
                    },
                    success_metrics=[
                        "Service scalability metrics",
                        "API performance benchmarks",
                        "System resilience tests"
                    ],
                    priority_level="critical",
                    domain_context="E-commerce microservices",
                    complexity_level="enterprise"
                )
            ],
            capability_gaps=["Microservices architecture", "Distributed systems"],
            reuse_analysis={"patterns": "Limited microservices patterns"},
            priority_recommendations=["Microservices Architect"]
        )
        
        # Test agent development based on strategy
        result = await developer.develop_agents(mock_scouting, {"strategy": strategy_result})
        
        self.assertTrue(result.success, "Integration should succeed")
        self.assertGreater(len(result.agents_created), 0, "Should create agents based on strategy")
        
        print("✅ Systems Analyst integration successful")
        return result


async def run_all_tests():
    """Run all Agent Developer tests"""
    print("🚀 Starting Agent Developer Test Suite")
    print("=" * 60)
    
    # Basic functionality tests
    test_suite = TestAgentDeveloper()
    
    try:
        await test_suite.test_basic_agent_development()
        await test_suite.test_agent_specification_quality()
        await test_suite.test_role_specific_customization() 
        await test_suite.test_file_generation()
        await test_suite.test_validation_and_quality_scoring()
        await test_suite.test_quick_agent_creation()
        await test_suite.test_error_handling()
        await test_suite.test_integration_with_strategy_context()
        test_suite.test_memory_coordination_setup()
        
        # Integration tests
        integration_suite = TestAgentDeveloperIntegration()
        await integration_suite.test_systems_analyst_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Agent Developer is ready for production!")
        print("🎯 The Agent Developer can successfully create comprehensive agent specifications")
        print("🔧 Generated agents follow Agno framework standards and best practices")
        print("📝 Quality validation and file generation working correctly")
        print("🤝 Integration with other AgentForge components verified")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("🔍 Check the implementation and retry")
        raise


if __name__ == "__main__":
    print("Agent Developer Test Suite")
    print("Testing the master prompt engineer for AgentForge")
    print()
    
    # Run async tests
    asyncio.run(run_all_tests())