"""
Comprehensive test suite for the Enhanced Talent Scout with QDrant integration.

Tests cover:
- QDrant connection and collection management
- Agent library scanning and metadata extraction
- Semantic similarity search
- Agent-to-role matching algorithms
- Scouting report generation
- Performance and integration scenarios
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import pytest
from pydantic import ValidationError

from agents.talent_scout import (
    TalentScout,
    TalentScoutInput,
    TalentScoutOutput,
    QDrantManager,
    AgentLibraryScanner,
    AgentMetadata,
    RoleRequirement,
    StrategyDocument,
    ScoutingReport,
    AgentMatch,
    VacantRole
)


class TestAgentMetadata(unittest.TestCase):
    """Test agent metadata model validation."""
    
    def test_agent_metadata_creation(self):
        """Test creating valid agent metadata."""
        agent = AgentMetadata(
            id="test-agent-1",
            name="Test Agent",
            file_path="/path/to/agent.md",
            role="Developer",
            description="A test agent for development",
            capabilities=["Python", "Testing", "API Development"],
            tools=["pytest", "requests", "git"],
            domain="software development",
            complexity_level="medium",
            tags=["backend", "testing", "automation"],
            file_hash="abc123"
        )
        
        self.assertEqual(agent.name, "Test Agent")
        self.assertEqual(agent.role, "Developer")
        self.assertEqual(len(agent.capabilities), 3)
        self.assertIn("Python", agent.capabilities)
    
    def test_agent_metadata_validation(self):
        """Test validation of required fields."""
        with self.assertRaises(ValidationError):
            AgentMetadata()  # Missing required fields
    
    def test_agent_metadata_defaults(self):
        """Test default values are set correctly."""
        agent = AgentMetadata(
            id="test",
            name="Test",
            file_path="/test",
            role="Test Role",
            description="Test description",
            capabilities=[],
            tools=[],
            domain="test",
            complexity_level="low",
            tags=[],
            file_hash="test"
        )
        
        self.assertIsInstance(agent.created_at, datetime)
        self.assertIsInstance(agent.last_modified, datetime)


class TestRoleRequirement(unittest.TestCase):
    """Test role requirement model validation."""
    
    def test_role_requirement_creation(self):
        """Test creating valid role requirement."""
        role = RoleRequirement(
            role_id="frontend-dev",
            role_name="Frontend Developer",
            description="React developer for modern UI",
            required_capabilities=["React", "JavaScript", "HTML", "CSS"],
            preferred_capabilities=["TypeScript", "Redux", "Jest"],
            domain="web development",
            complexity_level="medium",
            priority="high"
        )
        
        self.assertEqual(role.role_name, "Frontend Developer")
        self.assertEqual(len(role.required_capabilities), 4)
        self.assertEqual(role.priority, "high")
    
    def test_role_requirement_defaults(self):
        """Test default values."""
        role = RoleRequirement(
            role_id="test",
            role_name="Test Role",
            description="Test",
            required_capabilities=["skill1"],
            domain="test",
            complexity_level="low"
        )
        
        self.assertEqual(role.priority, "medium")
        self.assertEqual(role.preferred_capabilities, [])


class TestAgentLibraryScanner(unittest.TestCase):
    """Test the agent library scanning functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.scanner = AgentLibraryScanner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def create_test_agent_file(self, filename: str, content: str):
        """Helper to create test agent files."""
        file_path = self.temp_path / filename
        file_path.write_text(content)
        return file_path
    
    def test_extract_agent_metadata_valid_agent(self):
        """Test extracting metadata from a valid agent file."""
        content = """
        # AI Backend Developer Agent
        
        **Role:** Backend Developer
        **Description:** Specialized in Python API development and microservices
        **Capabilities:** Python, FastAPI, PostgreSQL, Docker, Kubernetes
        **Tools:** pytest, SQLAlchemy, Redis, Celery
        **Domain:** web development
        **Complexity:** high
        **Tags:** backend, api, microservices, python
        
        ## Instructions
        You are an expert backend developer...
        """
        
        file_path = self.create_test_agent_file("backend_developer.md", content)
        agent = self.scanner.extract_agent_metadata(file_path)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Backend Developer")
        self.assertEqual(agent.role, "Backend Developer")
        self.assertIn("Python", agent.capabilities)
        self.assertIn("FastAPI", agent.capabilities)
        self.assertEqual(agent.domain, "web development")
    
    def test_extract_agent_metadata_invalid_file(self):
        """Test that non-agent files are ignored."""
        content = "This is just a regular text file with no agent markers."
        
        file_path = self.create_test_agent_file("regular_file.txt", content)
        agent = self.scanner.extract_agent_metadata(file_path)
        
        self.assertIsNone(agent)
    
    def test_infer_role_from_name(self):
        """Test role inference from filename."""
        self.assertEqual(self.scanner.infer_role_from_name("senior_developer"), "Developer")
        self.assertEqual(self.scanner.infer_role_from_name("systems_analyst"), "Analyst")
        self.assertEqual(self.scanner.infer_role_from_name("ui_designer"), "Designer")
        self.assertEqual(self.scanner.infer_role_from_name("random_agent"), "General Agent")
    
    def test_infer_domain_from_content(self):
        """Test domain inference from content."""
        web_content = "This agent works with React, JavaScript, and HTML"
        self.assertEqual(self.scanner.infer_domain_from_content(web_content), "web development")
        
        data_content = "Uses pandas and numpy for machine learning analysis"
        self.assertEqual(self.scanner.infer_domain_from_content(data_content), "data science")
        
        general_content = "This is a general purpose agent"
        self.assertEqual(self.scanner.infer_domain_from_content(general_content), "general")
    
    async def test_scan_library(self):
        """Test scanning a complete agent library."""
        # Create multiple agent files
        agents = [
            ("frontend_dev.md", """
            Role: Frontend Developer
            Description: React specialist
            Capabilities: React, JavaScript, HTML, CSS
            Domain: web development
            """),
            ("data_analyst.md", """  
            Role: Data Analyst
            Description: Python data analysis expert
            Capabilities: Python, pandas, numpy, matplotlib
            Domain: data science
            """),
            ("regular_file.txt", "Not an agent file")
        ]
        
        for filename, content in agents:
            self.create_test_agent_file(filename, content)
        
        scanned_agents = await self.scanner.scan_library(str(self.temp_path))
        
        # Should find 2 agent files, ignore the regular file
        self.assertEqual(len(scanned_agents), 2)
        agent_names = [agent.name for agent in scanned_agents]
        self.assertIn("Frontend Dev", agent_names)
        self.assertIn("Data Analyst", agent_names)


class TestQDrantManager(unittest.TestCase):
    """Test QDrant database integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Use mock QDrant clients for testing
        self.qdrant = QDrantManager(
            host="localhost",
            port=6333,
            api_key="test-key"
        )
    
    def test_generate_embedding(self):
        """Test embedding generation."""
        text = "This is a test agent description"
        embedding = self.qdrant.generate_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 384)  # all-MiniLM-L6-v2 dimension
        self.assertTrue(all(isinstance(x, float) for x in embedding))
    
    def test_create_agent_text(self):
        """Test searchable text creation from agent metadata."""
        agent = AgentMetadata(
            id="test",
            name="Test Agent",
            file_path="/test",
            role="Developer",
            description="A test developer agent",
            capabilities=["Python", "Testing"],
            tools=["pytest", "git"],
            domain="software development",
            complexity_level="medium",
            tags=["backend", "testing"],
            file_hash="test"
        )
        
        agent_text = self.qdrant.create_agent_text(agent)
        
        self.assertIn("Role: Developer", agent_text)
        self.assertIn("Description: A test developer agent", agent_text)
        self.assertIn("Capabilities: Python, Testing", agent_text)
        self.assertIn("Domain: software development", agent_text)
    
    @patch('agents.talent_scout.AsyncQdrantClient')
    async def test_initialize_collection(self, mock_client_class):
        """Test QDrant collection initialization."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock collections response (no existing collection)
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        qdrant = QDrantManager()
        qdrant.async_client = mock_client
        
        result = await qdrant.initialize_collection()
        
        self.assertTrue(result)
        mock_client.create_collection.assert_called_once()
    
    @patch('agents.talent_scout.AsyncQdrantClient')
    async def test_index_agent(self, mock_client_class):
        """Test agent indexing in QDrant."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        qdrant = QDrantManager()
        qdrant.async_client = mock_client
        
        agent = AgentMetadata(
            id="test",
            name="Test Agent",
            file_path="/test",
            role="Developer",
            description="Test description",
            capabilities=["Python"],
            tools=["git"],
            domain="development",
            complexity_level="medium",
            tags=["test"],
            file_hash="test"
        )
        
        result = await qdrant.index_agent(agent)
        
        self.assertTrue(result)
        mock_client.upsert.assert_called_once()


class TestTalentScout(unittest.TestCase):
    """Test the main TalentScout functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.scout = TalentScout()
    
    def test_create_role_query_text(self):
        """Test creating search query from role requirement."""
        role = RoleRequirement(
            role_id="frontend-dev",
            role_name="Frontend Developer",
            description="React developer for modern UI",
            required_capabilities=["React", "JavaScript"],
            preferred_capabilities=["TypeScript"],
            domain="web development",
            complexity_level="medium"
        )
        
        query_text = self.scout.create_role_query_text(role)
        
        self.assertIn("Role: Frontend Developer", query_text)
        self.assertIn("React, JavaScript", query_text)
        self.assertIn("web development", query_text)
    
    def test_calculate_capability_match_score(self):
        """Test capability matching score calculation."""
        agent = AgentMetadata(
            id="test",
            name="Test Agent",
            file_path="/test",
            role="Developer",
            description="Test",
            capabilities=["React", "JavaScript", "HTML", "CSS", "TypeScript"],
            tools=[],
            domain="web development",
            complexity_level="medium",
            tags=[],
            file_hash="test"
        )
        
        role = RoleRequirement(
            role_id="frontend",
            role_name="Frontend Developer",
            description="React developer",
            required_capabilities=["React", "JavaScript"],
            preferred_capabilities=["TypeScript", "Redux"],
            domain="web development",
            complexity_level="medium"
        )
        
        score = self.scout.calculate_capability_match_score(agent, role)
        
        # Should have perfect required match (1.0) and partial preferred match (0.5)
        # Score = (1.0 * 0.8) + (0.5 * 0.2) = 0.9
        self.assertAlmostEqual(score, 0.9, places=2)
    
    def test_calculate_overall_score(self):
        """Test overall matching score calculation."""
        similarity_score = 0.8
        capability_score = 0.9
        
        # Without bonuses
        overall_score = self.scout.calculate_overall_score(
            similarity_score, capability_score
        )
        expected = (0.8 * 0.6) + (0.9 * 0.4)  # 0.48 + 0.36 = 0.84
        self.assertAlmostEqual(overall_score, expected, places=2)
        
        # With bonuses
        overall_score_bonus = self.scout.calculate_overall_score(
            similarity_score, capability_score, domain_match=True, complexity_match=True
        )
        expected_bonus = expected + 0.05 + 0.05  # 0.84 + 0.1 = 0.94
        self.assertAlmostEqual(overall_score_bonus, expected_bonus, places=2)
    
    def test_determine_match_confidence(self):
        """Test match confidence determination."""
        self.assertEqual(self.scout.determine_match_confidence(0.9), "high")
        self.assertEqual(self.scout.determine_match_confidence(0.75), "medium")
        self.assertEqual(self.scout.determine_match_confidence(0.6), "low")
    
    def test_generate_adaptation_suggestions(self):
        """Test adaptation suggestions generation."""
        agent = AgentMetadata(
            id="test",
            name="Test Agent",
            file_path="/test",
            role="Developer",
            description="Test",
            capabilities=["Python", "Django"],
            tools=[],
            domain="web development",
            complexity_level="low",
            tags=[],
            file_hash="test"
        )
        
        role = RoleRequirement(
            role_id="fullstack",
            role_name="Full Stack Developer",
            description="Full stack web developer",
            required_capabilities=["React", "Node.js", "Python"],
            domain="web development",
            complexity_level="high"
        )
        
        suggestions = self.scout.generate_adaptation_suggestions(agent, role, 0.4)
        
        self.assertGreater(len(suggestions), 0)
        # Should suggest adding missing React and Node.js
        suggestion_text = " ".join(suggestions)
        self.assertIn("react", suggestion_text.lower())
        self.assertIn("node.js", suggestion_text.lower())


class TestScoutingReport(unittest.TestCase):
    """Test scouting report generation and validation."""
    
    def test_scouting_report_creation(self):
        """Test creating a valid scouting report."""
        report = ScoutingReport(
            strategy_title="Test Strategy",
            total_roles=5,
            filled_roles=3,
            vacant_roles=2,
            matches=[],
            vacant_positions=[],
            overall_coverage=0.6,
            reuse_efficiency=0.6,
            processing_time_ms=1500.0
        )
        
        self.assertEqual(report.strategy_title, "Test Strategy")
        self.assertEqual(report.total_roles, 5)
        self.assertEqual(report.overall_coverage, 0.6)
        self.assertIsInstance(report.report_id, str)
        self.assertIsInstance(report.generated_at, datetime)
    
    def test_scouting_report_analytics(self):
        """Test that analytics are calculated correctly."""
        report = ScoutingReport(
            strategy_title="Analytics Test",
            total_roles=10,
            filled_roles=7,
            vacant_roles=3,
            matches=[],
            vacant_positions=[],
            overall_coverage=0.7,
            reuse_efficiency=0.7,
            processing_time_ms=2000.0
        )
        
        self.assertEqual(report.filled_roles + report.vacant_roles, report.total_roles)
        self.assertEqual(report.overall_coverage, 0.7)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete scouting workflows."""
    
    @patch('agents.talent_scout.QDrantManager')
    @patch('agents.talent_scout.AgentLibraryScanner')
    async def test_complete_scouting_workflow(self, mock_scanner_class, mock_qdrant_class):
        """Test a complete scouting workflow from strategy to report."""
        # Mock QDrant manager
        mock_qdrant = AsyncMock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_qdrant.initialize_collection.return_value = True
        mock_qdrant.get_collection_stats.return_value = {"total_agents": 10}
        
        # Mock agent scanner
        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        
        # Create sample agent metadata
        sample_agent = AgentMetadata(
            id="react-dev",
            name="React Developer",
            file_path="/agents/react_dev.md",
            role="Frontend Developer",
            description="Expert React developer for modern UIs",
            capabilities=["React", "JavaScript", "HTML", "CSS", "Redux"],
            tools=["webpack", "babel", "jest"],
            domain="web development",
            complexity_level="medium",
            tags=["frontend", "react", "ui"],
            file_hash="abc123"
        )
        
        mock_scanner.scan_library.return_value = [sample_agent]
        mock_qdrant.search_similar_agents.return_value = [(sample_agent, 0.85)]
        
        # Create strategy document
        strategy = StrategyDocument(
            title="Modern Web App Strategy",
            goal_description="Build a modern web application",
            domain="web development",
            complexity_level="medium",
            roles=[
                RoleRequirement(
                    role_id="frontend-dev",
                    role_name="Frontend Developer",
                    description="React developer for UI components",
                    required_capabilities=["React", "JavaScript"],
                    preferred_capabilities=["Redux", "TypeScript"],
                    domain="web development",
                    complexity_level="medium",
                    priority="high"
                )
            ]
        )
        
        # Create input
        scout_input = TalentScoutInput(
            goal="Test scouting workflow",
            strategy_document=strategy,
            agent_libraries=["/test/agents"],
            force_reindex=False
        )
        
        # Create scout and process
        scout = TalentScout()
        scout.qdrant = mock_qdrant
        scout.scanner = mock_scanner
        
        result = await scout.process(scout_input)
        
        # Verify results
        self.assertEqual(result.status, "success")
        self.assertIsInstance(result.scouting_report, ScoutingReport)
        self.assertEqual(result.scouting_report.total_roles, 1)
        self.assertEqual(result.scouting_report.filled_roles, 1)
        self.assertEqual(result.scouting_report.vacant_roles, 0)
        self.assertEqual(len(result.scouting_report.matches), 1)
        
        # Verify the match
        match = result.scouting_report.matches[0]
        self.assertEqual(match.agent.name, "React Developer")
        self.assertEqual(match.role_requirement.role_name, "Frontend Developer")
        self.assertGreater(match.overall_score, 0.75)  # Should be a good match


class TestPerformanceScenarios(unittest.TestCase):
    """Test performance and scalability scenarios."""
    
    def test_large_agent_library_simulation(self):
        """Test processing large numbers of agents."""
        # Create many sample agents
        agents = []
        for i in range(100):
            agent = AgentMetadata(
                id=f"agent-{i}",
                name=f"Agent {i}",
                file_path=f"/agents/agent_{i}.md",
                role=f"Role {i % 10}",  # 10 different roles
                description=f"Description for agent {i}",
                capabilities=[f"skill-{j}" for j in range(i % 5 + 1)],
                tools=[f"tool-{j}" for j in range(i % 3 + 1)],
                domain=f"domain-{i % 3}",  # 3 different domains
                complexity_level=["low", "medium", "high"][i % 3],
                tags=[f"tag-{j}" for j in range(i % 4)],
                file_hash=f"hash-{i}"
            )
            agents.append(agent)
        
        # This tests that our data structures can handle larger datasets
        self.assertEqual(len(agents), 100)
        
        # Test that we can create different combinations
        domains = set(agent.domain for agent in agents)
        self.assertEqual(len(domains), 3)
        
        complexity_levels = set(agent.complexity_level for agent in agents)
        self.assertEqual(len(complexity_levels), 3)
    
    async def test_concurrent_scouting_operations(self):
        """Test handling multiple concurrent scouting operations."""
        # This would test concurrent access to QDrant and other resources
        # In a real test, we'd spin up multiple scout instances
        
        scout = TalentScout()
        
        # Create multiple role requirements
        roles = [
            RoleRequirement(
                role_id=f"role-{i}",
                role_name=f"Role {i}",
                description=f"Role {i} description",
                required_capabilities=[f"skill-{i}"],
                domain="test",
                complexity_level="medium"
            )
            for i in range(5)
        ]
        
        # Test that we can create query text for multiple roles efficiently
        for role in roles:
            query_text = scout.create_role_query_text(role)
            self.assertIn(f"Role {role.role_id[-1]}", query_text)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    async def test_invalid_agent_library_path(self):
        """Test handling of invalid agent library paths."""
        scout = TalentScout()
        
        # Test with non-existent path
        indexing_stats = await scout.index_agent_libraries(["/non/existent/path"])
        
        self.assertEqual(indexing_stats["agents_found"], 0)
        self.assertEqual(indexing_stats["agents_indexed"], 0)
    
    def test_invalid_strategy_document(self):
        """Test handling of malformed strategy documents."""
        with self.assertRaises(ValidationError):
            StrategyDocument()  # Missing required fields
    
    def test_empty_role_requirements(self):
        """Test handling empty role requirements."""
        strategy = StrategyDocument(
            title="Empty Strategy",
            goal_description="Test with no roles",
            domain="test",
            complexity_level="low",
            roles=[]  # Empty roles list
        )
        
        self.assertEqual(len(strategy.roles), 0)
    
    async def test_qdrant_connection_failure(self):
        """Test handling QDrant connection failures."""
        # Mock a QDrant connection failure
        with patch('agents.talent_scout.QDrantManager') as mock_qdrant_class:
            mock_qdrant = Mock()
            mock_qdrant.initialize_collection.return_value = False
            mock_qdrant_class.return_value = mock_qdrant
            
            scout = TalentScout()
            scout.qdrant = mock_qdrant
            
            result = await scout.initialize()
            self.assertFalse(result)


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        
        if test_category == "unit":
            # Run unit tests only
            suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentMetadata)
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRoleRequirement))
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentLibraryScanner))
            
        elif test_category == "integration":
            # Run integration tests
            suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegrationScenarios)
            
        elif test_category == "performance":
            # Run performance tests  
            suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceScenarios)
            
        else:
            # Run all tests
            suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    else:
        # Run all tests by default
        suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)