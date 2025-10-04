#!/usr/bin/env python3
"""
Core Enhanced Talent Scout Verification Script

Tests the core functionality without Agno framework dependencies.
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

def test_basic_functionality():
    """Test basic functionality that doesn't require external dependencies."""
    print("🚀 Testing Core Enhanced Talent Scout Functionality")
    print("=" * 60)
    
    try:
        # Test 1: Basic imports and data models
        print("📦 Testing imports and data models...")
        
        from agents.talent_scout import (
            AgentMetadata,
            RoleRequirement,
            ScoutingReport,
            StrategyDocument,
            QDrantManager,
            AgentLibraryScanner
        )
        print("✅ Core imports successful")
        
        # Test 2: Data model creation
        print("📋 Testing data model creation...")
        
        agent = AgentMetadata(
            id="test-agent",
            name="Test Agent",
            file_path="/test/path",
            role="Test Role",
            description="Test description",
            capabilities=["skill1", "skill2"],
            tools=["tool1"],
            domain="test",
            complexity_level="medium",
            tags=["tag1"],
            file_hash="hash123"
        )
        
        role = RoleRequirement(
            role_id="test-role",
            role_name="Test Role",
            description="Test role description",
            required_capabilities=["skill1", "skill2"],
            domain="test",
            complexity_level="medium"
        )
        
        strategy = StrategyDocument(
            title="Test Strategy",
            goal_description="Test goal",
            domain="test",
            complexity_level="medium",
            roles=[role]
        )
        
        assert agent.name == "Test Agent"
        assert role.role_name == "Test Role"
        assert strategy.title == "Test Strategy"
        print("✅ Data models working correctly")
        
        # Test 3: QDrant Manager basic functionality
        print("🔍 Testing QDrant Manager...")
        
        qdrant = QDrantManager()
        assert qdrant.host == "localhost"
        assert qdrant.port == 6333
        assert qdrant.embedding_dim == 384
        
        # Test embedding generation
        embedding = qdrant.generate_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        print("✅ QDrant Manager basic functionality working")
        
        # Test 4: Agent Library Scanner basic functionality
        print("📚 Testing Agent Library Scanner...")
        
        scanner = AgentLibraryScanner()
        
        # Test role inference
        assert scanner.infer_role_from_name("senior_developer") == "Developer"
        assert scanner.infer_role_from_name("systems_analyst") == "Analyst"
        assert scanner.infer_role_from_name("random_name") == "General Agent"
        
        # Test domain inference
        web_content = "This agent works with React and JavaScript"
        assert scanner.infer_domain_from_content(web_content) == "web development"
        
        data_content = "Uses pandas and numpy for data analysis"
        assert scanner.infer_domain_from_content(data_content) == "data science"
        print("✅ Agent Library Scanner working correctly")
        
        # Test 5: Agent text creation and searchability
        print("🔎 Testing agent text creation...")
        
        agent_text = qdrant.create_agent_text(agent)
        assert "Test Role" in agent_text
        assert "skill1" in agent_text
        assert "skill2" in agent_text
        assert "test" in agent_text.lower()
        print("✅ Agent text creation working")
        
        # Test 6: Performance characteristics
        print("⚡ Testing performance...")
        
        start_time = time.time()
        for i in range(10):
            embedding = qdrant.generate_embedding(f"Performance test {i}")
            assert len(embedding) == 384
        
        embedding_time = (time.time() - start_time) * 1000
        assert embedding_time < 5000, f"Embedding generation too slow: {embedding_time}ms"
        print(f"✅ Generated 10 embeddings in {embedding_time:.1f}ms")
        
        print("\n" + "=" * 60)
        print("🎉 ALL CORE TESTS PASSED!")
        print("=" * 60)
        print("✅ Enhanced Talent Scout core functionality is working correctly")
        print("📋 Key components verified:")
        print("   • Data models and validation")
        print("   • QDrant Manager and embedding generation")
        print("   • Agent Library Scanner and metadata extraction")
        print("   • Performance characteristics")
        print("   • Text processing and searchability")
        print("\n🚀 Enhanced Talent Scout is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CORE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_file_scanning():
    """Test file scanning functionality."""
    print("\n📂 Testing file scanning functionality...")
    
    try:
        from agents.talent_scout import AgentLibraryScanner
        
        # Create temporary test files
        temp_dir = tempfile.mkdtemp(prefix="scout_test_")
        temp_path = Path(temp_dir)
        
        # Create test agent file
        agent_file = temp_path / "test_agent.md"
        agent_content = """# Test Agent

**Role:** Test Developer
**Description:** A test agent for verification
**Capabilities:** Python, Testing, Verification
**Tools:** pytest, unittest, mock
**Domain:** software development
**Complexity:** medium
**Tags:** test, development, python

## System Prompt
This is a test agent for verification purposes.
"""
        agent_file.write_text(agent_content)
        
        # Create non-agent file
        regular_file = temp_path / "regular.txt"
        regular_file.write_text("This is just a regular text file.")
        
        # Scan the library
        scanner = AgentLibraryScanner()
        agents = await scanner.scan_library(str(temp_path))
        
        # Verify results
        assert len(agents) == 1, f"Expected 1 agent, got {len(agents)}"
        
        agent = agents[0]
        assert "Test" in agent.name
        assert agent.role == "Test Developer"
        assert "Python" in agent.capabilities
        assert "Testing" in agent.capabilities
        assert agent.domain == "software development"
        
        print("✅ File scanning functionality working correctly")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ File scanning test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🔥 Enhanced Talent Scout Core Functionality Test")
    print("=" * 60)
    
    # Run core functionality tests
    core_success = test_basic_functionality()
    
    # Run async file scanning test
    file_success = asyncio.run(test_file_scanning())
    
    overall_success = core_success and file_success
    
    if overall_success:
        print(f"\n🎯 FINAL RESULT: ✅ ALL TESTS PASSED")
        print(f"🚀 Enhanced Talent Scout implementation is READY")
        return 0
    else:
        print(f"\n🎯 FINAL RESULT: ❌ SOME TESTS FAILED")  
        print(f"⚠️  Implementation needs attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())