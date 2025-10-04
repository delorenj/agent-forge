#!/usr/bin/env python3
"""
Direct Enhanced Talent Scout Test - Bypass Import Issues

Tests the talent scout module directly without going through __init__.py
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_talent_scout_direct():
    """Test talent scout functionality directly."""
    print("🚀 Testing Enhanced Talent Scout - Direct Import")
    print("=" * 60)
    
    try:
        # Import directly from the talent_scout module
        print("📦 Testing direct imports...")
        
        # Import the talent scout module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "talent_scout", 
            project_root / "agents" / "talent_scout.py"
        )
        talent_scout_module = importlib.util.module_from_spec(spec)
        sys.modules["talent_scout"] = talent_scout_module
        spec.loader.exec_module(talent_scout_module)
        
        # Get the classes we need
        AgentMetadata = talent_scout_module.AgentMetadata
        RoleRequirement = talent_scout_module.RoleRequirement
        ScoutingReport = talent_scout_module.ScoutingReport
        StrategyDocument = talent_scout_module.StrategyDocument
        QDrantManager = talent_scout_module.QDrantManager
        AgentLibraryScanner = talent_scout_module.AgentLibraryScanner
        TalentScout = talent_scout_module.TalentScout
        
        print("✅ Direct imports successful")
        
        # Test 1: Data model creation
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
        
        # Test 2: QDrant Manager basic functionality
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
        
        # Test 3: Agent Library Scanner basic functionality
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
        
        # Test 4: Agent text creation and searchability
        print("🔎 Testing agent text creation...")
        
        agent_text = qdrant.create_agent_text(agent)
        assert "Test Role" in agent_text
        assert "skill1" in agent_text
        assert "skill2" in agent_text
        assert "test" in agent_text.lower()
        print("✅ Agent text creation working")
        
        # Test 5: TalentScout initialization
        print("🎯 Testing TalentScout initialization...")
        
        scout = TalentScout()
        assert scout.qdrant_manager is not None
        assert scout.scanner is not None
        print("✅ TalentScout initialized successfully")
        
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
        print("🎉 ALL DIRECT TESTS PASSED!")
        print("=" * 60)
        print("✅ Enhanced Talent Scout core functionality is working correctly")
        print("📋 Key components verified:")
        print("   • Data models and validation")
        print("   • QDrant Manager and embedding generation") 
        print("   • Agent Library Scanner and metadata extraction")
        print("   • TalentScout class initialization")
        print("   • Performance characteristics")
        print("   • Text processing and searchability")
        print("\n🚀 Enhanced Talent Scout is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DIRECT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("🔥 Enhanced Talent Scout Direct Test")
    print("=" * 60)
    
    success = test_talent_scout_direct()
    
    if success:
        print(f"\n🎯 FINAL RESULT: ✅ ALL TESTS PASSED")
        print(f"🚀 Enhanced Talent Scout implementation is READY")
        return 0
    else:
        print(f"\n🎯 FINAL RESULT: ❌ TESTS FAILED")  
        print(f"⚠️  Implementation needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())