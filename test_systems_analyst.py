"""
Test script for Systems Analyst agent

This script demonstrates how the Systems Analyst works:
1. Receives input goals
2. Analyzes them using reasoning patterns
3. Produces comprehensive strategy documents
4. Defines ideal team structures
"""

import asyncio
from agents.systems_analyst import SystemsAnalyst, InputGoal


async def test_complex_goal():
    """Test with a complex multi-faceted goal"""
    print("🚀 Testing Systems Analyst with Complex Goal")
    print("=" * 60)
    
    # Initialize the analyst
    analyst = SystemsAnalyst(knowledge_base_path="./docs/agno")
    
    # Define a complex test goal
    complex_goal = InputGoal(
        description="Build an AI-powered e-commerce platform with personalized recommendations, real-time inventory management, multi-vendor support, and advanced analytics dashboard",
        context="Startup with $2M funding, need to launch in 6 months, targeting 100K+ users, must be scalable and secure",
        success_criteria=[
            "Handle 10K concurrent users",
            "Process 1M+ products from 500+ vendors", 
            "Achieve 25% conversion rate improvement via AI recommendations",
            "Real-time inventory sync across all channels",
            "Comprehensive analytics for vendors and admins",
            "Mobile-first responsive design",
            "PCI DSS compliance for payments"
        ],
        domain="E-commerce / AI / SaaS",
        complexity="Very High"
    )
    
    print("📋 Input Goal:")
    print(f"  Description: {complex_goal.description}")
    print(f"  Context: {complex_goal.context}")
    print(f"  Domain: {complex_goal.domain}")
    print(f"  Complexity: {complex_goal.complexity}")
    print(f"  Success Criteria: {len(complex_goal.success_criteria)} items")
    
    print("\n🔍 Analyzing goal (this may take a moment)...")
    
    try:
        # Analyze the goal
        strategy_result = await analyst.analyze_goal(complex_goal)
        
        print("\n📄 Strategy Document Generated:")
        print("=" * 40)
        print(strategy_result)
        
        # Create the formatted document
        doc_path = analyst.create_strategy_document(strategy_result, "test-strategy.md")
        print(f"\n✅ Strategy document saved to: {doc_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True


async def test_simple_goal():
    """Test with a simpler goal"""
    print("\n🎯 Testing Systems Analyst with Simple Goal")
    print("=" * 60)
    
    analyst = SystemsAnalyst()
    
    simple_goal = "Create a blog website with user authentication and commenting system"
    
    print(f"📋 Input Goal: {simple_goal}")
    print("\n🔍 Analyzing goal...")
    
    try:
        result = await analyst.quick_analysis(simple_goal)
        
        print("\n📄 Quick Analysis Result:")
        print("=" * 30)
        print(result)
        
    except Exception as e:
        print(f"❌ Error during quick analysis: {e}")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🧠 Systems Analyst Agent Test Suite")
    print("=====================================")
    
    # Test complex goal
    success1 = await test_complex_goal()
    
    # Test simple goal  
    success2 = await test_simple_goal()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"  Complex Goal Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"  Simple Goal Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Systems Analyst is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())