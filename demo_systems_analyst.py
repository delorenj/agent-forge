"""
Demo: Systems Analyst in AgentForge Workflow

This demonstrates how the Systems Analyst receives goals from the Engineering Manager
and produces Strategy Documents for the Talent Scout.
"""

import asyncio
from agents.systems_analyst import SystemsAnalyst, InputGoal


async def demo_workflow():
    """Demonstrate the Systems Analyst workflow"""
    
    print("🏭 AgentForge Demo: Systems Analyst Workflow")
    print("=" * 50)
    
    # Step 1: Engineering Manager delegates to Systems Analyst
    print("\n1️⃣ Engineering Manager receives goal and delegates to Systems Analyst")
    
    # Simulated goal from Engineering Manager
    incoming_goal = InputGoal(
        description="Build a comprehensive project management system with team collaboration, task tracking, time management, and reporting capabilities",
        context="Mid-size software company (50-200 employees), remote-first, using agile methodologies, needs to replace multiple tools",
        success_criteria=[
            "Centralize project management across all teams",
            "Improve team collaboration and visibility", 
            "Reduce time spent in status meetings by 50%",
            "Generate automated progress reports",
            "Integrate with existing tools (Slack, GitHub, etc.)",
            "Support both agile and waterfall methodologies"
        ],
        domain="Project Management / Collaboration",
        complexity="High"
    )
    
    print(f"   📋 Goal: {incoming_goal.description}")
    print(f"   🏢 Context: {incoming_goal.context}")
    print(f"   📊 Success Criteria: {len(incoming_goal.success_criteria)} defined")
    
    # Step 2: Systems Analyst analyzes the goal
    print("\n2️⃣ Systems Analyst analyzes goal and defines ideal team structure")
    print("   🧠 Using reasoning tools to decompose the problem...")
    print("   📚 Searching knowledge base for relevant patterns...")
    
    analyst = SystemsAnalyst(knowledge_base_path="./docs/agno")
    
    try:
        # Perform the analysis
        strategy = await analyst.analyze_goal(incoming_goal)
        
        print("\n   ✅ Analysis complete!")
        print(f"   📝 Strategy length: {len(strategy.split())} words")
        
        # Step 3: Create Strategy Document
        print("\n3️⃣ Creating Strategy Document for Talent Scout")
        
        doc_path = analyst.create_strategy_document(strategy, "agent-strategy.md")
        
        print(f"   📄 Strategy Document created: {doc_path}")
        print("   📤 Ready for Talent Scout to analyze existing resources")
        
        # Step 4: Show sample output
        print("\n4️⃣ Strategy Document Preview:")
        print("=" * 30)
        
        # Show first part of the strategy
        preview = strategy[:800] + "..." if len(strategy) > 800 else strategy
        print(preview)
        
        print("\n🔄 Next Steps in AgentForge Workflow:")
        print("   → Talent Scout analyzes existing agent pool")
        print("   → Identifies matches and gaps")
        print("   → Agent Developer creates new agents for gaps")
        print("   → Integration Architect assembles final team")
        
        print("\n✅ Systems Analyst workflow demonstration complete!")
        
    except Exception as e:
        print(f"   ❌ Error during analysis: {e}")
        print("   💡 This might be due to missing API keys or network issues")
        
        # Show what the output structure would look like
        print("\n   📋 Expected Strategy Document Structure:")
        print("   • Goal Analysis: Problem decomposition and requirements")
        print("   • Team Composition: Specific agent roles and responsibilities")
        print("   • Team Structure: Organization and coordination patterns")
        print("   • Risk Assessment: Challenges and mitigation strategies")
        print("   • Resource Requirements: Technical and operational needs")
        print("   • Timeline Estimate: Development and deployment phases")


if __name__ == "__main__":
    asyncio.run(demo_workflow())