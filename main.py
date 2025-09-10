"""
AgentForge Main Entry Point - Updated for complete orchestration system.
"""

import asyncio
from typing import Optional
from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel


async def interactive_mode():
    """Interactive mode for AgentForge."""
    print("🔥 Welcome to AgentForge - The Meta-Agent System")
    print("=" * 50)
    
    # Get user input for goal
    goal_description = input("📝 Describe your goal: ")
    domain = input("🏷️  What domain (e.g., 'web development', 'data analysis'): ")
    
    # Complexity selection
    print("\n📊 Complexity levels:")
    print("1. Low - Simple, single-purpose tasks")
    print("2. Medium - Multi-component projects")
    print("3. High - Complex systems with integration")
    print("4. Enterprise - Large-scale, mission-critical systems")
    
    complexity_choice = input("Select complexity (1-4) [default: 2]: ").strip() or "2"
    complexity_map = {
        "1": ComplexityLevel.LOW,
        "2": ComplexityLevel.MEDIUM, 
        "3": ComplexityLevel.HIGH,
        "4": ComplexityLevel.ENTERPRISE
    }
    complexity = complexity_map.get(complexity_choice, ComplexityLevel.MEDIUM)
    
    # Optional inputs
    timeline = input("⏰ Timeline (optional): ").strip() or None
    constraints_input = input("🚫 Constraints (comma-separated, optional): ").strip()
    constraints = [c.strip() for c in constraints_input.split(",") if c.strip()] if constraints_input else []
    
    success_input = input("✅ Success criteria (comma-separated, optional): ").strip()
    success_criteria = [c.strip() for c in success_input.split(",") if c.strip()] if success_input else []
    
    # Create the goal
    goal = InputGoal(
        goal_description=goal_description,
        domain=domain,
        complexity_level=complexity,
        timeline=timeline,
        constraints=constraints,
        success_criteria=success_criteria
    )
    
    print(f"\n🎯 Processing your goal through AgentForge...")
    print(f"Goal: {goal.goal_description}")
    print(f"Domain: {goal.domain}")
    print(f"Complexity: {goal.complexity_level}")
    
    # Create and run the orchestrator
    em = EngineeringManager()
    
    try:
        result = await em.process(goal)
        
        print(f"\n🎉 AgentForge Complete!")
        print(f"📅 Created: {result.created_at}")
        print(f"🤖 New agents: {len(result.new_agents)}")
        print(f"📋 Strategy: {len(result.strategy_document)} chars")
        print(f"📖 Documentation: {len(result.roster_documentation)} chars")
        
        # Ask if user wants to see details
        show_details = input("\n📖 Show detailed results? (y/N): ").lower().startswith('y')
        
        if show_details:
            print(f"\n📋 Strategy Document:")
            print("-" * 40)
            print(result.strategy_document[:500] + "..." if len(result.strategy_document) > 500 else result.strategy_document)
            
            print(f"\n🕵️ Scouting Report:")
            print("-" * 40)
            print(result.scouting_report[:500] + "..." if len(result.scouting_report) > 500 else result.scouting_report)
            
            print(f"\n📖 Roster Documentation:")
            print("-" * 40)
            print(result.roster_documentation[:500] + "..." if len(result.roster_documentation) > 500 else result.roster_documentation)
            
            print(f"\n🚀 Deployment Instructions:")
            print("-" * 40)
            print(result.deployment_instructions)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def example_goal():
    """Run with an example goal."""
    print("🔥 AgentForge Example: Task Management System")
    
    goal = InputGoal(
        goal_description="Create a comprehensive task management system with real-time collaboration, user authentication, and mobile support",
        domain="web development",
        complexity_level=ComplexityLevel.HIGH,
        timeline="3 months",
        constraints=["React frontend", "Node.js backend", "Mobile responsive", "Budget: $75k"],
        success_criteria=[
            "User registration and authentication",
            "Real-time task updates",
            "Team collaboration features", 
            "Mobile-friendly interface",
            "File attachments and comments",
            "Reporting and analytics"
        ]
    )
    
    em = EngineeringManager()
    result = await em.process(goal)
    
    print(f"\n✅ Example Complete!")
    print(f"New agents created: {len(result.new_agents)}")
    print(f"Strategy length: {len(result.strategy_document)} characters")
    
    return result


async def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        await example_goal()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("🔥 AgentForge - Meta-Agent System")
        print("Usage:")
        print("  python main.py              # Interactive mode")
        print("  python main.py --example    # Run example")
        print("  python main.py --help       # Show this help")
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())