"""
Tutorial Examples for AgentForge Hands-On Guide

This file contains practical examples from the hands-on tutorial,
demonstrating both fine-grained control and automated optimization approaches.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel
from agents.agent_developer import AgentDeveloper, VacantRole


async def scenario_1_fine_grained_control():
    """
    Scenario 1: Fine-grained control example
    Create exactly 4 specific agents for DevDash project
    """
    print("\n🎯 Scenario 1: Fine-Grained Control")
    print("=" * 50)
    
    goal = InputGoal(
        goal_description=(
            "Create a specialized agent team for DevDash - a developer dashboard with real-time insights. "
            "I need exactly 4 agents: an Agno Agent Architect for system design, an Agentic Engineer "
            "for implementation, a Data Scientist for analytics pipeline, and an Infrastructure Expert "
            "for Kubernetes deployment."
        ),
        domain="full-stack development with data analytics",
        complexity_level=ComplexityLevel.HIGH,
        timeline="4 months",
        constraints=[
            "Must use Agno framework",
            "React/TypeScript frontend",
            "Node.js backend",
            "Kubernetes infrastructure", 
            "real-time data processing"
        ],
        success_criteria=[
            "Agno-based agent architecture",
            "Real-time dashboard functionality",
            "Scalable microservices",
            "ML-powered insights",
            "Production-ready Kubernetes deployment"
        ]
    )
    
    print(f"📝 Goal: {goal.goal_description[:100]}...")
    print(f"🏷️ Domain: {goal.domain}")
    print(f"📊 Complexity: {goal.complexity_level}")
    print(f"⏰ Timeline: {goal.timeline}")
    
    em = EngineeringManager()
    
    try:
        result = await em.process(goal)
        
        print(f"\n✅ Fine-Grained Team Created!")
        print(f"🤖 New agents: {len(result.new_agents)}")
        print(f"📋 Strategy length: {len(result.strategy_document)} chars")
        
        # Display agent summary
        print(f"\n👥 Team Members:")
        for i, agent in enumerate(result.new_agents, 1):
            print(f"  {i}. {agent.get('name', 'Unknown')} - {agent.get('role', 'Unknown role')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in fine-grained control: {e}")
        return None


async def scenario_2_automated_optimization():
    """
    Scenario 2: Automated optimization example
    Let AgentForge decide the optimal team composition
    """
    print("\n🤖 Scenario 2: Automated Optimization")
    print("=" * 50)
    
    goal = InputGoal(
        goal_description=(
            "Analyze the DevDash PRD and tech stack documents to create an optimally balanced agent team. "
            "Focus on delivering a production-ready developer dashboard with real-time analytics, "
            "high performance, and excellent user experience."
        ),
        domain="enterprise web application development",
        complexity_level=ComplexityLevel.ENTERPRISE,
        timeline="6 months",
        constraints=[
            "Must follow tech stack in docs/sample_tech_stack.md",
            "SOC 2 compliance required",
            "budget-conscious approach preferred"
        ],
        success_criteria=[
            "Production-ready application",
            "All PRD requirements met",
            "Scalable architecture",
            "Comprehensive testing",
            "Security compliance"
        ]
    )
    
    print(f"📝 Goal: {goal.goal_description[:100]}...")
    print(f"🏷️ Domain: {goal.domain}")
    print(f"📊 Complexity: {goal.complexity_level}")
    print(f"⏰ Timeline: {goal.timeline}")
    
    em = EngineeringManager()
    
    try:
        result = await em.process(goal)
        
        print(f"\n✅ Optimized Team Created!")
        print(f"🤖 New agents: {len(result.new_agents)}")
        print(f"📋 Strategy length: {len(result.strategy_document)} chars")
        
        # Display optimization insights
        print(f"\n🔍 Optimization Results:")
        print(f"  📊 Total agents: {len(result.new_agents)}")
        print(f"  💰 Estimated efficiency: High (automated optimization)")
        print(f"  🎯 Coverage: Comprehensive")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in automated optimization: {e}")
        return None


async def create_custom_agno_architect():
    """
    Example: Create a custom Agno Agent Architect with specific requirements
    """
    print("\n🏗️ Creating Custom Agno Agent Architect")
    print("=" * 50)
    
    role = VacantRole(
        role_name="Agno Agent Architect",
        title="Senior Agno Framework Architect",
        core_responsibilities=[
            "Design multi-agent system architectures using Agno",
            "Create agent workflow orchestration patterns",
            "Optimize agent communication and coordination",
            "Ensure scalable and maintainable agent systems"
        ],
        required_capabilities=[
            "Agno framework expertise",
            "System architecture design",
            "Agent workflow patterns",
            "Performance optimization",
            "Documentation and standards"
        ],
        interaction_patterns={
            "engineering_manager": "reports_to",
            "agentic_engineer": "collaborates_with",
            "infrastructure_expert": "coordinates_with"
        },
        success_metrics=[
            "System architecture clarity and completeness",
            "Agent workflow efficiency",
            "Team coordination effectiveness",
            "Documentation quality"
        ],
        priority_level="critical"
    )
    
    print(f"🎯 Role: {role.role_name}")
    print(f"📋 Responsibilities: {len(role.core_responsibilities)} items")
    print(f"🛠️ Capabilities: {len(role.required_capabilities)} required")
    
    try:
        developer = AgentDeveloper()
        spec = await developer.create_agent_from_role(role)
        
        print(f"\n✅ Custom Agent Created!")
        print(f"📛 Name: {spec.name}")
        print(f"🎭 Role: {spec.role}")
        print(f"🛠️ Tools: {', '.join(spec.tools_required[:3])}...")
        print(f"💬 Communication: {spec.communication_style}")
        
        return spec
        
    except Exception as e:
        print(f"❌ Error creating custom agent: {e}")
        return None


async def compare_approaches():
    """
    Compare the results of both approaches
    """
    print("\n📊 Comparing Approaches")
    print("=" * 50)
    
    # Run both scenarios
    fine_grained_result = await scenario_1_fine_grained_control()
    automated_result = await scenario_2_automated_optimization()
    
    if fine_grained_result and automated_result:
        print(f"\n🔍 Comparison Results:")
        print(f"Fine-Grained Control:")
        print(f"  🤖 Agents: {len(fine_grained_result.new_agents)}")
        print(f"  🎯 Precision: High (exact specifications)")
        print(f"  💰 Efficiency: Medium (may have gaps/overlaps)")
        
        print(f"\nAutomated Optimization:")
        print(f"  🤖 Agents: {len(automated_result.new_agents)}")
        print(f"  🎯 Precision: Medium (system-optimized)")
        print(f"  💰 Efficiency: High (optimal resource allocation)")
        
        print(f"\n💡 Recommendation:")
        if len(fine_grained_result.new_agents) < len(automated_result.new_agents):
            print("  Fine-grained approach created fewer agents - good for focused teams")
        else:
            print("  Automated approach optimized for comprehensive coverage")
    
    return fine_grained_result, automated_result


async def save_results(results: Dict[str, Any], filename: str):
    """Save tutorial results to file"""
    output_dir = Path("tutorial_output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / filename
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {output_file}")


async def run_tutorial_examples():
    """
    Run all tutorial examples
    """
    print("🚀 AgentForge Tutorial Examples")
    print("=" * 60)
    
    results = {}
    
    try:
        # Scenario 1: Fine-grained control
        fine_grained = await scenario_1_fine_grained_control()
        results['fine_grained'] = {
            'success': fine_grained is not None,
            'agent_count': len(fine_grained.new_agents) if fine_grained else 0
        }
        
        # Scenario 2: Automated optimization  
        automated = await scenario_2_automated_optimization()
        results['automated'] = {
            'success': automated is not None,
            'agent_count': len(automated.new_agents) if automated else 0
        }
        
        # Custom agent creation
        custom_agent = await create_custom_agno_architect()
        results['custom_agent'] = {
            'success': custom_agent is not None,
            'agent_name': custom_agent.name if custom_agent else None
        }
        
        # Comparison
        await compare_approaches()
        
        # Save results
        await save_results(results, "tutorial_results.json")
        
        print(f"\n🎉 Tutorial Complete!")
        print(f"✅ Fine-grained: {'Success' if results['fine_grained']['success'] else 'Failed'}")
        print(f"✅ Automated: {'Success' if results['automated']['success'] else 'Failed'}")
        print(f"✅ Custom Agent: {'Success' if results['custom_agent']['success'] else 'Failed'}")
        
    except Exception as e:
        print(f"❌ Tutorial failed: {e}")
        results['error'] = str(e)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_tutorial_examples())
