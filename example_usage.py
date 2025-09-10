"""
AgentForge Usage Examples

This file demonstrates various ways to use the AgentForge Engineering Manager
to create specialized agent teams for different types of goals.
"""

import asyncio
from orchestrator import EngineeringManager, InputGoal, create_orchestrator

async def example_web_development():
    """Example: Web development project orchestration."""
    
    print("\n🌐 Example: Web Development Project")
    print("=" * 40)
    
    orchestrator = create_orchestrator()
    
    goal = InputGoal(
        goal_description="Create a modern e-commerce platform with AI-powered product recommendations, real-time inventory, and multi-payment gateway support",
        domain="web development",
        complexity_level="enterprise",
        timeline="8 weeks",
        constraints=[
            "Must handle 10,000+ concurrent users",
            "GDPR compliant data handling",
            "Mobile-first design",
            "Sub-2-second page load times"
        ],
        success_criteria=[
            "User authentication and authorization system",
            "Product catalog with search and filtering",
            "Shopping cart and checkout process",
            "AI recommendation engine",
            "Real-time inventory management",
            "Multiple payment gateway integration",
            "Admin dashboard for store management",
            "Performance benchmarks met"
        ],
        existing_resources=[
            "React.js and Node.js expertise",
            "AWS cloud infrastructure",
            "Stripe payment integration experience",
            "PostgreSQL database setup"
        ]
    )
    
    team_package = await orchestrator.orchestrate_goal(goal)
    
    print(f"✅ Team: {team_package.team_name}")
    print(f"👥 Members: {len(team_package.team_members)}")
    print(f"📋 Workflow Steps: {len(team_package.workflow_steps)}")
    
    return team_package

async def example_data_science():
    """Example: Data science project orchestration."""
    
    print("\n📊 Example: Data Science Project")
    print("=" * 40)
    
    orchestrator = create_orchestrator()
    
    goal = InputGoal(
        goal_description="Build a machine learning system to predict customer churn with 95%+ accuracy, automated retraining, and real-time inference API",
        domain="data science",
        complexity_level="complex",
        timeline="6 weeks",
        constraints=[
            "Must use existing customer database",
            "Privacy-preserving ML techniques required",
            "Real-time predictions under 100ms",
            "Explainable AI for business stakeholders"
        ],
        success_criteria=[
            "Achieve 95%+ prediction accuracy",
            "Automated data pipeline",
            "Real-time inference API",
            "Model monitoring and alerting",
            "Explainable predictions",
            "A/B testing framework",
            "Automated retraining system"
        ],
        existing_resources=[
            "Python ML stack (scikit-learn, pandas)",
            "Customer transaction database",
            "Apache Kafka for streaming",
            "MLOps infrastructure (MLflow)"
        ]
    )
    
    team_package = await orchestrator.orchestrate_goal(goal)
    
    print(f"✅ Team: {team_package.team_name}")
    print(f"👥 Members: {len(team_package.team_members)}")
    print(f"🔧 New Agents: {len(team_package.new_agents_created)}")
    
    return team_package

async def example_mobile_app():
    """Example: Mobile app development orchestration."""
    
    print("\n📱 Example: Mobile App Project")
    print("=" * 40)
    
    orchestrator = create_orchestrator()
    
    goal = InputGoal(
        goal_description="Develop a cross-platform fitness tracking app with social features, wearable integration, and personalized coaching AI",
        domain="mobile development",
        complexity_level="complex",
        timeline="10 weeks",
        constraints=[
            "Cross-platform (iOS and Android)",
            "Offline-first architecture",
            "Integration with major wearables",
            "HIPAA compliance for health data"
        ],
        success_criteria=[
            "Activity tracking and logging",
            "Social features (friends, challenges)",
            "Wearable device integration",
            "AI-powered coaching recommendations",
            "Offline functionality",
            "Data synchronization",
            "App store approval",
            "Performance on mid-range devices"
        ],
        existing_resources=[
            "React Native development team",
            "Firebase backend infrastructure",
            "HealthKit/Google Fit integration experience",
            "Machine learning model for recommendations"
        ]
    )
    
    team_package = await orchestrator.orchestrate_goal(goal)
    
    print(f"✅ Team: {team_package.team_name}")
    print(f"📋 Strategy: {len(team_package.strategy_document)} characters")
    
    return team_package

async def example_simple_automation():
    """Example: Simple automation project."""
    
    print("\n⚙️ Example: Simple Automation")
    print("=" * 40)
    
    orchestrator = create_orchestrator()
    
    # Use the convenience method for simple goals
    team_package = await orchestrator.run_with_goal(
        goal_description="Automate daily report generation from multiple data sources and email distribution",
        domain="automation",
        complexity_level="simple",
        constraints=["Must run on existing Windows servers", "No additional software licenses"],
        success_criteria=["Daily reports generated automatically", "Email distribution working", "Error handling and notifications"]
    )
    
    print(f"✅ Team: {team_package.team_name}")
    print(f"♻️  Reused Agents: {len(team_package.existing_agents_used)}")
    
    return team_package

async def demonstrate_workflow_tracking():
    """Demonstrate workflow tracking capabilities."""
    
    print("\n📊 Workflow Tracking Demo")
    print("=" * 40)
    
    orchestrator = create_orchestrator()
    
    # Start a goal
    goal = InputGoal(
        goal_description="Create a simple blog platform with CMS",
        domain="web development",
        complexity_level="medium"
    )
    
    print("🔄 Starting orchestration...")
    team_package = await orchestrator.orchestrate_goal(goal)
    
    # Check workflow status
    status = orchestrator.get_workflow_status()
    print(f"📋 Workflow Status: {status.get('status', 'unknown')}")
    print(f"⏱️  Steps Completed: {len(status.get('steps', []))}")
    
    # Show workflow history
    print(f"📚 Total Workflows: {len(orchestrator.workflow_history)}")
    
    return team_package

async def run_all_examples():
    """Run all example scenarios."""
    
    print("🚀 AgentForge Examples")
    print("=" * 50)
    
    examples = [
        example_simple_automation,
        example_mobile_app,
        example_data_science, 
        example_web_development,
        demonstrate_workflow_tracking
    ]
    
    results = []
    
    for example in examples:
        try:
            result = await example()
            results.append(result)
            print("✅ Example completed successfully\n")
        except Exception as e:
            print(f"❌ Example failed: {e}\n")
            results.append(None)
    
    print("\n📊 Summary")
    print("=" * 20)
    successful = len([r for r in results if r is not None])
    print(f"✅ Successful: {successful}/{len(examples)}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_examples())