"""
Demo script for Integration Architect Agent

This script demonstrates the Integration Architect's capability to create 
comprehensive operational playbooks for agent teams.
"""

import asyncio
from agents.integration_architect import (
    IntegrationArchitect, 
    StrategyDocument, 
    ScoutingReport, 
    NewAgent
)


async def demo_integration_architect():
    """Demonstrate the Integration Architect functionality."""
    
    print("🏗️ Integration Architect Demo - Creating Team Operational Playbook")
    print("=" * 80)
    
    # Initialize the Integration Architect
    print("\n1️⃣ Initializing Integration Architect...")
    architect = IntegrationArchitect(db_file="demo_integration.db")
    
    # Create test data representing outputs from other agents
    print("\n2️⃣ Preparing input data from other agents...")
    
    # Strategy Document from Systems Analyst
    strategy_doc = StrategyDocument(
        goal_analysis={
            "complexity": "high",
            "domain": "e-commerce",
            "scope": "full-stack web application",
            "key_challenges": ["user authentication", "payment processing", "inventory management"],
            "success_factors": ["scalability", "security", "user experience"]
        },
        team_composition=[
            {
                "role": "Frontend Developer",
                "capabilities": ["react", "typescript", "ui/ux", "responsive_design"],
                "responsibilities": ["user interface", "user experience", "client-side logic"]
            },
            {
                "role": "Backend Developer", 
                "capabilities": ["node.js", "api_design", "database_design", "authentication"],
                "responsibilities": ["api development", "business logic", "data management"]
            },
            {
                "role": "DevOps Engineer",
                "capabilities": ["aws", "docker", "ci/cd", "monitoring", "security"],
                "responsibilities": ["deployment", "infrastructure", "monitoring", "security"]
            },
            {
                "role": "QA Engineer",
                "capabilities": ["testing", "automation", "quality_assurance", "bug_tracking"],
                "responsibilities": ["test planning", "automated testing", "quality gates"]
            }
        ],
        team_structure={
            "topology": "hierarchical",
            "coordination": "agile_workflows",
            "communication": "slack_and_standups",
            "decision_making": "tech_lead_driven"
        },
        risk_assessment=[
            "Integration complexity between frontend and backend",
            "Payment processing security requirements",
            "Scalability under high load",
            "Timeline pressure affecting quality"
        ],
        resource_requirements={
            "development_environment": "cloud-based",
            "testing_infrastructure": "automated_pipelines",
            "deployment_platform": "aws_ecs",
            "monitoring_tools": "datadog_newrelic"
        },
        timeline_estimate={
            "planning": "1 week",
            "development": "6 weeks", 
            "testing": "2 weeks",
            "deployment": "1 week",
            "total": "10 weeks"
        }
    )
    
    # Scouting Report from Talent Scout
    scouting_report = ScoutingReport(
        matched_agents=[
            {
                "name": "ExperiencedFrontendDev",
                "match_score": 0.92,
                "capabilities": ["react", "typescript", "tailwind", "nextjs"],
                "experience": "5+ years",
                "availability": "full-time"
            },
            {
                "name": "SeniorQAEngineer",
                "match_score": 0.88,
                "capabilities": ["selenium", "jest", "cypress", "performance_testing"],
                "experience": "7+ years",
                "availability": "part-time"
            }
        ],
        capability_gaps=[
            {
                "role": "Backend Developer",
                "gap_size": 1.0,
                "missing_capabilities": ["node.js", "microservices", "payment_integration"],
                "priority": "critical"
            },
            {
                "role": "DevOps Engineer", 
                "gap_size": 1.0,
                "missing_capabilities": ["aws", "kubernetes", "security_compliance"],
                "priority": "high"
            }
        ],
        reuse_recommendations={
            "frontend_development": "reuse_existing_with_upskilling",
            "qa_processes": "reuse_existing_expand_scope",
            "backend_development": "create_new_specialist",
            "devops_infrastructure": "create_new_specialist"
        },
        library_analysis={
            "total_agents_reviewed": 150,
            "matching_agents_found": 2,
            "capability_coverage": "40%",
            "recommendation": "hybrid_approach_new_and_existing"
        }
    )
    
    # New Agents from Agent Developer
    new_agents = [
        NewAgent(
            name="E-commerceBackendSpecialist",
            role="Backend Developer",
            capabilities=[
                "node.js", "express", "mongodb", "stripe_integration", 
                "jwt_authentication", "microservices", "api_documentation"
            ],
            system_prompt="""
            You are an E-commerce Backend Specialist with expertise in building scalable,
            secure backend systems for online retail platforms.
            
            Your core responsibilities:
            - Design and implement RESTful APIs for e-commerce functionality
            - Integrate payment processing systems (Stripe, PayPal)
            - Implement secure user authentication and authorization
            - Design efficient database schemas for products, orders, and users
            - Ensure system security and compliance with payment standards
            - Create comprehensive API documentation
            - Implement logging and monitoring for backend services
            
            You work closely with the Frontend Developer to ensure seamless API integration
            and coordinate with the DevOps Engineer for deployment and scaling requirements.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": ["api_design", "implementation", "security_review"]},
                    "requirements": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}}
                }
            },
            output_schema={
                "type": "object", 
                "properties": {
                    "deliverable": {"type": "string"},
                    "status": {"type": "string"},
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}}
                }
            },
            dependencies=["Frontend Developer", "DevOps Engineer"]
        ),
        NewAgent(
            name="CloudInfrastructureSpecialist", 
            role="DevOps Engineer",
            capabilities=[
                "aws", "docker", "kubernetes", "terraform", "ci/cd_pipelines",
                "monitoring", "security_compliance", "cost_optimization"
            ],
            system_prompt="""
            You are a Cloud Infrastructure Specialist focused on deploying and managing
            scalable e-commerce applications in AWS environments.
            
            Your core responsibilities:
            - Design and implement cloud infrastructure using Infrastructure as Code
            - Set up CI/CD pipelines for automated deployment
            - Implement monitoring, logging, and alerting systems
            - Ensure security compliance and best practices
            - Optimize infrastructure costs and performance
            - Manage container orchestration with Kubernetes
            - Coordinate with development team for deployment requirements
            
            You ensure the application can scale to handle traffic spikes and maintain
            high availability while keeping operational costs manageable.
            """,
            input_schema={
                "type": "object",
                "properties": {
                    "infrastructure_type": {"type": "string", "enum": ["setup", "deployment", "monitoring", "optimization"]},
                    "requirements": {"type": "string"},
                    "environment": {"type": "string", "enum": ["development", "staging", "production"]}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "infrastructure_plan": {"type": "string"},
                    "deployment_status": {"type": "string"},
                    "monitoring_setup": {"type": "object"},
                    "cost_estimate": {"type": "number"}
                }
            },
            dependencies=["Backend Developer"]
        )
    ]
    
    # Display the input summary
    print(f"\n📊 Input Summary:")
    print(f"   • Strategy defines {len(strategy_doc.team_composition)} required roles")
    print(f"   • Scout found {len(scouting_report.matched_agents)} matching existing agents")
    print(f"   • Scout identified {len(scouting_report.capability_gaps)} capability gaps")
    print(f"   • Developer created {len(new_agents)} new specialized agents")
    
    # Execute integration
    print(f"\n3️⃣ Executing Integration Architect analysis...")
    print("   (This may take a moment as the AI analyzes team integration...)")
    
    original_goal = """
    Build a complete e-commerce web application with the following requirements:
    - User registration and authentication system
    - Product catalog with search and filtering
    - Shopping cart and checkout process
    - Payment processing integration
    - Order management and history
    - Admin panel for inventory management
    - Mobile-responsive design
    - High performance and scalability
    - Security compliance for payment processing
    """
    
    try:
        roster_documentation = await architect.integrate_team(
            strategy_doc,
            scouting_report,
            new_agents,
            original_goal.strip()
        )
        
        print("\n4️⃣ Integration Analysis Complete!")
        print("=" * 60)
        print(roster_documentation)
        
        # Create the final documentation
        print(f"\n5️⃣ Creating Roster Documentation...")
        doc_path = architect.create_roster_documentation(
            roster_documentation,
            "E-commerce Development Team",
            "EcommerceDevelopmentTeam_Roster.md"
        )
        
        print(f"✅ Roster documentation created: {doc_path}")
        print(f"\n🎯 Integration Architect has successfully created a comprehensive")
        print(f"   operational playbook for the assembled agent team!")
        
        print(f"\n📋 Key Deliverables:")
        print(f"   • Complete team operational playbook")
        print(f"   • Detailed workflow steps and coordination protocols")
        print(f"   • Communication patterns between agents")
        print(f"   • Quality gates and success metrics")
        print(f"   • Deployment instructions and troubleshooting guide")
        
        # Test quick integration functionality
        print(f"\n6️⃣ Testing Quick Integration Feature...")
        quick_result = await architect.quick_integration(
            "Mobile app development team",
            "Create a social media mobile application",
            "iOS Developer, Android Developer, Backend API Developer, UI/UX Designer"
        )
        
        print(f"\n📱 Quick Integration Result:")
        print("=" * 40)
        print(quick_result)
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_integration_architect())