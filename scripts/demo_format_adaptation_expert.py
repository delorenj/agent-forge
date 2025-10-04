"""
Demo: Format Adaptation Expert - Platform Adapter

Demonstrates how the Format Adaptation Expert can adapt AgentForge roster outputs
to different client platforms like Claude Code, OpenCode, and Amazon Q.
"""

import asyncio
import json
from agents.format_adaptation_expert import (
    FormatAdaptationExpert,
    SourceAgent,
    AdaptationRequest
)


def create_sample_roster_agents():
    """Create a sample AgentForge roster to demonstrate adaptation"""
    
    # Data Analysis Team from AgentForge
    agents = []
    
    # 1. Lead Data Scientist
    agents.append(SourceAgent(
        name="LeadDataScientist",
        role="Senior Data Science Lead",
        description="Expert data scientist who leads complex analytics projects and mentors team members",
        capabilities=[
            "Advanced statistical modeling",
            "Machine learning implementation",
            "Data pipeline architecture", 
            "Team leadership",
            "Business intelligence",
            "Predictive analytics"
        ],
        instructions="""You are a Senior Data Science Lead with deep expertise in statistical modeling and machine learning.

Your responsibilities:
- Design and implement complex data science solutions
- Lead cross-functional analytics projects from concept to deployment
- Mentor junior data scientists and analysts
- Translate business requirements into technical solutions
- Ensure data quality and model performance
- Communicate insights to executive stakeholders

Your approach:
1. Understand business context and success metrics
2. Design robust data collection and preprocessing pipelines
3. Select appropriate algorithms and modeling techniques
4. Validate models thoroughly using statistical methods
5. Deploy scalable solutions with monitoring
6. Present findings with clear visualizations and recommendations

Focus on scalable, production-ready solutions that drive business value.""",
        interaction_patterns={
            "input": "Business problems, raw datasets, performance requirements",
            "output": "ML models, analysis reports, strategic recommendations, team guidance",
            "coordination": "Leads data team, collaborates with engineering and business stakeholders"
        },
        metadata={
            "seniority": "senior",
            "domain": "data_science",
            "team_size": "5-8 people",
            "complexity": "high"
        }
    ))
    
    # 2. ML Engineer
    agents.append(SourceAgent(
        name="MLEngineer",
        role="Machine Learning Engineer", 
        description="Specialist in deploying and scaling machine learning models in production environments",
        capabilities=[
            "Model deployment",
            "MLOps implementation",
            "Performance optimization",
            "Cloud infrastructure",
            "API development",
            "Model monitoring"
        ],
        instructions="""You are a Machine Learning Engineer focused on productionizing ML models and building robust MLOps pipelines.

Your expertise areas:
- Deploy ML models to cloud platforms (AWS, GCP, Azure)
- Build scalable ML pipelines using tools like Kubeflow, MLflow, or Airflow
- Implement model versioning, A/B testing, and monitoring
- Optimize model performance for production workloads
- Design RESTful APIs for model serving
- Set up continuous integration/deployment for ML systems

Your workflow:
1. Collaborate with data scientists to understand model requirements
2. Design scalable deployment architecture
3. Implement automated testing and validation pipelines
4. Deploy models with proper monitoring and alerting
5. Optimize performance and cost efficiency
6. Maintain model performance through retraining pipelines

Prioritize reliability, scalability, and maintainability in all ML systems.""",
        interaction_patterns={
            "input": "Trained models, deployment requirements, performance constraints",
            "output": "Production ML systems, APIs, monitoring dashboards, deployment pipelines",
            "coordination": "Works with data scientists, DevOps, and product teams"
        },
        metadata={
            "seniority": "mid-senior",
            "domain": "ml_engineering", 
            "focus": "production_deployment"
        }
    ))
    
    # 3. Business Analyst
    agents.append(SourceAgent(
        name="BusinessAnalyst",
        role="Senior Business Analyst",
        description="Expert in translating business needs into data requirements and interpreting analytics for decision-making",
        capabilities=[
            "Requirements gathering",
            "Stakeholder communication",
            "Business process analysis",
            "KPI definition",
            "Data storytelling",
            "Strategic planning"
        ],
        instructions="""You are a Senior Business Analyst who bridges the gap between business needs and data science solutions.

Your core competencies:
- Gather and document business requirements for analytics projects
- Define meaningful KPIs and success metrics
- Analyze business processes to identify optimization opportunities  
- Translate complex data insights into actionable business recommendations
- Facilitate communication between technical teams and business stakeholders
- Design dashboards and reports for executive decision-making

Your methodology:
1. Conduct stakeholder interviews to understand business challenges
2. Document current state processes and identify improvement areas
3. Define clear success criteria and measurement frameworks
4. Collaborate with data teams to design appropriate solutions
5. Validate analyses against business logic and domain expertise
6. Present findings with compelling narratives and visualizations

Focus on driving measurable business impact through data-driven insights.""",
        interaction_patterns={
            "input": "Business challenges, stakeholder requirements, process documentation", 
            "output": "Requirements documents, KPI frameworks, business recommendations, executive reports",
            "coordination": "Facilitates between business stakeholders and technical teams"
        },
        metadata={
            "seniority": "senior",
            "domain": "business_analysis",
            "stakeholder_level": "executive"
        }
    ))
    
    return agents


async def demo_multi_platform_adaptation():
    """Demonstrate adapting the same roster to multiple platforms"""
    
    print("🚀 Format Adaptation Expert Demo")
    print("=" * 60)
    print("Converting AgentForge Data Analysis Team to multiple platforms\n")
    
    # Initialize expert
    expert = FormatAdaptationExpert()
    
    # Get sample roster
    roster_agents = create_sample_roster_agents()
    
    print(f"📋 Source Roster: {len(roster_agents)} agents")
    for i, agent in enumerate(roster_agents, 1):
        print(f"   {i}. {agent.name} ({agent.role})")
    
    # Platform adaptations to demonstrate
    platforms = [
        {
            "name": "claude-code",
            "description": "Claude Code MCP format",
            "customizations": {
                "temperature": 0.7,
                "model": "claude-3-5-sonnet-20241022"
            }
        },
        {
            "name": "opencode", 
            "description": "VS Code extension format",
            "customizations": {
                "category": "Data Science",
                "activation_events": ["onLanguage:python", "onLanguage:r"]
            }
        },
        {
            "name": "amazonq",
            "description": "Amazon Q skill format", 
            "customizations": {
                "skill_category": "Analytics",
                "invocation_phrases": ["analyze data", "create model", "generate insights"]
            }
        }
    ]
    
    # Adapt to each platform
    for platform in platforms:
        print(f"\n🔄 Adapting to {platform['name']} ({platform['description']})...")
        
        request = AdaptationRequest(
            source_agents=roster_agents,
            target_platform=platform["name"],
            customizations=platform["customizations"],
            preserve_semantics=True
        )
        
        try:
            result = await expert.adapt_agents(request)
            
            print(f"✅ {result.summary}")
            
            successful_adaptations = [a for a in result.adapted_agents if a.validation_status == "valid"]
            warning_adaptations = [a for a in result.adapted_agents if a.validation_status == "warning"]
            error_adaptations = [a for a in result.adapted_agents if a.validation_status == "error"]
            
            print(f"   ✅ Successful: {len(successful_adaptations)}")
            print(f"   ⚠️  Warnings: {len(warning_adaptations)}")  
            print(f"   ❌ Errors: {len(error_adaptations)}")
            
            if result.warnings:
                print(f"   Issues: {'; '.join(result.warnings[:2])}")
            
            # Show sample adapted content
            if successful_adaptations:
                sample_agent = successful_adaptations[0]
                print(f"\n📄 Sample adapted content ({sample_agent.original_name}):")
                content_lines = sample_agent.adapted_content.split('\n')
                preview_lines = content_lines[:8] + (['...'] if len(content_lines) > 8 else [])
                for line in preview_lines:
                    print(f"      {line}")
            
            print(f"\n📋 Deployment Instructions:")
            for instruction in result.deployment_instructions[:3]:
                print(f"   • {instruction}")
            
        except Exception as e:
            print(f"❌ Adaptation failed: {str(e)}")
    
    return expert


async def demo_custom_platform_adaptation():
    """Demonstrate adapting to a custom platform format"""
    
    print(f"\n" + "=" * 60)
    print("🔧 Custom Platform Adaptation Demo")
    print("Creating adapter for custom 'TeamBot' platform\n")
    
    expert = FormatAdaptationExpert()
    
    # Define custom platform template
    from agents.format_adaptation_expert import PlatformTemplate
    
    custom_template = PlatformTemplate(
        platform="teambot",
        format_type="yaml",
        structure={
            "bot_name": "string",
            "bot_role": "string", 
            "personality": "object",
            "skills": "array",
            "responses": "object",
            "integrations": "array"
        },
        example="""
bot_name: DataExpert
bot_role: Senior Data Analyst
personality:
  tone: professional
  style: analytical
  expertise_level: senior
skills:
  - statistical_analysis
  - data_visualization
  - report_generation
responses:
  greeting: "Hello! I'm your data analysis expert."
  help: "I can help you analyze datasets and generate insights."
integrations:
  - slack
  - microsoft_teams
        """.strip(),
        requirements=[
            "Must include bot_name and bot_role",
            "Personality must specify tone and style", 
            "At least 3 skills required",
            "Must define greeting and help responses"
        ],
        constraints=[
            "bot_name must be alphanumeric",
            "skills limited to predefined list",
            "integrations must be supported platforms"
        ]
    )
    
    # Get one agent to adapt
    sample_agents = create_sample_roster_agents()
    lead_scientist = sample_agents[0]  # Use the Lead Data Scientist
    
    print(f"🎯 Adapting {lead_scientist.name} to custom TeamBot platform...")
    
    request = AdaptationRequest(
        source_agents=[lead_scientist],
        target_platform="teambot",
        platform_template=custom_template,
        customizations={
            "personality_tone": "expert_friendly",
            "integration_priority": ["slack", "microsoft_teams"],
            "response_style": "detailed_explanations"
        },
        preserve_semantics=True
    )
    
    try:
        result = await expert.adapt_agents(request)
        
        print(f"✅ {result.summary}")
        
        if result.adapted_agents:
            adapted = result.adapted_agents[0]
            print(f"   Validation: {adapted.validation_status}")
            print(f"   Format: {adapted.format_type}")
            
            print(f"\n📄 Custom Platform Adaptation:")
            print("-" * 40)
            print(adapted.adapted_content)
            print("-" * 40)
            
            if adapted.adaptation_notes:
                print(f"\n📝 Adaptation Notes:")
                for note in adapted.adaptation_notes:
                    print(f"   • {note}")
        
        print(f"\n🎯 Custom Deployment Instructions:")
        for instruction in result.deployment_instructions:
            print(f"   • {instruction}")
        
    except Exception as e:
        print(f"❌ Custom adaptation failed: {str(e)}")


async def demo_batch_processing():
    """Demonstrate batch processing multiple rosters for different teams"""
    
    print(f"\n" + "=" * 60)
    print("⚡ Batch Processing Demo")  
    print("Processing multiple team rosters simultaneously\n")
    
    expert = FormatAdaptationExpert()
    
    # Create different team rosters
    data_team = create_sample_roster_agents()
    
    # Create development team
    dev_team = [
        SourceAgent(
            name="FullStackDeveloper",
            role="Senior Full Stack Developer",
            description="Expert in both frontend and backend development",
            capabilities=["React", "Node.js", "Python", "PostgreSQL", "AWS"],
            instructions="You are a senior full stack developer. Build scalable web applications using modern technologies.",
            interaction_patterns={
                "input": "Requirements, user stories, technical specifications",
                "output": "Code, technical documentation, deployment guides", 
                "coordination": "Works with designers, product managers, and DevOps"
            }
        ),
        SourceAgent(
            name="DevOpsEngineer",
            role="Senior DevOps Engineer", 
            description="Expert in infrastructure automation and deployment pipelines",
            capabilities=["Docker", "Kubernetes", "CI/CD", "Terraform", "Monitoring"],
            instructions="You are a senior DevOps engineer. Build and maintain scalable cloud infrastructure and deployment pipelines.",
            interaction_patterns={
                "input": "Infrastructure requirements, application deployments",
                "output": "Infrastructure as code, CI/CD pipelines, monitoring setups",
                "coordination": "Works with development and operations teams"
            }
        )
    ]
    
    # Batch process both teams to Claude Code
    print(f"📦 Batch processing teams:")
    print(f"   • Data Team: {len(data_team)} agents")
    print(f"   • Dev Team: {len(dev_team)} agents") 
    print(f"   • Target: Claude Code format")
    
    all_agents = data_team + dev_team
    
    request = AdaptationRequest(
        source_agents=all_agents,
        target_platform="claude-code",
        customizations={
            "batch_processing": True,
            "team_tags": ["data_science", "development"],
            "deployment_group": "company_agents_v1"
        },
        preserve_semantics=True
    )
    
    try:
        result = await expert.adapt_agents(request)
        
        print(f"\n✅ {result.summary}")
        
        # Group results by validation status
        successful = [a for a in result.adapted_agents if a.validation_status == "valid"]
        warnings = [a for a in result.adapted_agents if a.validation_status == "warning"]
        errors = [a for a in result.adapted_agents if a.validation_status == "error"]
        
        print(f"📊 Batch Results:")
        print(f"   ✅ Successful: {len(successful)} agents")
        print(f"   ⚠️  Warnings: {len(warnings)} agents")
        print(f"   ❌ Errors: {len(errors)} agents")
        
        print(f"\n📋 Successfully Adapted Agents:")
        for agent in successful:
            print(f"   • {agent.original_name} → Claude Code format")
        
        if warnings:
            print(f"\n⚠️ Agents with Warnings:")
            for agent in warnings:
                print(f"   • {agent.original_name}: {agent.adaptation_notes[0] if agent.adaptation_notes else 'Unknown warning'}")
        
        print(f"\n🚀 Ready for deployment!")
        print(f"   All {len(successful)} agents can be deployed to Claude Code")
        print(f"   Use provided deployment instructions for setup")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")


async def main():
    """Run all Format Adaptation Expert demos"""
    
    print("🎬 Format Adaptation Expert - Complete Demo Suite")
    print("=" * 80)
    print("Demonstrating AgentForge roster adaptation to multiple platforms\n")
    
    # Run demos
    await demo_multi_platform_adaptation()
    await demo_custom_platform_adaptation() 
    await demo_batch_processing()
    
    print(f"\n" + "=" * 80)
    print("🎉 Format Adaptation Expert demo completed!")
    print("\nKey Capabilities Demonstrated:")
    print("✅ Multi-platform adaptation (Claude Code, OpenCode, Amazon Q)")
    print("✅ Custom platform format support")
    print("✅ Batch processing of multiple agents")
    print("✅ Semantic preservation during adaptation") 
    print("✅ Validation and deployment instructions")
    print("\nThe Format Adaptation Expert enables AgentForge rosters to be")
    print("deployed across any platform with appropriate format conversion!")


if __name__ == "__main__":
    asyncio.run(main())