"""
Test file for Format Adaptation Expert Agent
"""

import asyncio
import json
from agents.format_adaptation_expert import (
    FormatAdaptationExpert, 
    SourceAgent, 
    AdaptationRequest
)


async def test_format_adaptation_expert():
    """Test the Format Adaptation Expert functionality"""
    
    print("🧪 Testing Format Adaptation Expert Agent")
    print("=" * 50)
    
    # Initialize the expert
    expert = FormatAdaptationExpert()
    
    # Create test source agent
    test_agent = SourceAgent(
        name="DataAnalyzer",
        role="Senior Data Analyst",
        description="Expert in analyzing large datasets and generating insights",
        capabilities=[
            "Statistical analysis",
            "Data visualization", 
            "Pattern recognition",
            "Report generation",
            "SQL querying"
        ],
        instructions="""You are a senior data analyst with expertise in statistical analysis and data visualization. 

Your core responsibilities:
- Analyze large datasets to identify patterns and trends
- Create compelling visualizations that tell data stories
- Generate actionable insights from complex data
- Write SQL queries to extract relevant data
- Produce comprehensive analysis reports

Your approach:
1. Understand the business context and questions
2. Explore data to identify quality issues and patterns
3. Apply appropriate statistical methods and visualizations
4. Communicate findings clearly to both technical and non-technical audiences
5. Recommend data-driven actions and strategies

Focus on accuracy, clarity, and actionable insights.""",
        interaction_patterns={
            "input": "Raw datasets, business questions, analysis requirements",
            "output": "Analysis reports, visualizations, insights, recommendations",
            "coordination": "Works with data engineers, business stakeholders, and decision makers"
        },
        metadata={
            "complexity": "high",
            "domain": "data_science",
            "priority": "high"
        }
    )
    
    print(f"📋 Source Agent: {test_agent.name}")
    print(f"   Role: {test_agent.role}")
    print(f"   Capabilities: {len(test_agent.capabilities)} capabilities")
    
    # Test 1: Adapt to Claude Code format
    print(f"\n🔄 Test 1: Adapting to Claude Code format...")
    
    claude_request = AdaptationRequest(
        source_agents=[test_agent],
        target_platform="claude-code",
        preserve_semantics=True
    )
    
    try:
        claude_result = await expert.adapt_agents(claude_request)
        
        print(f"✅ Claude Code adaptation: {claude_result.summary}")
        print(f"   Platform: {claude_result.platform}")
        print(f"   Warnings: {len(claude_result.warnings)}")
        
        if claude_result.adapted_agents:
            adapted = claude_result.adapted_agents[0]
            print(f"   Validation: {adapted.validation_status}")
            print(f"   Format: {adapted.format_type}")
            
            # Show first 300 characters of adapted content
            content_preview = adapted.adapted_content[:300] + "..." if len(adapted.adapted_content) > 300 else adapted.adapted_content
            print(f"\n📄 Adapted Content Preview:")
            print(content_preview)
        
        print(f"\n📋 Deployment Instructions:")
        for i, instruction in enumerate(claude_result.deployment_instructions[:3], 1):
            print(f"   {i}. {instruction}")
        
    except Exception as e:
        print(f"❌ Claude Code adaptation failed: {str(e)}")
    
    # Test 2: Adapt to OpenCode format  
    print(f"\n🔄 Test 2: Adapting to OpenCode format...")
    
    opencode_request = AdaptationRequest(
        source_agents=[test_agent],
        target_platform="opencode",
        preserve_semantics=True,
        customizations={
            "extension_category": "Data Analysis",
            "activation_events": ["onLanguage:sql", "onLanguage:python"]
        }
    )
    
    try:
        opencode_result = await expert.adapt_agents(opencode_request)
        
        print(f"✅ OpenCode adaptation: {opencode_result.summary}")
        print(f"   Platform: {opencode_result.platform}")
        print(f"   Warnings: {len(opencode_result.warnings)}")
        
        if opencode_result.adapted_agents:
            adapted = opencode_result.adapted_agents[0]
            print(f"   Validation: {adapted.validation_status}")
            
            # Show first 300 characters of adapted content
            content_preview = adapted.adapted_content[:300] + "..." if len(adapted.adapted_content) > 300 else adapted.adapted_content
            print(f"\n📄 Adapted Content Preview:")
            print(content_preview)
        
    except Exception as e:
        print(f"❌ OpenCode adaptation failed: {str(e)}")
    
    # Test 3: Quick adaptation
    print(f"\n🔄 Test 3: Quick adaptation test...")
    
    try:
        quick_result = await expert.quick_adapt(
            "CodeReviewer",
            "You are a code reviewer expert. Review code for quality, security, and best practices.",
            "amazonq"
        )
        
        print(f"✅ Quick adaptation successful")
        content_preview = quick_result[:200] + "..." if len(quick_result) > 200 else quick_result
        print(f"📄 Quick Result Preview:")
        print(content_preview)
        
    except Exception as e:
        print(f"❌ Quick adaptation failed: {str(e)}")
    
    # Test 4: Multiple agents adaptation
    print(f"\n🔄 Test 4: Multiple agents adaptation...")
    
    # Create second test agent
    test_agent_2 = SourceAgent(
        name="APIDesigner", 
        role="API Architect",
        description="Expert in designing RESTful APIs and microservices",
        capabilities=[
            "API design",
            "OpenAPI specification",
            "Microservices architecture",
            "Security best practices"
        ],
        instructions="You are an API architect. Design scalable, secure REST APIs following industry best practices.",
        interaction_patterns={
            "input": "Requirements, existing systems",
            "output": "API specifications, architectural diagrams",
            "coordination": "Works with backend developers and product managers"
        }
    )
    
    multi_request = AdaptationRequest(
        source_agents=[test_agent, test_agent_2],
        target_platform="claude-code",
        preserve_semantics=True
    )
    
    try:
        multi_result = await expert.adapt_agents(multi_request)
        
        print(f"✅ Multiple agents adaptation: {multi_result.summary}")
        print(f"   Total agents: {len(multi_result.adapted_agents)}")
        print(f"   Successful: {len([a for a in multi_result.adapted_agents if a.validation_status == 'valid'])}")
        print(f"   Warnings: {len(multi_result.warnings)}")
        
        for adapted in multi_result.adapted_agents:
            print(f"   - {adapted.original_name}: {adapted.validation_status}")
        
    except Exception as e:
        print(f"❌ Multiple agents adaptation failed: {str(e)}")
    
    print(f"\n✅ Format Adaptation Expert tests completed!")


if __name__ == "__main__":
    asyncio.run(test_format_adaptation_expert())