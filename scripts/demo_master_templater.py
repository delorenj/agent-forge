"""
Demo: Master Templater - Template Generator

Demonstrates how the Master Templater can generalize specific agent files
and create template representations for onboarding external agents.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from agents.master_templater import (
    MasterTemplater,
    SpecificAgent,
    TemplateGenerationRequest
)


def create_sample_specific_agents():
    """Create sample specific agents from different platforms for analysis"""
    
    agents = []
    
    # 1. Claude Code Agent (JSON format)
    claude_agent = json.dumps({
        "name": "SQLQueryOptimizer",
        "description": "Optimizes SQL queries for performance and best practices using Claude Code MCP tools",
        "instructions": "You are a SQL query optimization expert for Claude Code. Use the database MCP tools to analyze query execution plans, identify performance bottlenecks, and suggest optimizations. Review SQL for best practices, security issues, and maintainability. Generate optimized queries with explanations of improvements made.",
        "tools": [
            "database-analyzer-mcp",
            "sql-formatter-mcp",
            "query-profiler-mcp"
        ],
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "max_tokens": 3000,
        "response_format": "detailed_analysis",
        "claude_code_specific": {
            "mcp_version": "1.0.0",
            "tool_timeout": 30000,
            "streaming": True
        }
    }, indent=2)
    
    agents.append(SpecificAgent(
        source_platform="claude-code",
        content=claude_agent,
        file_type="json",
        agent_name="SQLQueryOptimizer"
    ))
    
    # 2. VS Code Extension (JSON package.json format)
    vscode_extension = json.dumps({
        "name": "api-documentation-generator",
        "displayName": "API Documentation Generator",
        "description": "Automatically generates comprehensive API documentation from code",
        "version": "1.4.2",
        "publisher": "dev-tools-pro",
        "engines": {
            "vscode": "^1.70.0"
        },
        "categories": [
            "Other",
            "Documentation"
        ],
        "keywords": [
            "api",
            "documentation", 
            "openapi",
            "swagger",
            "rest"
        ],
        "activationEvents": [
            "onLanguage:javascript",
            "onLanguage:typescript",
            "onLanguage:python",
            "onCommand:apiDocs.generate"
        ],
        "main": "./out/extension.js",
        "contributes": {
            "commands": [
                {
                    "command": "apiDocs.generate",
                    "title": "Generate API Documentation"
                },
                {
                    "command": "apiDocs.preview",
                    "title": "Preview API Docs"
                }
            ],
            "configuration": {
                "type": "object",
                "title": "API Docs Generator",
                "properties": {
                    "apiDocs.outputFormat": {
                        "type": "string",
                        "default": "openapi",
                        "enum": ["openapi", "postman", "insomnia"],
                        "description": "Output documentation format"
                    },
                    "apiDocs.includeExamples": {
                        "type": "boolean",
                        "default": true,
                        "description": "Include request/response examples"
                    }
                }
            }
        },
        "scripts": {
            "vscode:prepublish": "npm run compile",
            "compile": "tsc -p ./",
            "watch": "tsc -watch -p ./"
        },
        "devDependencies": {
            "@types/vscode": "^1.70.0",
            "typescript": "^4.7.4"
        }
    }, indent=2)
    
    agents.append(SpecificAgent(
        source_platform="vscode",
        content=vscode_extension,
        file_type="json",
        agent_name="api-documentation-generator"
    ))
    
    # 3. Custom XML-based Agent Platform
    xml_agent = """<?xml version="1.0" encoding="UTF-8"?>
<agent xmlns="http://customplatform.com/agent/v1">
    <metadata>
        <name>SecurityAuditor</name>
        <version>2.1.0</version>
        <category>security</category>
        <author>security-team@company.com</author>
        <created>2024-01-15</created>
        <updated>2024-08-20</updated>
    </metadata>
    
    <description>
        Comprehensive security auditor that analyzes codebases, configurations,
        and infrastructure for security vulnerabilities and compliance issues.
    </description>
    
    <capabilities>
        <capability type="code_analysis">
            <name>Static Code Analysis</name>
            <description>Scans source code for security vulnerabilities</description>
            <languages>java,python,javascript,go,rust</languages>
        </capability>
        <capability type="config_audit">
            <name>Configuration Auditing</name> 
            <description>Reviews system and application configurations</description>
            <formats>yaml,json,xml,ini</formats>
        </capability>
        <capability type="compliance_check">
            <name>Compliance Validation</name>
            <description>Validates against security frameworks</description>
            <frameworks>OWASP,SOX,PCI-DSS,ISO27001</frameworks>
        </capability>
    </capabilities>
    
    <instructions>
        <role>You are a senior security auditor with expertise in application security, infrastructure security, and compliance frameworks.</role>
        
        <approach>
            <step>Analyze the provided code, configurations, or infrastructure specifications</step>
            <step>Identify security vulnerabilities using industry-standard methodologies</step>
            <step>Assess compliance against relevant security frameworks</step>
            <step>Prioritize findings based on risk severity and business impact</step>
            <step>Provide specific remediation recommendations with implementation guidance</step>
            <step>Generate comprehensive audit reports with executive summaries</step>
        </approach>
        
        <output_format>
            <section name="executive_summary">High-level findings and risk assessment</section>
            <section name="detailed_findings">Specific vulnerabilities with evidence</section>
            <section name="remediation_plan">Step-by-step fix recommendations</section>
            <section name="compliance_status">Framework compliance analysis</section>
        </output_format>
    </instructions>
    
    <configuration>
        <parameter name="severity_threshold" type="string" default="medium">Minimum severity level to report</parameter>
        <parameter name="include_best_practices" type="boolean" default="true">Include general security best practices</parameter>
        <parameter name="compliance_frameworks" type="array" default="OWASP,SOX">Active compliance frameworks</parameter>
        <parameter name="report_format" type="string" default="markdown">Output report format</parameter>
    </configuration>
    
    <integrations>
        <integration name="vulnerability_database">
            <type>external_api</type>
            <endpoint>https://nvd.nist.gov/api/v2/</endpoint>
            <description>National Vulnerability Database for CVE information</description>
        </integration>
        <integration name="compliance_checker">
            <type>internal_service</type>
            <endpoint>http://internal-compliance.company.com/api</endpoint>
            <description>Internal compliance validation service</description>
        </integration>
    </integrations>
</agent>"""
    
    agents.append(SpecificAgent(
        source_platform="custom-xml-platform",
        content=xml_agent,
        file_type="xml",
        agent_name="SecurityAuditor"
    ))
    
    # 4. Amazon Alexa Skill (JSON format) 
    alexa_skill = json.dumps({
        "interactionModel": {
            "languageModel": {
                "invocationName": "recipe master",
                "intents": [
                    {
                        "name": "FindRecipeIntent",
                        "slots": [
                            {
                                "name": "Ingredient",
                                "type": "INGREDIENT_TYPE"
                            },
                            {
                                "name": "CuisineType", 
                                "type": "CUISINE_TYPE"
                            },
                            {
                                "name": "DietaryRestriction",
                                "type": "DIETARY_TYPE"
                            }
                        ],
                        "samples": [
                            "find me a recipe with {Ingredient}",
                            "what can I make with {Ingredient}",
                            "show me {CuisineType} recipes",
                            "I need a {DietaryRestriction} recipe"
                        ]
                    },
                    {
                        "name": "GetInstructionsIntent",
                        "slots": [],
                        "samples": [
                            "how do I make this",
                            "give me the instructions",
                            "walk me through the recipe"
                        ]
                    }
                ],
                "types": [
                    {
                        "name": "INGREDIENT_TYPE",
                        "values": [
                            {"name": {"value": "chicken"}},
                            {"name": {"value": "beef"}},
                            {"name": {"value": "vegetables"}},
                            {"name": {"value": "pasta"}}
                        ]
                    },
                    {
                        "name": "CUISINE_TYPE",
                        "values": [
                            {"name": {"value": "italian"}},
                            {"name": {"value": "mexican"}},
                            {"name": {"value": "asian"}},
                            {"name": {"value": "mediterranean"}}
                        ]
                    },
                    {
                        "name": "DIETARY_TYPE", 
                        "values": [
                            {"name": {"value": "vegetarian"}},
                            {"name": {"value": "vegan"}},
                            {"name": {"value": "gluten free"}},
                            {"name": {"value": "keto"}}
                        ]
                    }
                ]
            }
        },
        "manifest": {
            "publishingInformation": {
                "locales": {
                    "en-US": {
                        "summary": "Find and get instructions for recipes based on ingredients and preferences",
                        "examplePhrases": [
                            "Alexa, ask recipe master to find me a chicken recipe",
                            "Alexa, ask recipe master for vegetarian options",
                            "Alexa, ask recipe master how to make this"
                        ],
                        "name": "Recipe Master",
                        "description": "Recipe Master helps you find delicious recipes based on ingredients you have and your dietary preferences. Get step-by-step cooking instructions and discover new dishes."
                    }
                },
                "isAvailableWorldwide": True,
                "testingInstructions": "Test with various ingredient and cuisine combinations",
                "category": "FOOD_AND_DRINK",
                "distributionCountries": ["US", "CA", "GB", "AU"]
            },
            "apis": {
                "custom": {
                    "endpoint": {
                        "sourceDir": "lambda",
                        "uri": "arn:aws:lambda:us-east-1:123456789:function:RecipeMaster"
                    }
                }
            }
        }
    }, indent=2)
    
    agents.append(SpecificAgent(
        source_platform="amazon-alexa",
        content=alexa_skill,
        file_type="json",
        agent_name="RecipeMaster"
    ))
    
    return agents


async def demo_agent_generalization():
    """Demonstrate generalizing specific agents into reusable forms"""
    
    print("🔄 Agent Generalization Demo")
    print("=" * 50)
    print("Converting platform-specific agents into generalized, reusable forms\n")
    
    templater = MasterTemplater()
    specific_agents = create_sample_specific_agents()
    
    print(f"📋 Specific Agents to Generalize ({len(specific_agents)}):")
    for i, agent in enumerate(specific_agents, 1):
        print(f"   {i}. {agent.agent_name} ({agent.source_platform}, {agent.file_type})")
    
    # Generalize each agent individually for detailed analysis
    print(f"\n🔍 Individual Agent Analysis:")
    print("-" * 40)
    
    for agent in specific_agents:
        print(f"\n🎯 Analyzing {agent.agent_name} ({agent.source_platform})...")
        
        request = TemplateGenerationRequest(
            specific_agents=[agent],
            generalize_agents=True,
            create_template=False,
            analysis_depth="comprehensive"
        )
        
        try:
            result = await templater.generate_templates(request)
            
            print(f"✅ {result.analysis_summary}")
            
            if result.generalized_agents:
                generalized = result.generalized_agents[0]
                print(f"   📄 Generalized as: {generalized.name}")
                print(f"   🎭 Role: {generalized.role}")
                print(f"   🔧 Capabilities: {', '.join(generalized.capabilities[:3])}{'...' if len(generalized.capabilities) > 3 else ''}")
                print(f"   📋 Requirements: {', '.join(generalized.requirements)}")
                
                # Show snippet of generalized instructions
                instructions_preview = generalized.instructions[:150] + "..." if len(generalized.instructions) > 150 else generalized.instructions
                print(f"   📖 Instructions: {instructions_preview}")
            
            if result.insights:
                print(f"   💡 Key Insight: {result.insights[0]}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
    
    # Batch generalization
    print(f"\n📦 Batch Generalization of All Agents:")
    print("-" * 40)
    
    batch_request = TemplateGenerationRequest(
        specific_agents=specific_agents,
        generalize_agents=True,
        create_template=False,
        analysis_depth="standard"
    )
    
    try:
        batch_result = await templater.generate_templates(batch_request)
        
        print(f"✅ {batch_result.analysis_summary}")
        print(f"📊 Results: {len(batch_result.generalized_agents)} generalized agents created")
        
        if batch_result.recommendations:
            print(f"\n🎯 Recommendations:")
            for rec in batch_result.recommendations:
                print(f"   • {rec}")
        
        # Show generalized agents summary
        print(f"\n📚 Generalized Agent Library:")
        for generalized in batch_result.generalized_agents:
            print(f"   📁 {generalized.name}")
            print(f"      Role: {generalized.role}")
            print(f"      Capabilities: {len(generalized.capabilities)} identified")
            print(f"      Reusability: High (platform-agnostic)")
        
    except Exception as e:
        print(f"❌ Batch generalization failed: {str(e)}")


async def demo_platform_template_creation():
    """Demonstrate creating platform templates from specific agents"""
    
    print(f"\n" + "=" * 60)
    print("🏗️ Platform Template Creation Demo")
    print("Analyzing agent patterns to create reusable platform templates\n")
    
    templater = MasterTemplater()
    
    # Group agents by platform for template creation
    specific_agents = create_sample_specific_agents()
    
    platforms = {}
    for agent in specific_agents:
        if agent.source_platform not in platforms:
            platforms[agent.source_platform] = []
        platforms[agent.source_platform].append(agent)
    
    print(f"🔍 Platforms identified: {list(platforms.keys())}")
    
    # Create templates for platforms with multiple agents
    for platform, agents in platforms.items():
        print(f"\n📋 Analyzing {platform} platform ({len(agents)} agent{'s' if len(agents) > 1 else ''})...")
        
        if len(agents) == 1:
            print(f"   ⚠️ Single agent - creating basic template analysis")
        else:
            print(f"   ✅ Multiple agents - comprehensive template possible")
        
        request = TemplateGenerationRequest(
            specific_agents=agents,
            target_platform=platform,
            generalize_agents=False,
            create_template=True,
            analysis_depth="comprehensive"
        )
        
        try:
            result = await templater.generate_templates(request)
            
            if result.platform_template:
                template = result.platform_template
                print(f"   🎯 Template created for {template.platform}")
                print(f"      Format: {template.format_type}")
                print(f"      Required fields: {', '.join(template.required_fields)}")
                print(f"      Optional fields: {', '.join(template.optional_fields[:3])}{'...' if len(template.optional_fields) > 3 else ''}")
                print(f"      Conventions: {len(template.conventions)} identified")
                print(f"      Validation rules: {len(template.validation_rules)} defined")
                
                if template.examples:
                    print(f"      Examples: {len(template.examples)} provided")
            
            if result.insights:
                print(f"   💡 Template Insights:")
                for insight in result.insights[:2]:
                    print(f"      • {insight}")
                    
        except Exception as e:
            print(f"❌ Template creation failed for {platform}: {str(e)}")
    
    # Demonstrate creating a cross-platform template
    print(f"\n🌐 Cross-Platform Analysis:")
    print("-" * 30)
    
    cross_platform_request = TemplateGenerationRequest(
        specific_agents=specific_agents,
        target_platform="multi-platform",
        generalize_agents=False,
        create_template=True,
        analysis_depth="comprehensive"
    )
    
    try:
        cross_result = await templater.generate_templates(cross_platform_request)
        
        if cross_result.platform_template:
            template = cross_result.platform_template
            print(f"✅ Cross-platform template created")
            print(f"   Common patterns identified across {len(specific_agents)} agents")
            print(f"   Universal fields: {', '.join(template.required_fields)}")
            print(f"   Platform variations: {', '.join(template.optional_fields[:4])}")
        
        if cross_result.insights:
            print(f"\n🔍 Cross-Platform Insights:")
            for insight in cross_result.insights:
                print(f"   • {insight}")
                
    except Exception as e:
        print(f"❌ Cross-platform analysis failed: {str(e)}")


async def demo_external_agent_onboarding():
    """Demonstrate onboarding external agents using template analysis"""
    
    print(f"\n" + "=" * 60)
    print("📥 External Agent Onboarding Demo")
    print("Using Master Templater to onboard agents from unknown formats\n")
    
    templater = MasterTemplater()
    
    # Simulate discovering agents in unknown formats
    unknown_agents = [
        # TOML-based configuration (new format)
        """
[agent]
name = "LogAnalyzer"
version = "1.0.0"
description = "Analyzes application logs for errors and performance issues"

[agent.capabilities]
log_parsing = true
error_detection = true
performance_metrics = true
alert_generation = true

[agent.instructions]
role = "Senior DevOps Engineer specializing in log analysis"
approach = [
  "Parse log files from various sources",
  "Identify error patterns and anomalies",
  "Generate performance metrics and insights",
  "Create actionable alerts and recommendations"
]

[agent.configuration]
log_formats = ["json", "plaintext", "syslog"]
retention_days = 30
alert_threshold = "high"

[platform]
type = "custom_toml_system"
version = "2.1.0"
deployment = "docker"
        """,
        
        # YAML-based agent (different from standard formats)
        """
---
agent_spec:
  identity:
    name: NetworkSecurityAnalyzer
    role: Cybersecurity Specialist
    expertise_level: expert
  
  mission: |
    Analyze network traffic, detect security threats, and provide
    comprehensive security recommendations for infrastructure protection.
  
  capabilities:
    - name: traffic_analysis
      description: Deep packet inspection and flow analysis
      tools: [wireshark, tcpdump, nmap]
    
    - name: threat_detection
      description: Identify malicious activities and intrusion attempts
      methods: [signature_based, behavioral_analysis, ml_detection]
    
    - name: incident_response
      description: Coordinate response to security incidents
      workflows: [containment, investigation, remediation]
  
  interactions:
    input_sources:
      - network_logs
      - firewall_data
      - ids_alerts
    
    output_formats:
      - security_reports
      - alert_notifications
      - dashboard_metrics
  
  deployment:
    platform: kubernetes
    resources:
      cpu: "2"
      memory: "4Gi"
    
    integrations:
      - splunk
      - elasticsearch
      - pagerduty

meta:
  format_version: "3.0"
  platform: "security_platform_yaml"
  created: "2024-03-15"
        """
    ]
    
    print(f"🔍 Discovered agents in unknown formats:")
    for i, agent_content in enumerate(unknown_agents, 1):
        lines = agent_content.strip().split('\n')
        format_hint = "TOML" if lines[0].startswith('[') else "YAML" if lines[0].startswith('---') else "Unknown"
        print(f"   {i}. Agent in {format_hint} format ({len(lines)} lines)")
    
    # Analyze each unknown agent
    for i, agent_content in enumerate(unknown_agents, 1):
        print(f"\n🔍 Analyzing Unknown Agent #{i}...")
        
        # Detect format
        format_type = "toml" if agent_content.strip().startswith('[') else "yaml"
        platform = f"unknown-{format_type}-platform"
        
        unknown_agent = SpecificAgent(
            source_platform=platform,
            content=agent_content,
            file_type=format_type,
            agent_name=f"UnknownAgent{i}"
        )
        
        # Analyze and create both generalized agent and platform template
        request = TemplateGenerationRequest(
            specific_agents=[unknown_agent],
            target_platform=platform,
            generalize_agents=True,
            create_template=True,
            analysis_depth="comprehensive"
        )
        
        try:
            result = await templater.generate_templates(request)
            
            print(f"✅ Analysis complete: {result.analysis_summary}")
            
            # Show generalized agent
            if result.generalized_agents:
                generalized = result.generalized_agents[0]
                print(f"   🎯 Agent successfully generalized:")
                print(f"      Name: {generalized.name}")
                print(f"      Role: {generalized.role}")
                print(f"      Capabilities: {', '.join(generalized.capabilities[:2])}...")
                print(f"      Status: Ready for AgentForge integration")
            
            # Show platform template
            if result.platform_template:
                template = result.platform_template
                print(f"   🏗️ Platform template created:")
                print(f"      Platform: {template.platform}")
                print(f"      Format: {template.format_type}")
                print(f"      Structure: {len(template.required_fields)} required, {len(template.optional_fields)} optional fields")
                print(f"      Status: Ready for future agents from this platform")
            
            if result.insights:
                print(f"   💡 Key Insights:")
                for insight in result.insights[:2]:
                    print(f"      • {insight}")
            
            print(f"   ✅ Agent successfully onboarded to AgentForge ecosystem!")
            
        except Exception as e:
            print(f"❌ Onboarding failed: {str(e)}")
    
    # Demonstrate using templates for similar agents
    print(f"\n🔄 Template Reuse Demo:")
    print("Using learned templates to onboard similar agents faster...")
    
    # This would use previously learned templates to process new agents faster
    print(f"   📚 Templates now available for:")
    print(f"      • TOML-based agent systems") 
    print(f"      • YAML security platform agents")
    print(f"      • Future agents from these platforms can be processed automatically")


async def demo_quick_operations():
    """Demonstrate quick templating operations for common scenarios"""
    
    print(f"\n" + "=" * 60)
    print("⚡ Quick Operations Demo")
    print("Fast templating for common scenarios\n")
    
    templater = MasterTemplater()
    
    # Quick generalization scenarios
    quick_scenarios = [
        {
            "name": "Simple Agent Config",
            "content": '{"name": "EmailBot", "prompt": "You help users write professional emails", "skills": ["writing", "communication"]}',
            "platform": "generic"
        },
        {
            "name": "Chatbot Definition",
            "content": """
            name: SupportBot
            description: Customer support chatbot
            intents:
              - greeting
              - help_request
              - complaint_handling
            responses:
              greeting: "Hello! How can I help you today?"
              help: "I'm here to assist with your questions."
            """,
            "platform": "chatbot-yaml"
        },
        {
            "name": "Skill Definition",
            "content": """
            <skill name="WeatherAssistant">
                <description>Provides weather information and forecasts</description>
                <intents>
                    <intent name="GetWeather" samples="what's the weather,weather forecast"/>
                    <intent name="GetTemperature" samples="how hot is it,current temperature"/>
                </intents>
                <responses>
                    <response intent="GetWeather">I'll get the weather information for you.</response>
                    <response intent="GetTemperature">Let me check the current temperature.</response>
                </responses>
            </skill>
            """,
            "platform": "xml-skills"
        }
    ]
    
    print(f"🚀 Quick Generalization Tests:")
    for i, scenario in enumerate(quick_scenarios, 1):
        print(f"\n{i}. {scenario['name']} ({scenario['platform']})...")
        
        try:
            result = await templater.quick_generalize(
                scenario["content"], 
                scenario["platform"]
            )
            
            print(f"   ✅ Generalized successfully")
            preview = result[:120] + "..." if len(result) > 120 else result
            print(f"   📄 Result: {preview}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    # Quick platform analysis
    print(f"\n🔍 Quick Platform Analysis:")
    
    sample_agents_by_platform = {
        "custom-bot-platform": [
            '{"botName": "Helper", "personality": "friendly", "skills": ["help", "support"]}',
            '{"botName": "Expert", "personality": "professional", "skills": ["analysis", "advice"]}'
        ],
        "micro-service-agents": [
            '{"service": "UserManager", "endpoints": ["/users", "/profiles"], "auth": "jwt"}',
            '{"service": "OrderProcessor", "endpoints": ["/orders", "/payments"], "auth": "oauth"}'
        ]
    }
    
    for platform, agents in sample_agents_by_platform.items():
        print(f"\n📋 Analyzing {platform}...")
        
        try:
            template = await templater.analyze_platform_format(agents, platform)
            
            print(f"   ✅ Platform template created")
            print(f"      Format: {template.format_type}")
            print(f"      Required: {', '.join(template.required_fields)}")
            print(f"      Examples: {len(template.examples)} provided")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {str(e)}")


async def main():
    """Run all Master Templater demos"""
    
    print("🎬 Master Templater - Complete Demo Suite")
    print("=" * 80)
    print("Demonstrating agent generalization and template creation capabilities\n")
    
    # Run all demos
    await demo_agent_generalization()
    await demo_platform_template_creation()
    await demo_external_agent_onboarding()
    await demo_quick_operations()
    
    print(f"\n" + "=" * 80)
    print("🎉 Master Templater demo completed!")
    print("\nKey Capabilities Demonstrated:")
    print("✅ Agent generalization from specific to reusable forms")
    print("✅ Platform template creation from agent analysis")
    print("✅ External agent onboarding from unknown formats")
    print("✅ Quick operations for common scenarios")
    print("✅ Cross-platform pattern recognition")
    print("✅ Template reuse for efficient processing")
    print("\nThe Master Templater enables AgentForge to:")
    print("• Onboard agents from any platform or format")
    print("• Create reusable templates for consistent processing")
    print("• Generalize specific agents for cross-platform reuse")
    print("• Analyze and understand new platform formats automatically")


if __name__ == "__main__":
    asyncio.run(main())