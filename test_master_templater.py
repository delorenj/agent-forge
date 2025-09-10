"""
Test file for Master Templater Agent
"""

import asyncio
import json
from agents.master_templater import (
    MasterTemplater,
    SpecificAgent, 
    TemplateGenerationRequest
)


async def test_master_templater():
    """Test the Master Templater functionality"""
    
    print("🧪 Testing Master Templater Agent")
    print("=" * 50)
    
    # Initialize the templater
    templater = MasterTemplater()
    
    # Test 1: Generalize a Claude Code agent
    print(f"🔄 Test 1: Generalizing Claude Code agent...")
    
    claude_agent_content = json.dumps({
        "name": "PythonCodeAnalyzer",
        "description": "Analyzes Python code for quality issues using Claude Code tools",
        "instructions": "You are a Python code analyzer for Claude Code. Use the MCP file tools to read Python files, analyze them for PEP8 compliance, security vulnerabilities, and performance issues. Generate detailed reports with specific line numbers and improvement suggestions. Use the git tools to understand code history and context.",
        "tools": [
            "file-reader-mcp",
            "git-tools-mcp", 
            "python-linter-mcp"
        ],
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.2,
        "max_tokens": 4000,
        "claude_code_version": "1.0.0"
    }, indent=2)
    
    claude_specific = SpecificAgent(
        source_platform="claude-code",
        content=claude_agent_content,
        file_type="json",
        agent_name="PythonCodeAnalyzer"
    )
    
    try:
        generalize_request = TemplateGenerationRequest(
            specific_agents=[claude_specific],
            generalize_agents=True,
            create_template=False,
            analysis_depth="comprehensive"
        )
        
        result1 = await templater.generate_templates(generalize_request)
        
        print(f"✅ Generalization: {result1.analysis_summary}")
        print(f"   Insights: {len(result1.insights)}")
        print(f"   Warnings: {len(result1.warnings)}")
        
        if result1.generalized_agents:
            generalized = result1.generalized_agents[0]
            print(f"\n📄 Generalized Agent:")
            print(f"   Name: {generalized.name}")
            print(f"   Role: {generalized.role}")
            print(f"   Capabilities: {generalized.capabilities}")
            print(f"   Requirements: {generalized.requirements}")
            
            # Show first 200 characters of instructions
            instructions_preview = generalized.instructions[:200] + "..." if len(generalized.instructions) > 200 else generalized.instructions
            print(f"   Instructions: {instructions_preview}")
        
        if result1.insights:
            print(f"\n💡 Key Insights:")
            for insight in result1.insights[:3]:
                print(f"   - {insight}")
        
    except Exception as e:
        print(f"❌ Generalization failed: {str(e)}")
    
    # Test 2: Create platform template from multiple agents
    print(f"\n🔄 Test 2: Creating platform template from multiple agents...")
    
    # VS Code extension agent
    vscode_agent_content = json.dumps({
        "name": "data-viz-helper",
        "displayName": "Data Visualization Helper", 
        "description": "Helps create data visualizations in VS Code",
        "version": "1.2.3",
        "publisher": "data-tools",
        "engines": {"vscode": "^1.60.0"},
        "categories": ["Data Science", "Visualization"],
        "activationEvents": [
            "onLanguage:python",
            "onLanguage:r", 
            "onCommand:dataViz.createChart"
        ],
        "contributes": {
            "commands": [{
                "command": "dataViz.createChart",
                "title": "Create Data Chart"
            }],
            "configuration": {
                "properties": {
                    "dataViz.defaultChartType": {
                        "type": "string",
                        "default": "bar",
                        "description": "Default chart type to create"
                    }
                }
            }
        }
    }, indent=2)
    
    vscode_specific = SpecificAgent(
        source_platform="vscode",
        content=vscode_agent_content,
        file_type="json",
        agent_name="data-viz-helper"
    )
    
    # Another VS Code agent
    vscode_agent2_content = json.dumps({
        "name": "sql-formatter",
        "displayName": "SQL Formatter Pro",
        "description": "Professional SQL formatting and validation",
        "version": "2.1.0", 
        "publisher": "sql-tools",
        "engines": {"vscode": "^1.65.0"},
        "categories": ["Formatters", "Other"],
        "activationEvents": [
            "onLanguage:sql",
            "onCommand:sqlFormatter.format"
        ],
        "contributes": {
            "commands": [{
                "command": "sqlFormatter.format",
                "title": "Format SQL"
            }, {
                "command": "sqlFormatter.validate", 
                "title": "Validate SQL"
            }],
            "languages": [{
                "id": "sql",
                "extensions": [".sql"],
                "configuration": "./language-configuration.json"
            }]
        }
    }, indent=2)
    
    vscode_specific2 = SpecificAgent(
        source_platform="vscode",
        content=vscode_agent2_content,
        file_type="json",
        agent_name="sql-formatter"
    )
    
    try:
        template_request = TemplateGenerationRequest(
            specific_agents=[vscode_specific, vscode_specific2],
            target_platform="vscode",
            generalize_agents=True,
            create_template=True,
            analysis_depth="comprehensive"
        )
        
        result2 = await templater.generate_templates(template_request)
        
        print(f"✅ Template creation: {result2.analysis_summary}")
        print(f"   Generalized agents: {len(result2.generalized_agents)}")
        print(f"   Template created: {'Yes' if result2.platform_template else 'No'}")
        
        if result2.platform_template:
            template = result2.platform_template
            print(f"\n📋 Platform Template:")
            print(f"   Platform: {template.platform}")
            print(f"   Format: {template.format_type}")
            print(f"   Required fields: {template.required_fields}")
            print(f"   Optional fields: {template.optional_fields}")
            print(f"   Conventions: {len(template.conventions)} identified")
            
            if template.validation_rules:
                print(f"   Validation rules: {template.validation_rules[:2]}")
        
        if result2.recommendations:
            print(f"\n🎯 Recommendations:")
            for rec in result2.recommendations[:3]:
                print(f"   - {rec}")
        
    except Exception as e:
        print(f"❌ Template creation failed: {str(e)}")
    
    # Test 3: Quick generalization
    print(f"\n🔄 Test 3: Quick generalization test...")
    
    try:
        simple_agent = """
        {
            "agent_name": "TaskManager",
            "role": "Project Manager", 
            "prompt": "You are a project management AI assistant. Help users organize tasks, set priorities, track deadlines, and manage team workflows. Use Gantt charts and kanban boards when appropriate.",
            "capabilities": ["task_planning", "deadline_tracking", "team_coordination"],
            "platform": "generic"
        }
        """
        
        quick_result = await templater.quick_generalize(simple_agent, "generic")
        
        print(f"✅ Quick generalization successful")
        preview = quick_result[:250] + "..." if len(quick_result) > 250 else quick_result
        print(f"📄 Generalized Result Preview:")
        print(preview)
        
    except Exception as e:
        print(f"❌ Quick generalization failed: {str(e)}")
    
    # Test 4: Analyze platform format
    print(f"\n🔄 Test 4: Analyzing new platform format...")
    
    custom_platform_agents = [
        """
        <agent>
            <name>DocumentProcessor</name>
            <type>document_analysis</type>
            <capabilities>
                <capability>pdf_parsing</capability>
                <capability>text_extraction</capability>
                <capability>metadata_analysis</capability>
            </capabilities>
            <instructions>
                Process documents and extract structured information.
                Support multiple formats including PDF, DOC, and TXT.
            </instructions>
        </agent>
        """,
        """
        <agent>
            <name>ImageAnalyzer</name>
            <type>computer_vision</type>
            <capabilities>
                <capability>object_detection</capability>
                <capability>text_recognition</capability>
                <capability>face_detection</capability>
            </capabilities>
            <instructions>
                Analyze images and provide detailed descriptions.
                Identify objects, text, and people in images.
            </instructions>
        </agent>
        """
    ]
    
    try:
        platform_template = await templater.analyze_platform_format(
            custom_platform_agents, 
            "custom-xml-platform"
        )
        
        print(f"✅ Platform analysis successful")
        print(f"📋 New Platform Template:")
        print(f"   Platform: {platform_template.platform}")
        print(f"   Format: {platform_template.format_type}")
        print(f"   Required fields: {platform_template.required_fields}")
        print(f"   Examples: {len(platform_template.examples)} provided")
        
    except Exception as e:
        print(f"❌ Platform analysis failed: {str(e)}")
    
    print(f"\n✅ Master Templater tests completed!")


if __name__ == "__main__":
    asyncio.run(test_master_templater())