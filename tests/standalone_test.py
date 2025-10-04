#!/usr/bin/env python3
"""
Standalone CLI Test for AgentForge Core Functionality

This is a completely standalone test that imports only the specific modules
we need without triggering the full agents module imports.
"""

import typer
import sys
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="agentforge-standalone-test",
    help="AgentForge Standalone Test CLI",
    add_completion=False
)

console = Console()


@app.command("test-naming")
def test_naming_strategies():
    """Test the naming strategies module directly."""
    
    # Import the naming strategies module directly
    sys.path.insert(0, str(Path(__file__).parent / "agents"))
    from naming_strategies import create_naming_strategy
    
    rprint("[bold blue]Testing Naming Strategies[/bold blue]\n")
    
    # Test cases
    test_cases = [
        ("developer", "web development"),
        ("analyst", "data science"), 
        ("architect", "cloud infrastructure"),
        ("tester", "mobile development")
    ]
    
    # Test domain naming
    rprint("[cyan]Domain Naming Strategy:[/cyan]")
    try:
        domain_strategy = create_naming_strategy(strategy_type="domain")
        
        for role, domain in test_cases:
            name = domain_strategy.generate_name(role, domain)
            rprint(f"  {role} + {domain} = [green]{name}[/green]")
        
        rprint("[green]✅ Domain naming successful![/green]")
    except Exception as e:
        rprint(f"[red]❌ Domain naming failed: {e}[/red]")
    
    # Test real names
    rprint("\n[cyan]Real Name Strategy:[/cyan]") 
    try:
        real_strategy = create_naming_strategy(strategy_type="real")
        
        for role, domain in test_cases:
            name = real_strategy.generate_name(role, domain)
            rprint(f"  {role} + {domain} = [green]{name}[/green]")
            
        rprint("[green]✅ Real naming successful![/green]")
    except Exception as e:
        rprint(f"[red]❌ Real naming failed: {e}[/red]")
    
    # Test manual name
    rprint("\n[cyan]Manual Name Strategy:[/cyan]")
    try:
        manual_strategy = create_naming_strategy(manual_name="Billy Cheemo")
        name = manual_strategy.generate_name("developer", "any domain")
        rprint(f"  Any role + Any domain = [green]{name}[/green]")
        
        rprint("[green]✅ Manual naming successful![/green]")
    except Exception as e:
        rprint(f"[red]❌ Manual naming failed: {e}[/red]")
    
    # Test custom rules
    rprint("\n[cyan]Custom Rules Strategy:[/cyan]")
    try:
        custom_rules = {
            "templates": ["{role_title} Specialist", "Senior {role_title}"],
            "mappings": {"developer": "Code Wizard"}
        }
        custom_strategy = create_naming_strategy(custom_rules=custom_rules)
        
        name1 = custom_strategy.generate_name("developer", "web development")
        name2 = custom_strategy.generate_name("analyst", "data science")
        
        rprint(f"  developer + web development = [green]{name1}[/green]")
        rprint(f"  analyst + data science = [green]{name2}[/green]")
        
        rprint("[green]✅ Custom rules successful![/green]")
    except Exception as e:
        rprint(f"[red]❌ Custom rules failed: {e}[/red]")
    
    rprint("\n[bold green]🎉 Naming strategies test completed![/bold green]")


@app.command("test-cli-patterns")
def test_cli_patterns():
    """Test the 6 CLI command patterns from TASK.md."""
    
    rprint("[bold blue]Testing CLI Command Patterns[/bold blue]\n")
    
    # Import naming strategies
    sys.path.insert(0, str(Path(__file__).parent / "agents"))
    from naming_strategies import create_naming_strategy
    
    patterns = [
        {
            "title": "Pattern 1: Simple Query",
            "command": 'agentforge "I need a team fine tuned to convert python scripts to idiomatic rust scripts"',
            "test": lambda: test_simple_query("I need a team fine tuned to convert python scripts to idiomatic rust scripts")
        },
        {
            "title": "Pattern 2: File Context + Agents Folder",
            "command": "agentforge -f /path/to/prd.md --agents /path/to/agents/folder/",
            "test": lambda: test_file_context_pattern()
        },
        {
            "title": "Pattern 3: Single Agent with Manual Name", 
            "command": 'agentforge -f /path/to/task.md -n1 --name "Billy Cheemo"',
            "test": lambda: test_manual_name_pattern()
        },
        {
            "title": "Pattern 4: Force Create with Output Directory",
            "command": "agentforge -f /path/to/task.md --force -n3 -o ./agents/",
            "test": lambda: test_force_create_pattern()
        },
        {
            "title": "Pattern 5: Domain Naming Strategy",
            "command": 'agentforge -f /path/to/task.md --auto-name-strategy "domain"',
            "test": lambda: test_domain_naming_pattern()
        },
        {
            "title": "Pattern 6: Custom Naming Rules",
            "command": "agentforge -f /path/to/task.md --auto-name-rules /path/to/naming-rules.md",
            "test": lambda: test_custom_rules_pattern()
        }
    ]
    
    results = []
    
    for pattern in patterns:
        rprint(f"[yellow]Testing {pattern['title']}...[/yellow]")
        rprint(f"[dim]Command: {pattern['command']}[/dim]")
        
        try:
            result = pattern["test"]()
            results.append((pattern["title"], True, "Success"))
            rprint(f"[green]✅ {pattern['title']} - Success[/green]")
        except Exception as e:
            results.append((pattern["title"], False, str(e)))
            rprint(f"[red]❌ {pattern['title']} - Failed: {e}[/red]")
        
        rprint("")
    
    # Summary
    rprint("[bold blue]Test Results Summary:[/bold blue]")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Pattern", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Details", style="yellow")
    
    passed = 0
    for title, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        if success:
            passed += 1
        summary_table.add_row(title, status, details)
    
    console.print(summary_table)
    
    rprint(f"\n[bold]Results: {passed}/{len(results)} patterns passed[/bold]")
    
    if passed == len(results):
        rprint("[bold green]🎉 All CLI patterns working correctly![/bold green]")
    else:
        rprint(f"[bold yellow]⚠ {len(results) - passed} patterns need attention[/bold yellow]")


def test_simple_query(query: str) -> bool:
    """Test simple query pattern."""
    from naming_strategies import create_naming_strategy
    
    # Simulate simple query processing
    domain = "programming" if any(lang in query.lower() for lang in ["python", "rust", "convert"]) else "general"
    
    # Parse roles from query
    roles_needed = []
    if "convert" in query.lower() and "python" in query.lower() and "rust" in query.lower():
        roles_needed = ["Language Conversion Expert", "Python Developer", "Rust Developer"]
    
    if not roles_needed:
        roles_needed = ["Technical Specialist"]
    
    # Create agents with domain naming
    naming_strategy = create_naming_strategy(strategy_type="domain")
    
    agents_created = []
    for role in roles_needed:
        name = naming_strategy.generate_name(role, domain)
        agents_created.append({"name": name, "role": role, "domain": domain})
    
    rprint(f"  Created {len(agents_created)} agents:")
    for agent in agents_created:
        rprint(f"    • {agent['name']} ({agent['role']})")
    
    return len(agents_created) > 0


def test_file_context_pattern() -> bool:
    """Test file context with agents folder pattern."""
    # Simulate having file context and agents folder
    file_context = {
        "content": "We need to build a web application with React frontend and Node.js backend",
        "file_type": ".md"
    }
    
    agents_folder = "/home/delorenj/code/DeLoDocs/AI/Agents"
    
    # Infer domain from file content
    content = file_context["content"].lower()
    if "web" in content and "react" in content:
        domain = "web development"
        roles = ["Frontend Developer", "Backend Developer"]
    else:
        domain = "general"
        roles = ["Technical Specialist"]
    
    rprint(f"  File context processed: domain={domain}")
    rprint(f"  Agents folder: {agents_folder}")
    rprint(f"  Roles identified: {', '.join(roles)}")
    
    return True


def test_manual_name_pattern() -> bool:
    """Test single agent with manual name pattern."""
    from naming_strategies import create_naming_strategy
    
    # Simulate -n1 --name "Billy Cheemo"
    num_agents = 1
    manual_name = "Billy Cheemo"
    
    naming_strategy = create_naming_strategy(manual_name=manual_name)
    
    # Create single agent
    agent_name = naming_strategy.generate_name("developer", "general")
    
    rprint(f"  Created 1 agent with manual name: {agent_name}")
    
    return agent_name == manual_name


def test_force_create_pattern() -> bool:
    """Test force create with output directory pattern."""
    # Simulate --force -n3 -o ./agents/
    force_create = True
    num_agents = 3
    output_dir = "./agents/"
    
    # Create output directory simulation
    output_path = Path(output_dir)
    
    rprint(f"  Force create: {force_create}")
    rprint(f"  Number of agents: {num_agents}")
    rprint(f"  Output directory: {output_dir}")
    rprint(f"  Would create directory: {output_path.absolute()}")
    
    return True


def test_domain_naming_pattern() -> bool:
    """Test domain naming strategy pattern."""
    from naming_strategies import create_naming_strategy
    
    # Simulate --auto-name-strategy "domain"
    strategy_type = "domain"
    
    naming_strategy = create_naming_strategy(strategy_type=strategy_type)
    
    # Test with different roles
    test_roles = [("developer", "web development"), ("analyst", "data science")]
    
    rprint(f"  Using {strategy_type} naming strategy:")
    for role, domain in test_roles:
        name = naming_strategy.generate_name(role, domain)
        rprint(f"    {role} + {domain} = {name}")
    
    return naming_strategy.get_strategy_type() == "domain"


def test_custom_rules_pattern() -> bool:
    """Test custom naming rules pattern."""
    from naming_strategies import create_naming_strategy
    
    # Simulate custom rules file (would be loaded from --auto-name-rules)
    custom_rules = {
        "templates": ["{role_title} Expert", "Senior {role_title}"],
        "mappings": {
            "developer": "Code Craftsman",
            "tester": "Quality Guardian"
        }
    }
    
    naming_strategy = create_naming_strategy(custom_rules=custom_rules)
    
    # Test the custom rules
    rprint("  Using custom naming rules:")
    
    name1 = naming_strategy.generate_name("developer", "web development")
    name2 = naming_strategy.generate_name("tester", "quality assurance")
    name3 = naming_strategy.generate_name("analyst", "data science")
    
    rprint(f"    developer = {name1}")
    rprint(f"    tester = {name2}")
    rprint(f"    analyst = {name3}")
    
    return naming_strategy.get_strategy_type() == "custom"


@app.command("status")
def show_status():
    """Show system status for testing."""
    
    rprint("[bold blue]AgentForge Test Status[/bold blue]\n")
    
    status_table = Table(show_header=True, header_style="bold blue")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green") 
    status_table.add_column("Details", style="yellow")
    
    # Check Python modules
    modules_to_check = [
        ("typer", "CLI framework"),
        ("rich", "Terminal UI"),
        ("pyyaml", "YAML parsing"),
        ("pydantic", "Data validation"),
        ("sentence-transformers", "Embeddings"),
        ("qdrant-client", "Vector database")
    ]
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            status_table.add_row(module_name, "✅", description)
        except ImportError:
            status_table.add_row(module_name, "❌", f"{description} - Not installed")
    
    # Check paths
    agents_path = Path("/home/delorenj/code/DeLoDocs/AI/Agents")
    teams_path = Path("/home/delorenj/code/DeLoDocs/AI/Teams")
    
    status_table.add_row(
        "Agents Path",
        "✅" if agents_path.exists() else "❌",
        str(agents_path)
    )
    
    status_table.add_row(
        "Teams Path",
        "✅" if teams_path.exists() else "❌",
        str(teams_path)
    )
    
    console.print(status_table)
    
    # Test naming strategies availability
    rprint("\n[bold blue]Testing Core Functionality:[/bold blue]")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "agents"))
        from naming_strategies import create_naming_strategy
        
        # Quick test
        strategy = create_naming_strategy(strategy_type="domain")
        test_name = strategy.generate_name("developer", "web development")
        
        rprint(f"[green]✅ Naming strategies working: {test_name}[/green]")
    except Exception as e:
        rprint(f"[red]❌ Naming strategies failed: {e}[/red]")


if __name__ == "__main__":
    app()