#!/usr/bin/env python3
"""
Simple CLI Test for AgentForge Core Functionality

This is a simplified CLI test that demonstrates the core functionality
without requiring the full Agno framework.
"""

import typer
import asyncio
import json
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import our standalone modules
from agents.naming_strategies import create_naming_strategy
from agents.talent_scout_enhanced import quick_agent_search, analyze_team_needs

app = typer.Typer(
    name="agentforge-test",
    help="AgentForge Test CLI - Simplified version for testing core functionality",
    add_completion=False
)

console = Console()


@app.command("test-naming")
def test_naming_strategies():
    """Test the naming strategies module."""
    
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
    domain_strategy = create_naming_strategy(strategy_type="domain")
    
    for role, domain in test_cases:
        name = domain_strategy.generate_name(role, domain)
        rprint(f"  {role} + {domain} = [green]{name}[/green]")
    
    # Test real names
    rprint("\n[cyan]Real Name Strategy:[/cyan]") 
    real_strategy = create_naming_strategy(strategy_type="real")
    
    for role, domain in test_cases:
        name = real_strategy.generate_name(role, domain)
        rprint(f"  {role} + {domain} = [green]{name}[/green]")
    
    # Test manual name
    rprint("\n[cyan]Manual Name Strategy:[/cyan]")
    manual_strategy = create_naming_strategy(manual_name="Billy Cheemo")
    name = manual_strategy.generate_name("developer", "any domain")
    rprint(f"  Any role + Any domain = [green]{name}[/green]")
    
    rprint("\n[green]✅ Naming strategies test completed![/green]")


@app.command("test-search")
def test_agent_search(
    query: str = typer.Argument("Python developer with web framework experience", help="Search query"),
    agents_path: Optional[str] = typer.Option(None, "--agents", help="Custom agents path"),
    limit: int = typer.Option(5, "-n", help="Max results")
):
    """Test agent search functionality."""
    
    rprint(f"[bold blue]Testing Agent Search[/bold blue]")
    rprint(f"Query: [cyan]{query}[/cyan]\n")
    
    async def run_search():
        try:
            results = await quick_agent_search(query, agents_path, limit)
            
            if not results:
                rprint("[yellow]No agents found. This might mean:[/yellow]")
                rprint("  • QDrant server is not running")
                rprint("  • Agent libraries are empty or not indexed")
                rprint("  • Search query didn't match any agents")
                return
            
            # Display results
            rprint(f"[green]Found {len(results)} agents:[/green]\n")
            
            for i, agent in enumerate(results, 1):
                agent_panel = Panel(
                    f"[bold]{agent['name']}[/bold]\n"
                    f"Role: {agent['role']}\n"
                    f"Domain: {agent['domain']}\n"
                    f"Score: {agent['score']}\n"
                    f"Capabilities: {', '.join(agent['capabilities'])}\n"
                    f"File: {agent['file_path']}",
                    title=f"Agent {i}",
                    border_style="green"
                )
                console.print(agent_panel)
                
        except Exception as e:
            rprint(f"[red]Search failed: {e}[/red]")
            rprint("[yellow]This might mean QDrant is not running or not accessible[/yellow]")
    
    asyncio.run(run_search())


@app.command("test-analysis") 
def test_team_analysis(
    goal: str = typer.Argument("Build a web application with user authentication", help="Goal description"),
    domain: str = typer.Option("web development", "-d", "--domain", help="Domain"),
    agents_path: Optional[str] = typer.Option(None, "--agents", help="Custom agents path")
):
    """Test team needs analysis."""
    
    rprint(f"[bold blue]Testing Team Analysis[/bold blue]")
    rprint(f"Goal: [cyan]{goal}[/cyan]")
    rprint(f"Domain: [cyan]{domain}[/cyan]\n")
    
    async def run_analysis():
        try:
            analysis = await analyze_team_needs(goal, domain, agents_path)
            
            # Create summary table
            table = Table(title="Analysis Results", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan") 
            table.add_column("Value", style="green")
            
            table.add_row("Total Roles", str(analysis["total_roles"]))
            table.add_row("Filled Roles", str(analysis["filled_roles"]))
            table.add_row("Vacant Roles", str(analysis["vacant_roles"]))
            table.add_row("Coverage", f"{analysis['coverage_percent']}%")
            
            console.print(table)
            
            # Show matches
            matches = analysis["matches"]
            if matches:
                rprint(f"\n[bold green]Existing Agent Matches ({len(matches)}):[/bold green]")
                for match in matches:
                    rprint(f"  • [bold]{match['agent_name']}[/bold] for {match['role']}")
                    rprint(f"    Score: {match['score']}, Confidence: {match['confidence']}")
                    if match['needs_adaptation']:
                        rprint(f"    [yellow]⚠ Adaptation needed[/yellow]")
                    rprint("")
            
            # Show gaps
            gaps = analysis["gaps"]
            if gaps:
                rprint(f"[bold yellow]Roles Needing New Agents ({len(gaps)}):[/bold yellow]")
                for gap in gaps:
                    rprint(f"  • [bold]{gap['role']}[/bold]")
                    rprint(f"    Capabilities: {', '.join(gap['capabilities_needed'])}")
                    rprint(f"    Recommendation: {gap['recommendation']}")
                    rprint("")
                    
        except Exception as e:
            rprint(f"[red]Analysis failed: {e}[/red]")
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
    
    asyncio.run(run_analysis())


@app.command("simple")
def simple_test(
    query: str = typer.Argument("I need a team for web development", help="Simple query"),
    naming: str = typer.Option("domain", "--naming", help="Naming strategy: domain, real, or manual:Name"),
    max_agents: int = typer.Option(3, "-n", help="Max number of agents"),
):
    """Simple end-to-end test combining all functionality."""
    
    rprint(f"[bold blue]AgentForge Simple Test[/bold blue]")
    rprint(f"Query: [cyan]{query}[/cyan]")
    rprint(f"Naming: [cyan]{naming}[/cyan]")
    rprint(f"Max Agents: [cyan]{max_agents}[/cyan]\n")
    
    async def run_simple_test():
        try:
            # Step 1: Analyze team needs
            rprint("[yellow]Step 1: Analyzing team needs...[/yellow]")
            domain = "web development" if "web" in query.lower() else "general"
            analysis = await analyze_team_needs(query, domain)
            
            rprint(f"  Found {analysis['total_roles']} roles needed")
            rprint(f"  {analysis['filled_roles']} can use existing agents")
            rprint(f"  {analysis['vacant_roles']} need new agents")
            
            # Step 2: Generate names for new agents needed
            rprint("\n[yellow]Step 2: Generating agent names...[/yellow]")
            
            # Create naming strategy
            if naming.startswith("manual:"):
                manual_name = naming.split(":", 1)[1]
                naming_strategy = create_naming_strategy(manual_name=manual_name)
            else:
                naming_strategy = create_naming_strategy(strategy_type=naming)
            
            # Generate team
            team = []
            
            # Add existing matches
            for match in analysis["matches"][:max_agents]:
                team.append({
                    "name": match["agent_name"],
                    "role": match["role"],
                    "type": "existing",
                    "confidence": match["confidence"]
                })
            
            # Add new agents for gaps
            remaining_slots = max_agents - len(team)
            for gap in analysis["gaps"][:remaining_slots]:
                agent_name = naming_strategy.generate_name(
                    role=gap["role"],
                    domain=domain,
                    context={"capabilities": gap["capabilities_needed"]}
                )
                
                team.append({
                    "name": agent_name,
                    "role": gap["role"], 
                    "type": "new",
                    "capabilities": gap["capabilities_needed"]
                })
            
            # Step 3: Display results
            rprint("\n[bold green]Generated Team:[/bold green]")
            
            for i, agent in enumerate(team, 1):
                type_icon = "🔄" if agent["type"] == "existing" else "✨"
                type_color = "yellow" if agent["type"] == "existing" else "green"
                
                details = [
                    f"[bold]{agent['name']}[/bold]",
                    f"Role: {agent['role']}",
                    f"Type: [{type_color}]{agent['type'].title()}[/{type_color}]"
                ]
                
                if agent["type"] == "existing":
                    details.append(f"Confidence: {agent.get('confidence', 'medium')}")
                else:
                    capabilities = agent.get("capabilities", [])
                    if capabilities:
                        details.append(f"Capabilities: {', '.join(capabilities[:3])}")
                
                agent_panel = Panel(
                    "\n".join(details),
                    title=f"{type_icon} Agent {i}",
                    border_style="blue" if agent["type"] == "new" else "yellow"
                )
                console.print(agent_panel)
            
            rprint(f"\n[bold green]✅ Generated {len(team)} agents using {naming_strategy.get_strategy_type()} naming![/bold green]")
            
        except Exception as e:
            rprint(f"[red]Test failed: {e}[/red]")
            import traceback
            rprint(f"[dim]{traceback.format_exc()}[/dim]")
    
    asyncio.run(run_simple_test())


@app.command("status")
def show_status():
    """Show system status."""
    
    rprint("[bold blue]AgentForge System Status[/bold blue]\n")
    
    # Check paths
    agents_path = Path("/home/delorenj/code/DeLoDocs/AI/Agents")
    teams_path = Path("/home/delorenj/code/DeLoDocs/AI/Teams")
    
    status_table = Table(show_header=True, header_style="bold blue")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green") 
    status_table.add_column("Details", style="yellow")
    
    status_table.add_row(
        "Default Agents Path",
        "✅" if agents_path.exists() else "❌",
        str(agents_path)
    )
    
    status_table.add_row(
        "Default Teams Path",
        "✅" if teams_path.exists() else "❌", 
        str(teams_path)
    )
    
    # Check QDrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, api_key="touchmyflappyfoldyholds")
        collections = client.get_collections()
        status_table.add_row(
            "QDrant Vector DB",
            "✅",
            f"localhost:6333 ({len(collections.collections)} collections)"
        )
    except Exception as e:
        status_table.add_row(
            "QDrant Vector DB",
            "❌",
            f"Connection failed: {str(e)[:50]}..."
        )
    
    # Check dependencies
    try:
        import sentence_transformers
        status_table.add_row("Sentence Transformers", "✅", "Available")
    except ImportError:
        status_table.add_row("Sentence Transformers", "❌", "Not installed")
    
    console.print(status_table)


if __name__ == "__main__":
    app()