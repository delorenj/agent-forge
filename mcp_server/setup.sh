#!/bin/bash
# Setup script for AgentForge MCP Server

set -e # Exit on error

echo "🚀 AgentForge MCP Server Setup"
echo "================================"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
	echo "❌ Python 3.12+ required. Found: $PYTHON_VERSION"
	exit 1
fi
echo "✅ Python version: $PYTHON_VERSION"

# Check if QDrant is running
echo ""
echo "Checking QDrant vector database..."
if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
	echo "✅ QDrant is running on localhost:6333"
else
	echo "⚠️  QDrant not detected. Starting QDrant with Docker..."
	exit
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
	uv pip install -e . --quiet
	echo "✅ Dependencies installed"
else
	echo "❌ pyproject.toml not found. Run from project root."
	exit 1
fi

# Create log directory
echo ""
echo "Creating log directory..."
mkdir -p mcp_server/logs
echo "✅ Log directory created"

# Test MCP server
echo ""
echo "Testing MCP server configuration..."
python3 -c "from mcp_server.team_server import TEAM_TOOLS; print(f'✅ {len(TEAM_TOOLS)} tools configured')"

claude mcp add agent-forge "uv run team_server"
cho "Claude Code will now have access to:"
echo "  • agentforge_create_team - Full team creation"
echo "  • agentforge_analyze_strategy - Strategic analysis"
echo "  • agentforge_scout_agents - Agent discovery"
echo "  • agentforge_develop_agents - Create new agents"
echo "  • agentforge_quick_agent - Quick agent creation"
echo "  • agentforge_get_workflow_status - Workflow status"
echo "  • agentforge_reindex_libraries - Reindex agents"
echo ""
echo "Try: 'Create a team to build a real-time chat application'"
echo ""
