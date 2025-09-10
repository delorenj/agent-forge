"""
AgentForge - A meta-agent system for building specialized agent teams.

This module contains the core agents that make up the AgentForge system:
- Engineering Manager (Orchestrator)
- Systems Analyst  
- Talent Scout
- Agent Developer
- Integration Architect

And supporting utility agents:
- Format Adaptation Expert (Platform Adapter)
- Master Templater (Template Generator)
"""

from .base import AgentForgeBase, AgentForgeInput, AgentForgeOutput
from .systems_analyst import SystemsAnalyst, InputGoal, StrategyDocument
from .agent_developer import AgentDeveloper, VacantRole, ScoutingReport, AgentSpecification, AgentGenerationResult
from .format_adaptation_expert import (
    FormatAdaptationExpert, 
    SourceAgent, 
    PlatformTemplate, 
    AdaptationRequest, 
    AdaptedAgent, 
    AdaptationResult
)
from .master_templater import (
    MasterTemplater,
    SpecificAgent,
    ExtractedComponents,
    GeneralizedAgent,
    TemplateGenerationRequest,
    TemplateGenerationResult
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "AgentForgeBase",
    "AgentForgeInput", 
    "AgentForgeOutput",
    
    # Core agents
    "SystemsAnalyst",
    "InputGoal",
    "StrategyDocument",
    "AgentDeveloper",
    "VacantRole",
    "ScoutingReport", 
    "AgentSpecification",
    "AgentGenerationResult",
    
    # Utility agents
    "FormatAdaptationExpert",
    "SourceAgent",
    "PlatformTemplate", 
    "AdaptationRequest",
    "AdaptedAgent",
    "AdaptationResult",
    "MasterTemplater",
    "SpecificAgent",
    "ExtractedComponents",
    "GeneralizedAgent",
    "TemplateGenerationRequest",
    "TemplateGenerationResult",
]