"""
Systems Analyst Agent - The Strategist

Expert in decomposing complex goals into discrete, manageable roles and capabilities.
Defines the IDEAL team structure required to solve problems without regard for existing resources.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from os import getenv
import json
from textwrap import dedent
import uuid


class InputGoal(BaseModel):
    """Structured input goal for analysis"""

    description: str = Field(..., description="The high-level goal description")
    context: Optional[str] = Field(
        None, description="Additional context or constraints"
    )
    success_criteria: Optional[List[str]] = Field(
        None, description="How success will be measured"
    )
    domain: Optional[str] = Field(None, description="Domain/industry context")
    complexity: Optional[str] = Field(None, description="Estimated complexity level")


class AgentRole(BaseModel):
    """Individual agent role specification"""

    name: str = Field(..., description="Role name")
    title: str = Field(..., description="Professional title/designation")
    core_responsibilities: List[str] = Field(
        ..., description="Primary responsibilities"
    )
    required_capabilities: List[str] = Field(
        ..., description="Essential skills and capabilities"
    )
    interaction_patterns: Dict[str, str] = Field(
        ..., description="How this role interacts with others"
    )
    success_metrics: List[str] = Field(
        ..., description="How success is measured for this role"
    )
    priority_level: str = Field(
        ..., description="Critical, High, Medium, or Low priority"
    )


class TeamStructure(BaseModel):
    """Overall team structure and coordination patterns"""

    topology: str = Field(
        ..., description="Team organization pattern (hierarchical, mesh, etc.)"
    )
    coordination_mechanism: str = Field(..., description="How agents coordinate work")
    decision_making_process: str = Field(..., description="How decisions are made")
    communication_protocols: List[str] = Field(
        ..., description="Communication patterns"
    )
    workflow_stages: List[str] = Field(..., description="Sequential workflow stages")


class StrategyDocument(BaseModel):
    """Complete strategy document output"""

    goal_analysis: Dict[str, Any] = Field(..., description="Analysis of the input goal")
    team_composition: List[AgentRole] = Field(..., description="Required agent roles")
    team_structure: TeamStructure = Field(
        ..., description="Team organization and coordination"
    )
    risk_assessment: List[str] = Field(
        ..., description="Potential risks and mitigation strategies"
    )
    resource_requirements: Dict[str, Any] = Field(
        ..., description="Resource needs assessment"
    )
    timeline_estimate: Dict[str, str] = Field(
        ..., description="Estimated time requirements"
    )


from .base import AgentForgeBase, AgentForgeInput, AgentForgeOutput

class SystemsAnalyst(AgentForgeBase):
    """The Systems Analyst agent implementation"""

    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize the Systems Analyst agent"""
        super().__init__("SystemsAnalyst", "The Strategist")
        
        # Initialize working embedder using sentence-transformers
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup QDrant client for knowledge storage
        try:
            self.qdrant_client = QdrantClient(
                host="localhost",
                port=6333,
                api_key=getenv("QDRANT_API_KEY", "touchmyflappyfoldyholds")
            )
            
            # Ensure collection exists
            collection_name = "agno_patterns"
            try:
                self.qdrant_client.get_collection(collection_name)
            except Exception:
                # Create collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
            
            self.collection_name = collection_name
            
        except Exception as e:
            print(f"Warning: Could not connect to QDrant: {e}")
            self.qdrant_client = None
            self.collection_name = None
        
        # Store knowledge base path
        self.knowledge_base_path = knowledge_base_path
        
        # Load local documentation if available
        if knowledge_base_path:
            self._load_knowledge_base(knowledge_base_path)
    
    def _load_knowledge_base(self, knowledge_base_path: str):
        """Load knowledge base from path into QDrant"""
        if not self.qdrant_client:
            return
            
        try:
            import os
            from pathlib import Path
            
            path = Path(knowledge_base_path)
            if path.exists() and path.is_dir():
                for file_path in path.rglob("*.md"):
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if content.strip():
                        # Create embedding
                        embedding = self.embedder.encode([content])[0].tolist()
                        
                        # Store in QDrant
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "content": content,
                                "file_path": str(file_path),
                                "type": "knowledge_base"
                            }
                        )
                        
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=[point]
                        )
                        
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")
    
    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information"""
        if not self.qdrant_client:
            return []
            
        try:
            # Create query embedding
            query_vector = self.embedder.encode([query])[0].tolist()
            
            # Search QDrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            # Convert to format expected by analysis
            results = []
            for result in search_results:
                results.append({
                    "content": result.payload.get("content", ""),
                    "score": result.score,
                    "file_path": result.payload.get("file_path", ""),
                    "type": result.payload.get("type", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Warning: Could not search knowledge base: {e}")
            return []

    async def process(self, input_goal: InputGoal) -> AgentForgeOutput:
        """
        Analyze an input goal and produce a comprehensive strategy document

        Args:
            input_goal: The structured input goal to analyze

        Returns:
            AgentForgeOutput: The output of the agent, containing the strategy document
        """

        # Search knowledge base for relevant patterns
        knowledge_results = self.search_knowledge(
            f"{input_goal.description} {input_goal.domain or ''}"
        )
        
        # Build context from knowledge base
        knowledge_context = ""
        if knowledge_results:
            knowledge_context = "\n**Relevant Knowledge Base Information:**\n"
            for result in knowledge_results[:3]:  # Top 3 results
                knowledge_context += f"- {result['content'][:200]}...\n"

        # Create comprehensive analysis using built-in reasoning
        analysis = self._analyze_goal_systematically(input_goal, knowledge_context)
        
        strategy_document = StrategyDocument(**analysis)

        return AgentForgeOutput(result=strategy_document, status="success")
    
    def _analyze_goal_systematically(self, input_goal: InputGoal, knowledge_context: str = "") -> Dict[str, Any]:
        """Systematic goal analysis using structured reasoning"""
        
        # Goal decomposition
        goal_components = self._decompose_goal(input_goal.description)
        
        # Team composition analysis
        required_roles = self._identify_required_roles(goal_components, input_goal.domain)
        
        # Risk and resource analysis
        risks = self._assess_risks(input_goal.description, input_goal.complexity)
        resources = self._estimate_resources(required_roles, input_goal.complexity)
        
        # Timeline estimation
        timeline = self._estimate_timeline(required_roles, input_goal.complexity)
        
        return {
            "goal_analysis": {
                "primary_goal": input_goal.description,
                "domain": input_goal.domain or 'General',
                "context": input_goal.context or 'Not specified',
                "complexity_level": input_goal.complexity or 'Medium',
                "success_criteria": input_goal.success_criteria,
                "goal_components": goal_components,
                "knowledge_context": knowledge_context,
            },
            "team_composition": required_roles,
            "team_structure": {
                "topology": "Hierarchical with mesh coordination for specialized tasks",
                "coordination_mechanism": "Event-driven with shared memory store",
                "decision_making_process": "Consensus-based with escalation to coordinator",
                "communication_protocols": [
                    "Regular status updates to shared coordination layer",
                    "Real-time collaboration channels for dependent tasks",
                    "Structured handoffs between sequential phases",
                ],
                "workflow_stages": [
                    "Requirements Analysis and Planning",
                    "Architecture and Design",
                    "Implementation and Development",
                    "Testing and Quality Assurance",
                    "Integration and Deployment",
                    "Monitoring and Optimization",
                ],
            },
            "risk_assessment": risks,
            "resource_requirements": resources,
            "timeline_estimate": timeline,
        }
    
    def _decompose_goal(self, description: str) -> List[str]:
        """Decompose goal into component parts"""
        # Simple heuristic-based decomposition
        components = []
        
        keywords = {
            "build": ["System Architecture", "Implementation", "Testing"],
            "create": ["Design", "Development", "Validation"], 
            "develop": ["Analysis", "Implementation", "Integration"],
            "deploy": ["Infrastructure", "Configuration", "Monitoring"],
            "analyze": ["Data Collection", "Processing", "Reporting"],
            "manage": ["Planning", "Coordination", "Oversight"]
        }
        
        desc_lower = description.lower()
        for keyword, comps in keywords.items():
            if keyword in desc_lower:
                components.extend(comps)
        
        # Add general components if none found
        if not components:
            components = ["Requirements Analysis", "Design", "Implementation", "Testing"]
        
        return list(set(components))  # Remove duplicates
    
    def _identify_required_roles(self, components: List[str], domain: Optional[str]) -> List[Dict[str, Any]]:
        """Identify required agent roles based on components and domain"""
        
        base_roles = [
            {
                "name": "Systems Architect",
                "title": "Systems Architect",
                "core_responsibilities": ["System design", "Technical architecture", "Integration planning"],
                "required_capabilities": ["System design", "Architecture patterns", "Technology selection"],
                "interaction_patterns": {},
                "success_metrics": [],
                "priority_level": "Critical"
            },
            {
                "name": "Implementation Specialist", 
                "title": "Implementation Specialist",
                "core_responsibilities": ["Core development", "Feature implementation", "Code quality"],
                "required_capabilities": ["Programming", "Development frameworks", "Code optimization"],
                "interaction_patterns": {},
                "success_metrics": [],
                "priority_level": "High"
            },
            {
                "name": "Quality Assurance Engineer",
                "title": "Quality Assurance Engineer",
                "core_responsibilities": ["Testing strategy", "Quality validation", "Performance testing"],
                "required_capabilities": ["Test automation", "Quality standards", "Performance analysis"],
                "interaction_patterns": {},
                "success_metrics": [],
                "priority_level": "High"
            }
        ]
        
        # Add domain-specific roles
        if domain:
            domain_lower = domain.lower()
            if "web" in domain_lower or "frontend" in domain_lower:
                base_roles.append({
                    "name": "Frontend Specialist",
                    "title": "Frontend Specialist",
                    "core_responsibilities": ["UI/UX implementation", "Client-side optimization"],
                    "required_capabilities": ["Frontend frameworks", "UI/UX design", "Browser optimization"],
                    "interaction_patterns": {},
                    "success_metrics": [],
                    "priority_level": "High"
                })
            
            if "data" in domain_lower or "analytics" in domain_lower:
                base_roles.append({
                    "name": "Data Engineer",
                    "title": "Data Engineer",
                    "core_responsibilities": ["Data pipeline design", "Analytics implementation"],
                    "required_capabilities": ["Data processing", "Analytics tools", "Database design"],
                    "interaction_patterns": {},
                    "success_metrics": [],
                    "priority_level": "High"
                })
                
            if "devops" in domain_lower or "infrastructure" in domain_lower:
                base_roles.append({
                    "name": "DevOps Engineer",
                    "title": "DevOps Engineer",
                    "core_responsibilities": ["Infrastructure automation", "Deployment pipelines"],
                    "required_capabilities": ["Cloud platforms", "CI/CD", "Infrastructure as code"],
                    "interaction_patterns": {},
                    "success_metrics": [],
                    "priority_level": "Medium"
                })
        
        return base_roles
    
    def _assess_risks(self, description: str, complexity: Optional[str]) -> List[str]:
        """Assess risks based on goal description and complexity"""
        
        risks = []
        
        # Complexity-based risks
        if complexity and complexity.lower() in ["high", "complex"]:
            risks.extend([
                "Technical complexity may exceed team capabilities. Mitigation: Ensure senior technical leadership and expert consultation",
                "Integration challenges between complex components. Mitigation: Design modular architecture with clear interfaces"
            ])
        
        # Description-based risks
        desc_lower = description.lower()
        if "new" in desc_lower or "innovative" in desc_lower:
            risks.append("Unproven technology or approach risks. Mitigation: Build prototypes early and validate assumptions")
        
        if "integration" in desc_lower or "connect" in desc_lower:
            risks.append("Integration complexity and compatibility issues. Mitigation: Design clear APIs and conduct integration testing early")
        
        # Default risks if none identified
        if not risks:
            risks = [
                "Resource availability and timeline constraints. Mitigation: Plan for contingencies and maintain flexible scope",
                "Requirements changes during implementation. Mitigation: Use iterative development with regular stakeholder reviews"
            ]
        
        return risks
    
    def _estimate_resources(self, roles: List[Dict[str, Any]], complexity: Optional[str]) -> Dict[str, List[str]]:
        """Estimate resource requirements"""
        
        complexity_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.5,
            "complex": 2.0
        }.get(complexity.lower() if complexity else "medium", 1.0)
        
        base_team_size = len(roles)
        estimated_team_size = int(base_team_size * complexity_multiplier)
        
        return {
            "team_size": [f"{estimated_team_size} agents"],
            "infrastructure": ["Development environment", "Testing infrastructure", "Deployment platform"],
            "tools": ["Project management tools", "Communication platforms", "Quality assurance tools"],
            "external": ["Subject matter experts", "Stakeholder access", "External integrations"]
        }
    
    def _estimate_timeline(self, roles: List[Dict[str, Any]], complexity: Optional[str]) -> Dict[str, str]:
        """Estimate project timeline"""
        
        base_duration_weeks = max(4, len(roles) * 2)  # Minimum 4 weeks, scale with team size
        
        complexity_multiplier = {
            "low": 0.75,
            "medium": 1.0,
            "high": 1.5,
            "complex": 2.0
        }.get(complexity.lower() if complexity else "medium", 1.0)
        
        total_weeks = int(base_duration_weeks * complexity_multiplier)
        
        phases = [
            {"name": "Planning & Analysis", "duration": f"{max(1, total_weeks // 6)} weeks"},
            {"name": "Architecture & Design", "duration": f"{max(2, total_weeks // 4)} weeks"},
            {"name": "Implementation", "duration": f"{max(3, total_weeks // 2)} weeks"},
            {"name": "Testing & Integration", "duration": f"{max(2, total_weeks // 4)} weeks"},
            {"name": "Deployment & Optimization", "duration": f"{max(1, total_weeks // 8)} weeks"}
        ]

        return {
            "total_duration": f"{total_weeks} weeks",
            "phases": "\n".join(f"- {phase['name']}: {phase['duration']}" for phase in phases)
        }
    
    # Formatting helper methods
    def _format_criteria(self, criteria: Optional[List[str]]) -> str:
        if not criteria:
            return "- Not specified"
        return "\n".join(f"- {criterion}" for criterion in criteria)
    
    def _format_components(self, components: List[str]) -> str:
        return "\n".join(f"- {comp}" for comp in components)
    
    def _format_roles(self, roles: List[Dict[str, Any]]) -> str:
        formatted_roles = []
        for role in roles:
            role_text = f"**{role['name']}** ({role['priority']} Priority)\n"
            role_text += f"  - Responsibilities: {', '.join(role['responsibilities'])}\n"
            role_text += f"  - Required Capabilities: {', '.join(role['capabilities'])}\n"
            formatted_roles.append(role_text)
        return "\n".join(formatted_roles)
    
    def _format_risks(self, risks: List[Dict[str, str]]) -> str:
        formatted_risks = []
        for risk in risks:
            formatted_risks.append(f"- **Risk:** {risk['risk']}")
            formatted_risks.append(f"  **Mitigation:** {risk['mitigation']}")
        return "\n".join(formatted_risks)
    
    def _format_resources(self, resources: Dict[str, List[str]]) -> str:
        formatted_resources = []
        for category, items in resources.items():
            formatted_resources.append(f"**{category.title()}:**")
            formatted_resources.extend(f"- {item}" for item in items)
        return "\n".join(formatted_resources)
    
    def _format_timeline(self, phases: List[Dict[str, str]]) -> str:
        return "\n".join(f"- {phase['name']}: {phase['duration']}" for phase in phases)

    def create_strategy_document(
        self, analysis_result: str, output_path: str = "agent-strategy.md"
    ) -> str:
        """
        Create a formatted strategy document from analysis results

        Args:
            analysis_result: The analysis output from the agent
            output_path: Path where to save the strategy document

        Returns:
            str: Path to the created document
        """

        # Create formatted strategy document
        from datetime import datetime

        current_time = datetime.now().isoformat()

        strategy_content = dedent(
            f"""\
            # Agent Strategy Document
            
            **Generated by:** Systems Analyst (AgentForge)
            **Date:** {current_time}
            **Document Type:** Strategy Document
            
            ---
            
            {analysis_result}
            
            ---
            
            **Note:** This strategy document defines the IDEAL team structure required to achieve the goal.
            The next step is for the Talent Scout to match these roles against existing agent resources.
        """
        )

        # Write the document
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(strategy_content)

        return output_path

    async def quick_analysis(self, goal_description: str) -> str:
        """
        Quick analysis for simple goal descriptions

        Args:
            goal_description: Simple text description of the goal

        Returns:
            str: Analysis result
        """
        input_goal = InputGoal(description=goal_description)
        return await self.analyze_goal(input_goal)


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_systems_analyst():
        """Test the Systems Analyst implementation"""

        analyst = SystemsAnalyst()

        # Test goal
        test_goal = InputGoal(
            description="Build a comprehensive customer support system with AI chatbots, human escalation, and knowledge management",
            context="For a mid-size SaaS company with 10,000+ customers",
            success_criteria=[
                "Reduce response time to under 2 minutes",
                "Handle 80% of queries automatically",
                "Maintain 95% customer satisfaction",
                "Integrate with existing CRM and ticketing systems",
            ],
            domain="Customer Support / SaaS",
            complexity="High",
        )

        # Analyze the goal
        print("🔍 Analyzing goal...")
        result = await analyst.analyze_goal(test_goal)

        print("\n📋 Strategy Document:")
        print("=" * 50)
        print(result)

        # Create strategy document
        doc_path = analyst.create_strategy_document(result)
        print(f"\n✅ Strategy document created: {doc_path}")

    # Run the test
    asyncio.run(test_systems_analyst())
