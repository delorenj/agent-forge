"""
Enhanced Talent Scout with QDrant Vector Database Integration

The Talent Scout specializes in semantic agent discovery, matching, and intelligent reuse
through advanced vector similarity search and capability analysis.
"""

import asyncio
import json
import os
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import re

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    CreateCollection,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    SearchRequest,
    SearchParams,
)
from sentence_transformers import SentenceTransformer

# Agno imports temporarily commented out due to circular import issues
# from agno.agent import Agent
# from agno.models.openrouter import OpenRouter
# from agno.db.sqlite import SqliteDb
# from agno.vectordb.QDrant import QDrantDb
# from agno.knowledge import text

# Direct imports as workaround
import qdrant_client

from .base import AgentForgeBase, AgentForgeInput, AgentForgeOutput


# === Core Data Models ===


class AgentMetadata(BaseModel):
    """Comprehensive metadata for an agent in the library."""

    id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Agent display name")
    file_path: str = Field(description="Full file path to agent")
    role: str = Field(description="Primary agent role/function")
    description: str = Field(description="Detailed agent description")
    capabilities: List[str] = Field(description="List of agent capabilities")
    tools: List[str] = Field(description="Available tools/integrations")
    domain: str = Field(description="Primary domain (e.g., 'web development')")
    complexity_level: str = Field(description="Complexity level (low/medium/high)")
    tags: List[str] = Field(description="Searchable tags")
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    file_hash: str = Field(description="Hash for change detection")
    embedding_vector: Optional[List[float]] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class RoleRequirement(BaseModel):
    """A role requirement from the strategy document."""

    role_id: str = Field(description="Unique identifier for this role")
    role_name: str = Field(description="Display name of the role")
    description: str = Field(description="Detailed role description")
    required_capabilities: List[str] = Field(description="Must-have capabilities")
    preferred_capabilities: List[str] = Field(
        default_factory=list, description="Nice-to-have capabilities"
    )
    domain: str = Field(description="Role domain")
    complexity_level: str = Field(description="Required complexity handling")
    priority: str = Field(
        default="medium", description="Role priority (low/medium/high/critical)"
    )


class AgentMatch(BaseModel):
    """A potential match between an existing agent and a role requirement."""

    agent: AgentMetadata = Field(description="The matched agent")
    role_requirement: RoleRequirement = Field(description="The role being filled")
    similarity_score: float = Field(description="Semantic similarity score (0-1)")
    capability_match_score: float = Field(description="Capability matching score (0-1)")
    overall_score: float = Field(description="Combined matching score (0-1)")
    match_confidence: str = Field(description="Confidence level (low/medium/high)")
    match_reasoning: str = Field(description="Why this is a good match")
    adaptation_needed: bool = Field(
        default=False, description="Whether adaptation is needed"
    )
    adaptation_suggestions: List[str] = Field(
        default_factory=list, description="How to adapt the agent"
    )


class VacantRole(BaseModel):
    """A role that needs a new agent to be created."""

    role_requirement: RoleRequirement = Field(description="The unfilled role")
    gap_analysis: str = Field(description="Analysis of why no suitable agent exists")
    closest_matches: List[AgentMatch] = Field(
        description="Closest existing agents (for inspiration)"
    )
    creation_recommendations: List[str] = Field(
        description="Recommendations for creating new agent"
    )


class ScoutingReport(BaseModel):
    """Complete analysis of agent matching for a strategy document."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_title: str = Field(description="Title of the strategy document")
    total_roles: int = Field(description="Total number of roles required")
    filled_roles: int = Field(description="Number of roles with suitable matches")
    vacant_roles: int = Field(description="Number of roles needing new agents")

    # Matched agents
    matches: List[AgentMatch] = Field(description="Successful agent-to-role matches")

    # Roles needing new agents
    vacant_positions: List[VacantRole] = Field(description="Roles requiring new agents")

    # Analytics
    overall_coverage: float = Field(description="Percentage of roles covered (0-1)")
    reuse_efficiency: float = Field(
        description="Percentage of roles filled by existing agents"
    )

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    scout_version: str = Field(default="2.0.0")
    processing_time_ms: float = Field(description="Time taken to generate report")


# === Strategy Document Input ===


class StrategyDocument(BaseModel):
    """Input from the Systems Analyst containing role requirements."""

    title: str = Field(description="Strategy document title")
    goal_description: str = Field(description="Original goal being addressed")
    domain: str = Field(description="Primary domain")
    complexity_level: str = Field(description="Overall complexity level")
    roles: List[RoleRequirement] = Field(
        description="Required roles and their specifications"
    )
    timeline: Optional[str] = Field(default=None, description="Project timeline")
    constraints: List[str] = Field(default_factory=list, description="Any constraints")


class TalentScoutInput(AgentForgeInput):
    """Input for the Talent Scout agent."""

    strategy_document: StrategyDocument = Field(
        description="Strategy document to analyze"
    )
    agent_libraries: List[str] = Field(
        default_factory=lambda: [
            "/home/delorenj/code/DeLoDocs/AI/Agents",
            "/home/delorenj/code/DeLoDocs/AI/Teams",
        ],
        description="Paths to agent libraries to search",
    )
    force_reindex: bool = Field(
        default=False, description="Force re-indexing of agents"
    )


class TalentScoutOutput(AgentForgeOutput):
    """Output from the Talent Scout agent."""

    scouting_report: ScoutingReport = Field(description="Complete scouting analysis")
    indexing_stats: Dict[str, Any] = Field(
        description="Statistics about agent indexing"
    )
    performance_metrics: Dict[str, Any] = Field(description="Performance metrics")


# === QDrant Vector Database Manager ===


class QDrantManager:
    """Manages QDrant vector database operations for agent embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = "touchmyflappyfoldyholds",
        collection_name: str = "agent_embeddings",
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name

        # Initialize clients
        self.client = QdrantClient(host=host, port=port, api_key=api_key, https=False)
        self.async_client = AsyncQdrantClient(host=host, port=port, api_key=api_key, https=False)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Fast, good quality
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def initialize_collection(self) -> bool:
        """Initialize the QDrant collection for agent embeddings."""
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                await self.async_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, distance=Distance.COSINE
                    ),
                )
                self.logger.info(f"Created QDrant collection: {self.collection_name}")
            else:
                self.logger.info(
                    f"QDrant collection already exists: {self.collection_name}"
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize QDrant collection: {e}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dim

    def create_agent_text(self, agent: AgentMetadata) -> str:
        """Create searchable text from agent metadata."""
        components = [
            f"Role: {agent.role}",
            f"Description: {agent.description}",
            f"Capabilities: {', '.join(agent.capabilities)}",
            f"Tools: {', '.join(agent.tools)}",
            f"Domain: {agent.domain}",
            f"Tags: {', '.join(agent.tags)}",
        ]
        return " | ".join(components)

    async def index_agent(self, agent: AgentMetadata) -> bool:
        """Index a single agent in QDrant."""
        try:
            # Generate embedding
            agent_text = self.create_agent_text(agent)
            embedding = self.generate_embedding(agent_text)

            # Create point
            point = PointStruct(
                id=agent.id,
                vector=embedding,
                payload={
                    "name": agent.name,
                    "file_path": agent.file_path,
                    "role": agent.role,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "tools": agent.tools,
                    "domain": agent.domain,
                    "complexity_level": agent.complexity_level,
                    "tags": agent.tags,
                    "created_at": agent.created_at.isoformat(),
                    "last_modified": agent.last_modified.isoformat(),
                    "file_hash": agent.file_hash,
                    "searchable_text": agent_text,
                },
            )

            # Upsert to QDrant
            await self.async_client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to index agent {agent.id}: {e}")
            return False

    async def search_similar_agents(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[AgentMetadata, float]]:
        """Search for similar agents using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)

            # Build filter if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multi-value filter (OR condition)
                        for v in value:
                            conditions.append(
                                FieldCondition(key=key, match={"value": v})
                            )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match={"value": value})
                        )

                if conditions:
                    search_filter = Filter(should=conditions)

            # Search in QDrant
            search_results = await self.async_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )

            # Convert results to AgentMetadata
            results = []
            for result in search_results:
                try:
                    payload = result.payload
                    agent = AgentMetadata(
                        id=str(result.id),
                        name=payload["name"],
                        file_path=payload["file_path"],
                        role=payload["role"],
                        description=payload["description"],
                        capabilities=payload["capabilities"],
                        tools=payload["tools"],
                        domain=payload["domain"],
                        complexity_level=payload["complexity_level"],
                        tags=payload["tags"],
                        created_at=datetime.fromisoformat(payload["created_at"]),
                        last_modified=datetime.fromisoformat(payload["last_modified"]),
                        file_hash=payload["file_hash"],
                    )
                    results.append((agent, result.score))
                except Exception as e:
                    self.logger.error(f"Failed to parse search result: {e}")
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Failed to search agents: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent collection."""
        try:
            info = await self.async_client.get_collection(self.collection_name)
            return {
                "total_agents": info.points_count,
                "vector_dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.name,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}


# === Agent Library Scanner ===


class AgentLibraryScanner:
    """Scans agent libraries and extracts metadata."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Supported file extensions for agents
        self.agent_extensions = {".md", ".txt", ".py", ".json", ".yaml", ".yml"}

        # Common agent indicator patterns
        self.agent_patterns = [
            "role:",
            "agent:",
            "description:",
            "capabilities:",
            "tools:",
            "system prompt:",
            "instructions:",
            "persona:",
            "behavior:",
        ]

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for change detection."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to hash {file_path}: {e}")
            return ""

    def extract_agent_metadata(self, file_path: Path) -> Optional[AgentMetadata]:
        """Extract agent metadata from a file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Basic agent detection
            content_lower = content.lower()
            if not any(pattern in content_lower for pattern in self.agent_patterns):
                return None

            # Generate unique ID based on file path
            agent_id = hashlib.md5(str(file_path).encode()).hexdigest()

            # Extract basic info
            name = file_path.stem.replace("_", " ").replace("-", " ").title()

            # Try to extract structured info (this is a simplified version)
            # In a real implementation, you'd have more sophisticated parsing
            role = self.extract_field(content, "role") or self.infer_role_from_name(
                name
            )
            description = self.extract_field(content, "description") or content[:200]
            capabilities = self.extract_list_field(content, "capabilities")
            tools = self.extract_list_field(content, "tools")
            domain = self.extract_field(
                content, "domain"
            ) or self.infer_domain_from_content(content)
            complexity_level = self.extract_field(content, "complexity") or "medium"
            tags = self.extract_list_field(content, "tags")

            # Generate file hash
            file_hash = self.calculate_file_hash(file_path)

            return AgentMetadata(
                id=agent_id,
                name=name,
                file_path=str(file_path),
                role=role,
                description=description,
                capabilities=capabilities,
                tools=tools,
                domain=domain,
                complexity_level=complexity_level,
                tags=tags,
                file_hash=file_hash,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return None

    def extract_field(self, content: str, field_name: str) -> Optional[str]:
        """Extract a field value from content."""
        import re

        pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_list_field(self, content: str, field_name: str) -> List[str]:
        """Extract a list field from content."""
        field_value = self.extract_field(content, field_name)
        if not field_value:
            return []

        # Split by comma, semicolon, or newline
        items = [
            item.strip() for item in re.split(r"[,;\n]", field_value) if item.strip()
        ]
        return items

    def infer_role_from_name(self, name: str) -> str:
        """Infer role from file name."""
        name_lower = name.lower()

        role_mappings = {
            "developer": "Developer",
            "analyst": "Analyst",
            "architect": "Architect",
            "manager": "Manager",
            "tester": "Tester",
            "designer": "Designer",
            "researcher": "Researcher",
            "writer": "Content Writer",
            "reviewer": "Code Reviewer",
        }

        for keyword, role in role_mappings.items():
            if keyword in name_lower:
                return role

        return "General Agent"

    def infer_domain_from_content(self, content: str) -> str:
        """Infer domain from content."""
        content_lower = content.lower()

        domain_keywords = {
            "web development": ["html", "css", "javascript", "react", "vue", "angular"],
            "data science": [
                "python",
                "pandas",
                "numpy",
                "machine learning",
                "data analysis",
            ],
            "mobile development": ["android", "ios", "flutter", "react native"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment"],
            "security": ["security", "vulnerability", "penetration testing"],
            "design": ["ui", "ux", "design", "figma", "sketch"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return domain

        return "general"

    async def scan_library(self, library_path: str) -> List[AgentMetadata]:
        """Scan an agent library and extract all agent metadata."""
        agents = []
        library_path = Path(library_path)

        if not library_path.exists():
            self.logger.warning(f"Library path does not exist: {library_path}")
            return agents

        self.logger.info(f"Scanning agent library: {library_path}")

        # Recursively scan directory
        for file_path in library_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.agent_extensions
            ):
                agent_metadata = self.extract_agent_metadata(file_path)
                if agent_metadata:
                    agents.append(agent_metadata)
                    self.logger.debug(f"Found agent: {agent_metadata.name}")

        self.logger.info(f"Found {len(agents)} agents in {library_path}")
        return agents


# === Enhanced Talent Scout Agent ===


class TalentScout(AgentForgeBase):
    """
    Enhanced Talent Scout with QDrant integration for semantic agent discovery.

    Specializes in:
    - Semantic similarity search across agent libraries
    - Intelligent capability matching
    - Adaptation recommendations
    - Gap analysis and new agent requirements
    """

    def __init__(self):
        super().__init__(
            name="Enhanced Talent Scout",
            description="Semantic agent discovery and intelligent reuse specialist",
        )
        self.logger = logging.getLogger(__name__)

        # Initialize QDrant manager
        self.qdrant = QDrantManager()

        # Initialize agent scanner
        self.scanner = AgentLibraryScanner()



        # Configuration
        self.similarity_threshold = 0.75
        self.adaptation_threshold = 0.6

        # Analytics
        self.performance_metrics = {
            "agents_indexed": 0,
            "search_queries": 0,
            "matches_found": 0,
            "adaptations_suggested": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the Talent Scout."""
        try:
            # Initialize QDrant collection
            success = await self.qdrant.initialize_collection()
            if not success:
                self.logger.error("Failed to initialize QDrant")
                return False

            self.logger.info("Talent Scout initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Talent Scout: {e}")
            return False

    async def index_agent_libraries(
        self, library_paths: List[str], force_reindex: bool = False
    ) -> Dict[str, Any]:
        """Index all agent libraries in QDrant."""
        start_time = datetime.now()
        indexing_stats = {
            "libraries_scanned": 0,
            "agents_found": 0,
            "agents_indexed": 0,
            "errors": 0,
            "processing_time_ms": 0,
        }

        try:
            all_agents = []

            # Scan all libraries
            for library_path in library_paths:
                try:
                    agents = await self.scanner.scan_library(library_path)
                    all_agents.extend(agents)
                    indexing_stats["libraries_scanned"] += 1
                    self.logger.info(
                        f"Scanned {len(agents)} agents from {library_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to scan {library_path}: {e}")
                    indexing_stats["errors"] += 1

            indexing_stats["agents_found"] = len(all_agents)

            # Index agents in QDrant
            for agent in all_agents:
                try:
                    success = await self.qdrant.index_agent(agent)
                    if success:
                        indexing_stats["agents_indexed"] += 1
                        self.performance_metrics["agents_indexed"] += 1
                    else:
                        indexing_stats["errors"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to index agent {agent.name}: {e}")
                    indexing_stats["errors"] += 1

            # Calculate processing time
            end_time = datetime.now()
            indexing_stats["processing_time_ms"] = (
                end_time - start_time
            ).total_seconds() * 1000

            self.logger.info(f"Indexing complete: {indexing_stats}")
            return indexing_stats

        except Exception as e:
            self.logger.error(f"Failed to index agent libraries: {e}")
            indexing_stats["errors"] += 1
            return indexing_stats

    def create_role_query_text(self, role: RoleRequirement) -> str:
        """Create search query text from role requirement."""
        components = [
            f"Role: {role.role_name}",
            f"Description: {role.description}",
            f"Required capabilities: {', '.join(role.required_capabilities)}",
            f"Preferred capabilities: {', '.join(role.preferred_capabilities)}",
            f"Domain: {role.domain}",
            f"Complexity: {role.complexity_level}",
        ]
        return " | ".join(components)

    def calculate_capability_match_score(
        self, agent: AgentMetadata, role: RoleRequirement
    ) -> float:
        """Calculate capability matching score between agent and role."""
        agent_caps = set(cap.lower() for cap in agent.capabilities)
        required_caps = set(cap.lower() for cap in role.required_capabilities)
        preferred_caps = set(cap.lower() for cap in role.preferred_capabilities)

        # Required capabilities match (weighted heavily)
        if required_caps:
            required_match = len(agent_caps & required_caps) / len(required_caps)
        else:
            required_match = 1.0

        # Preferred capabilities match (nice to have)
        if preferred_caps:
            preferred_match = len(agent_caps & preferred_caps) / len(preferred_caps)
        else:
            preferred_match = 1.0

        # Combined score (80% required, 20% preferred)
        return (required_match * 0.8) + (preferred_match * 0.2)

    def calculate_overall_score(
        self,
        similarity_score: float,
        capability_score: float,
        domain_match: bool = False,
        complexity_match: bool = False,
    ) -> float:
        """Calculate overall matching score."""
        # Base score from similarity and capabilities
        base_score = (similarity_score * 0.6) + (capability_score * 0.4)

        # Bonuses for domain and complexity match
        if domain_match:
            base_score += 0.05
        if complexity_match:
            base_score += 0.05

        return min(base_score, 1.0)  # Cap at 1.0

    def determine_match_confidence(self, overall_score: float) -> str:
        """Determine confidence level for a match."""
        if overall_score >= 0.85:
            return "high"
        elif overall_score >= 0.70:
            return "medium"
        else:
            return "low"

    def generate_match_reasoning(
        self,
        agent: AgentMetadata,
        role: RoleRequirement,
        similarity_score: float,
        capability_score: float,
    ) -> str:
        """Generate human-readable reasoning for the match."""
        reasons = []

        if similarity_score > 0.8:
            reasons.append(
                f"Strong semantic similarity ({similarity_score:.2f}) between agent description and role requirements"
            )

        if capability_score > 0.7:
            reasons.append(
                f"Good capability alignment ({capability_score:.2f}) with required skills"
            )

        if agent.domain.lower() == role.domain.lower():
            reasons.append("Exact domain match")

        if agent.complexity_level == role.complexity_level:
            reasons.append("Matching complexity level")

        return "; ".join(reasons) if reasons else "Basic compatibility found"

    def generate_adaptation_suggestions(
        self, agent: AgentMetadata, role: RoleRequirement, capability_score: float
    ) -> List[str]:
        """Generate suggestions for adapting an agent to better fit a role."""
        suggestions = []

        agent_caps = set(cap.lower() for cap in agent.capabilities)
        required_caps = set(cap.lower() for cap in role.required_capabilities)
        missing_caps = required_caps - agent_caps

        if missing_caps:
            suggestions.append(f"Add missing capabilities: {', '.join(missing_caps)}")

        if agent.domain.lower() != role.domain.lower():
            suggestions.append(f"Adapt from {agent.domain} to {role.domain} domain")

        if capability_score < 0.6:
            suggestions.append("Significant capability enhancement needed")

        return suggestions

    async def find_matches_for_role(
        self, role: RoleRequirement, limit: int = 10
    ) -> List[AgentMatch]:
        """Find potential agent matches for a specific role."""
        # Create search query
        query_text = self.create_role_query_text(role)

        # Search for similar agents
        search_results = await self.qdrant.search_similar_agents(
            query_text=query_text,
            limit=limit,
            score_threshold=0.4,  # Lower threshold to find adaptation candidates
            filters={"domain": role.domain} if role.domain != "general" else None,
        )

        matches = []
        self.performance_metrics["search_queries"] += 1

        for agent, similarity_score in search_results:
            # Calculate capability match
            capability_score = self.calculate_capability_match_score(agent, role)

            # Calculate overall score
            domain_match = agent.domain.lower() == role.domain.lower()
            complexity_match = agent.complexity_level == role.complexity_level
            overall_score = self.calculate_overall_score(
                similarity_score, capability_score, domain_match, complexity_match
            )

            # Determine if adaptation is needed
            adaptation_needed = overall_score < self.similarity_threshold
            adaptation_suggestions = []

            if adaptation_needed:
                adaptation_suggestions = self.generate_adaptation_suggestions(
                    agent, role, capability_score
                )
                self.performance_metrics["adaptations_suggested"] += 1

            # Create match
            match = AgentMatch(
                agent=agent,
                role_requirement=role,
                similarity_score=similarity_score,
                capability_match_score=capability_score,
                overall_score=overall_score,
                match_confidence=self.determine_match_confidence(overall_score),
                match_reasoning=self.generate_match_reasoning(
                    agent, role, similarity_score, capability_score
                ),
                adaptation_needed=adaptation_needed,
                adaptation_suggestions=adaptation_suggestions,
            )

            matches.append(match)
            self.performance_metrics["matches_found"] += 1

        # Sort by overall score (descending)
        matches.sort(key=lambda m: m.overall_score, reverse=True)

        return matches

    def create_vacant_role(
        self, role: RoleRequirement, closest_matches: List[AgentMatch]
    ) -> VacantRole:
        """Create a vacant role analysis."""
        # Analyze why no suitable agent exists
        if not closest_matches:
            gap_analysis = (
                "No similar agents found in the library for this specialized role."
            )
        else:
            best_match = closest_matches[0]
            if best_match.overall_score < 0.4:
                gap_analysis = "No agents with even basic compatibility found. This appears to be a novel role requirement."
            elif best_match.capability_match_score < 0.3:
                gap_analysis = f"Closest match ({best_match.agent.name}) lacks critical capabilities. Significant capability gap exists."
            else:
                gap_analysis = f"Closest match ({best_match.agent.name}) has {best_match.overall_score:.2f} compatibility but falls short of requirements."

        # Generate creation recommendations
        recommendations = [
            f"Create agent with role: {role.role_name}",
            f"Required capabilities: {', '.join(role.required_capabilities)}",
            f"Domain specialization: {role.domain}",
            f"Complexity level: {role.complexity_level}",
        ]

        if role.preferred_capabilities:
            recommendations.append(
                f"Consider adding: {', '.join(role.preferred_capabilities)}"
            )

        if closest_matches:
            best_match = closest_matches[0]
            recommendations.append(
                f"Use {best_match.agent.name} as inspiration but extend capabilities"
            )

        return VacantRole(
            role_requirement=role,
            gap_analysis=gap_analysis,
            closest_matches=closest_matches[:3],  # Top 3 for inspiration
            creation_recommendations=recommendations,
        )

    async def process(self, input_data: TalentScoutInput) -> TalentScoutOutput:
        """Process a strategy document and generate a comprehensive scouting report."""
        start_time = datetime.now()

        # Initialize if needed
        await self.initialize()

        # Index agent libraries (if needed)
        indexing_stats = await self.index_agent_libraries(
            input_data.agent_libraries, input_data.force_reindex
        )

        strategy = input_data.strategy_document
        matches = []
        vacant_positions = []

        # Process each role in the strategy
        for role in strategy.roles:
            role_matches = await self.find_matches_for_role(role)

            # Check if we have a suitable match
            if (
                role_matches
                and role_matches[0].overall_score >= self.similarity_threshold
            ):
                # Take the best match
                best_match = role_matches[0]
                matches.append(best_match)

                self.logger.info(
                    f"Matched role '{role.role_name}' with agent '{best_match.agent.name}' (score: {best_match.overall_score:.3f})"
                )

            else:
                # Create vacant role
                vacant_role = self.create_vacant_role(role, role_matches)
                vacant_positions.append(vacant_role)

                self.logger.info(
                    f"No suitable match for role '{role.role_name}' - creating vacant position"
                )

        # Calculate analytics
        total_roles = len(strategy.roles)
        filled_roles = len(matches)
        vacant_roles = len(vacant_positions)
        overall_coverage = filled_roles / total_roles if total_roles > 0 else 0
        reuse_efficiency = filled_roles / total_roles if total_roles > 0 else 0

        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        # Create scouting report
        scouting_report = ScoutingReport(
            strategy_title=strategy.title,
            total_roles=total_roles,
            filled_roles=filled_roles,
            vacant_roles=vacant_roles,
            matches=matches,
            vacant_positions=vacant_positions,
            overall_coverage=overall_coverage,
            reuse_efficiency=reuse_efficiency,
            processing_time_ms=processing_time_ms,
        )

        # Get collection stats
        collection_stats = await self.qdrant.get_collection_stats()

        return TalentScoutOutput(
            result="Scouting analysis completed successfully",
            status="success",
            metadata={
                "total_roles_analyzed": total_roles,
                "matches_found": filled_roles,
                "vacant_positions": vacant_roles,
                "reuse_efficiency": f"{reuse_efficiency:.1%}",
                "processing_time_ms": processing_time_ms,
            },
            scouting_report=scouting_report,
            indexing_stats=indexing_stats,
            performance_metrics={
                **self.performance_metrics,
                "collection_stats": collection_stats,
            },
        )


# === CLI Integration Functions ===


async def scout_agents_cli(
    strategy_path: str,
    agent_libraries: List[str] = None,
    output_path: str = None,
    force_reindex: bool = False,
) -> ScoutingReport:
    """CLI function for scouting agents."""
    # Default agent libraries
    if agent_libraries is None:
        agent_libraries = [
            "/home/delorenj/code/DeLoDocs/AI/Agents",
            "/home/delorenj/code/DeLoDocs/AI/Teams",
        ]

    # Load strategy document (placeholder - you'd implement actual loading)
    # For now, create a sample strategy
    sample_strategy = StrategyDocument(
        title="Sample Strategy",
        goal_description="Test goal",
        domain="web development",
        complexity_level="medium",
        roles=[
            RoleRequirement(
                role_id="dev1",
                role_name="Frontend Developer",
                description="React developer for UI",
                required_capabilities=["React", "JavaScript", "HTML", "CSS"],
                domain="web development",
                complexity_level="medium",
            )
        ],
    )

    # Create Talent Scout input
    scout_input = TalentScoutInput(
        goal="Scout agents for strategy",
        strategy_document=sample_strategy,
        agent_libraries=agent_libraries,
        force_reindex=force_reindex,
    )

    # Create and run scout
    scout = TalentScout()
    result = await scout.process(scout_input)

    # Save output if requested
    if output_path:
        with open(output_path, "w") as f:
            f.write(result.scouting_report.model_dump_json(indent=2))

    return result.scouting_report


# === Logging Setup ===


def setup_logging():
    """Setup logging for the Talent Scout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("talent_scout.log"), logging.StreamHandler()],
    )


if __name__ == "__main__":
    # Example usage
    setup_logging()

    async def main():
        scout = TalentScout()
        await scout.initialize()

        # Example indexing
        indexing_stats = await scout.index_agent_libraries(
            ["/home/delorenj/code/DeLoDocs/AI/Agents"]
        )
        print("Indexing Stats:", indexing_stats)

    asyncio.run(main())
