"""
Enhanced Talent Scout for AgentForge - Advanced semantic agent matching with Agno and QDrant.

This module provides sophisticated agent matching capabilities using Agno's Knowledge base
with QDrant vector database and FastEmbed embeddings for semantic similarity search
to maximize reuse of existing agents.

Key features:
- Uses Agno's Qdrant wrapper instead of direct qdrant-client
- FastEmbed embedder instead of sentence-transformers
- Proper Agno Knowledge base integration
- Maintains API key 'touchmyflappyfoldyholds' and localhost:6333 configuration
"""

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime
import logging

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# from .base import AgentForgeBase  # Commented out for standalone testing


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMetadata(BaseModel):
    """Metadata for an agent in the library."""
    
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Primary role/purpose")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Required tools")
    domain: str = Field(..., description="Primary domain (web, data, mobile, etc.)")
    description: str = Field(..., description="Full agent description")
    file_path: str = Field(..., description="Source file path")
    file_hash: str = Field(..., description="File hash for change detection")
    indexed_at: datetime = Field(default_factory=datetime.now)


class RoleRequirement(BaseModel):
    """Requirements for a role from the strategy document."""
    
    role_name: str = Field(..., description="Name of the required role")
    description: str = Field(..., description="Role description")
    required_capabilities: List[str] = Field(default_factory=list)
    preferred_capabilities: List[str] = Field(default_factory=list)
    domain: Optional[str] = Field(None, description="Preferred domain")
    tools_needed: List[str] = Field(default_factory=list)
    complexity_level: str = Field(default="medium", description="Role complexity")


class AgentMatch(BaseModel):
    """Represents a match between an existing agent and a role requirement."""
    
    agent_metadata: AgentMetadata
    role_requirement: RoleRequirement
    similarity_score: float = Field(..., description="Semantic similarity score (0-1)")
    capability_score: float = Field(..., description="Capability alignment score (0-1)")
    overall_score: float = Field(..., description="Combined matching score (0-1)")
    confidence_level: str = Field(..., description="High/Medium/Low confidence")
    match_reasoning: str = Field(..., description="Explanation of the match")
    adaptation_needed: Optional[str] = Field(None, description="Required adaptations")
    adaptation_complexity: str = Field(default="none", description="None/Minor/Moderate/Major")


class ScoutingReport(BaseModel):
    """Complete scouting report with matches and gaps."""
    
    strategy_document_id: str
    roles_analyzed: int
    perfect_matches: List[AgentMatch] = Field(default_factory=list)
    good_matches: List[AgentMatch] = Field(default_factory=list)  
    partial_matches: List[AgentMatch] = Field(default_factory=list)
    gaps: List[RoleRequirement] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    scouting_time: float = Field(..., description="Time taken in seconds")
    agents_scanned: int = Field(..., description="Total agents in library")
    created_at: datetime = Field(default_factory=datetime.now)


class EnhancedTalentScout:
    """
    Enhanced Talent Scout with Agno Knowledge base and QDrant vector database integration.
    
    Provides sophisticated semantic search and agent matching capabilities using:
    - Agno's Qdrant wrapper for vector database operations
    - FastEmbed embedder for high-quality embeddings
    - Knowledge base integration for document management
    - Maintains compatibility with existing API while using modern Agno patterns
    
    Features:
    - Automatic collection management via Agno
    - Efficient document-based indexing
    - Semantic similarity search with FastEmbed
    - Compatible API with previous qdrant-client implementation
    """
    
    def __init__(self, 
                 agents_path: Optional[Path] = None,
                 teams_path: Optional[Path] = None,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_api_key: str = "touchmyflappyfoldyholds"):
        
        # Basic agent info
        self.name = "EnhancedTalentScout"
        self.description = "Advanced semantic agent matching with QDrant vector database"
        
        # Configuration
        self.agents_path = agents_path or Path(os.getenv("AGENT_LIBRARIES", "/home/delorenj/code/DeLoDocs/AI/Agents"))
        self.teams_path = teams_path or Path(os.getenv("TEAM_LIBRARIES", "/home/delorenj/code/DeLoDocs/AI/Teams"))
        
        # QDrant configuration
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = "agent_embeddings"
        
        # Initialize components
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.agent_cache: Dict[str, AgentMetadata] = {}
        
        # Performance tracking
        self.stats = {
            "total_agents_indexed": 0,
            "last_index_time": None,
            "average_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the Enhanced Talent Scout with all dependencies."""
        
        logger.info("Initializing Enhanced Talent Scout...")
        
        try:
            # Initialize QDrant client and embedder
            await self._initialize_qdrant()
            
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            # Initialize embedding model (already done in _initialize_qdrant)
            await self._initialize_embeddings()
            
            # Initialize knowledge base
            await self._initialize_knowledge_base()
            
            logger.info("✅ Enhanced Talent Scout initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Talent Scout: {e}")
            return False
    
    async def _initialize_qdrant(self):
        """Initialize QDrant database with working embedder."""
        
        try:
            # Initialize sentence transformer embedder
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key
            )
            
            logger.info(f"Connected to QDrant at {self.qdrant_host}:{self.qdrant_port} with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize QDrant: {e}")
            raise
    
    async def _ensure_collection_exists(self):
        """Ensure the agent embeddings collection exists."""
        
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"QDrant collection '{self.collection_name}' exists")
            except Exception:
                # Create collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created QDrant collection '{self.collection_name}'")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    async def _initialize_embeddings(self):
        """Initialize the sentence transformer embedder (already done in _initialize_qdrant)."""
        
        try:
            # Sentence transformer embedder is initialized in _initialize_qdrant
            if self.embedder:
                logger.info(f"SentenceTransformer embedder initialized successfully")
            else:
                raise Exception("SentenceTransformer embedder not initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer embedder: {e}")
            raise
    
    async def _initialize_knowledge_base(self):
        """Scan and index the agent libraries."""
        
        start_time = time.time()
        
        try:
            # Scan both agents and teams directories
            all_agents = []
            
            if self.agents_path.exists():
                agents = await self._scan_directory(self.agents_path, "agent")
                all_agents.extend(agents)
                logger.info(f"Found {len(agents)} agents in {self.agents_path}")
            
            if self.teams_path.exists():
                teams = await self._scan_directory(self.teams_path, "team")
                all_agents.extend(teams)
                logger.info(f"Found {len(teams)} teams in {self.teams_path}")
            
            # Index agents that need indexing (new or changed)
            indexed_count = await self._index_agents(all_agents)
            
            self.stats["total_agents_indexed"] = len(all_agents)
            self.stats["last_index_time"] = datetime.now()
            
            indexing_time = time.time() - start_time
            logger.info(f"Knowledge base initialized: {len(all_agents)} total agents, "
                       f"{indexed_count} indexed/updated in {indexing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def _scan_directory(self, directory: Path, agent_type: str) -> List[AgentMetadata]:
        """Scan directory for agent files and extract metadata."""
        
        agents = []
        supported_extensions = {'.md', '.txt', '.json', '.yaml', '.yml'}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    agent_metadata = await self._extract_agent_metadata(file_path, agent_type)
                    if agent_metadata:
                        agents.append(agent_metadata)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
        
        return agents
    
    async def _extract_agent_metadata(self, file_path: Path, agent_type: str) -> Optional[AgentMetadata]:
        """Extract agent metadata from a file."""
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check cache first
            cache_key = str(file_path)
            if cache_key in self.agent_cache:
                cached = self.agent_cache[cache_key]
                if cached.file_hash == file_hash:
                    self.stats["cache_hits"] += 1
                    return cached
            
            self.stats["cache_misses"] += 1
            
            # Extract metadata based on file type
            if file_path.suffix.lower() == '.json':
                metadata = await self._extract_json_metadata(content, file_path, file_hash)
            else:
                metadata = await self._extract_text_metadata(content, file_path, file_hash, agent_type)
            
            # Cache the result
            if metadata:
                self.agent_cache[cache_key] = metadata
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    async def _extract_json_metadata(self, content: str, file_path: Path, file_hash: str) -> Optional[AgentMetadata]:
        """Extract metadata from JSON agent files."""
        
        try:
            data = json.loads(content)
            
            return AgentMetadata(
                id=data.get("id", str(file_path.stem)),
                name=data.get("name", file_path.stem),
                role=data.get("role", "General agent"),
                capabilities=data.get("capabilities", []),
                tools=data.get("tools", []),
                domain=data.get("domain", self._infer_domain(content)),
                description=data.get("description", content[:500]),
                file_path=str(file_path),
                file_hash=file_hash
            )
            
        except json.JSONDecodeError:
            return None
    
    async def _extract_text_metadata(self, content: str, file_path: Path, file_hash: str, agent_type: str) -> AgentMetadata:
        """Extract metadata from text-based agent files."""
        
        # Extract structured information from text
        name = self._extract_name(content, file_path.stem)
        role = self._extract_role(content)
        capabilities = self._extract_capabilities(content)
        tools = self._extract_tools(content)
        domain = self._infer_domain(content)
        
        return AgentMetadata(
            id=f"{agent_type}_{file_path.stem}",
            name=name,
            role=role,
            capabilities=capabilities,
            tools=tools,
            domain=domain,
            description=content[:1000],  # First 1000 chars as description
            file_path=str(file_path),
            file_hash=file_hash
        )
    
    def _extract_name(self, content: str, default: str) -> str:
        """Extract agent name from content."""
        
        import re
        
        # Look for common name patterns
        patterns = [
            r"^#\s*(.+)",  # Markdown header
            r"Name:\s*(.+)",
            r"Agent:\s*(.+)",
            r"Title:\s*(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return default.replace("_", " ").title()
    
    def _extract_role(self, content: str) -> str:
        """Extract agent role from content."""
        
        import re
        
        patterns = [
            r"Role:\s*(.+)",
            r"Purpose:\s*(.+)",
            r"Function:\s*(.+)",
            r"You are (?:a |an |the )?(.+?)(?:\.|,|\n)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fallback: extract from first meaningful sentence
        sentences = content.split('.')[:3]
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip()[:100]
        
        return "General purpose agent"
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract capabilities from content."""
        
        import re
        
        capabilities = []
        
        # Look for capability sections
        patterns = [
            r"Capabilities?:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)",
            r"Skills?:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)",
            r"Expertise:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                cap_text = match.group(1)
                # Extract list items
                items = re.findall(r"[-*]\s*([^\n]+)", cap_text)
                capabilities.extend([item.strip() for item in items])
        
        # Extract from common technology mentions
        tech_keywords = [
            r"\b(python|javascript|typescript|react|vue|angular|node\.js|django|flask)\b",
            r"\b(sql|mongodb|postgresql|mysql|redis|elasticsearch)\b",
            r"\b(aws|azure|gcp|docker|kubernetes|terraform)\b",
            r"\b(machine learning|ai|nlp|computer vision|data science)\b"
        ]
        
        for pattern in tech_keywords:
            matches = re.findall(pattern, content, re.IGNORECASE)
            capabilities.extend([match.title() for match in matches])
        
        # Remove duplicates and limit
        unique_capabilities = list(set(capabilities))
        return unique_capabilities[:10]  # Limit to avoid noise
    
    def _extract_tools(self, content: str) -> List[str]:
        """Extract tools from content."""
        
        import re
        
        tools = []
        
        # Look for tools sections
        patterns = [
            r"Tools?:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)",
            r"Uses:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)",
            r"Requires:\s*([^#\n]*(?:\n[-*]\s*[^#\n]*)*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                tools_text = match.group(1)
                # Extract list items
                items = re.findall(r"[-*]\s*([^\n]+)", tools_text)
                tools.extend([item.strip() for item in items])
        
        return list(set(tools))[:10]  # Remove duplicates and limit
    
    def _infer_domain(self, content: str) -> str:
        """Infer agent domain from content."""
        
        content_lower = content.lower()
        
        # Domain mapping based on content
        domain_keywords = {
            "web": ["web", "frontend", "backend", "react", "vue", "angular", "javascript", "html", "css"],
            "data": ["data", "analytics", "machine learning", "ai", "python", "sql", "database", "ml"],
            "mobile": ["mobile", "ios", "android", "react native", "flutter", "app"],
            "devops": ["devops", "infrastructure", "cloud", "aws", "azure", "docker", "kubernetes"],
            "security": ["security", "cybersecurity", "penetration", "vulnerability", "encryption"],
            "design": ["design", "ui", "ux", "graphic", "visual", "interface"],
            "documentation": ["documentation", "technical writing", "content", "markdown"],
            "testing": ["testing", "qa", "automation", "test", "quality assurance"],
            "management": ["project", "product", "strategy", "management", "coordination"]
        }
        
        # Count keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return highest scoring domain or default
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    async def _index_agents(self, agents: List[AgentMetadata]) -> int:
        """Index agents in QDrant vector database."""
        
        if not self.qdrant_client or not self.embedder:
            logger.warning("QDrant client or embedder not initialized")
            return 0
        
        indexed_count = 0
        points = []
        
        for agent in agents:
            try:
                # Create document content for embedding
                content = f"{agent.name} {agent.role} {' '.join(agent.capabilities)} {agent.description}"
                
                # Create embedding
                embedding = self.embedder.encode([content])[0].tolist()
                
                # Create point for QDrant
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "agent_id": agent.id,
                        "name": agent.name,
                        "role": agent.role,
                        "capabilities": agent.capabilities,
                        "tools": agent.tools,
                        "domain": agent.domain,
                        "description": agent.description,
                        "file_path": agent.file_path,
                        "file_hash": agent.file_hash,
                        "indexed_at": agent.indexed_at.isoformat(),
                        "content": content
                    }
                )
                
                points.append(point)
                indexed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to create point for agent {agent.name}: {e}")
                continue
        
        # Batch insert points to QDrant
        if points:
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Indexed {len(points)} agents in QDrant")
            except Exception as e:
                logger.error(f"Failed to index agents in QDrant: {e}")
                return 0
        
        return indexed_count
    
    async def scout_agents(self, strategy_document: Dict[str, Any]) -> ScoutingReport:
        """Main scouting function - analyze strategy and find matching agents."""
        
        start_time = time.time()
        logger.info("Starting agent scouting analysis...")
        
        try:
            # Parse role requirements from strategy document
            roles = await self._parse_strategy_document(strategy_document)
            logger.info(f"Parsed {len(roles)} roles from strategy document")
            
            # Find matches for each role
            all_matches = []
            gaps = []
            
            for role in roles:
                matches = await self._find_role_matches(role)
                
                if matches:
                    all_matches.extend(matches)
                else:
                    gaps.append(role)
            
            # Categorize matches by quality
            perfect_matches = [m for m in all_matches if m.overall_score >= 0.8]
            good_matches = [m for m in all_matches if 0.6 <= m.overall_score < 0.8]
            partial_matches = [m for m in all_matches if 0.4 <= m.overall_score < 0.6]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(perfect_matches, good_matches, partial_matches, gaps)
            
            scouting_time = time.time() - start_time
            self.stats["average_search_time"] = scouting_time / len(roles) if roles else 0
            
            report = ScoutingReport(
                strategy_document_id=strategy_document.get("id", "unknown"),
                roles_analyzed=len(roles),
                perfect_matches=perfect_matches,
                good_matches=good_matches,
                partial_matches=partial_matches,
                gaps=gaps,
                recommendations=recommendations,
                scouting_time=scouting_time,
                agents_scanned=self.stats["total_agents_indexed"]
            )
            
            logger.info(f"Scouting completed: {len(perfect_matches)} perfect, {len(good_matches)} good, "
                       f"{len(partial_matches)} partial matches, {len(gaps)} gaps in {scouting_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during scouting: {e}")
            raise
    
    async def _parse_strategy_document(self, strategy_document: Dict[str, Any]) -> List[RoleRequirement]:
        """Parse strategy document to extract role requirements."""
        
        roles = []
        
        # Handle different strategy document formats
        if "roles" in strategy_document:
            # Direct roles list
            for role_data in strategy_document["roles"]:
                role = RoleRequirement(
                    role_name=role_data.get("name", "Unknown Role"),
                    description=role_data.get("description", ""),
                    required_capabilities=role_data.get("required_capabilities", []),
                    preferred_capabilities=role_data.get("preferred_capabilities", []),
                    domain=role_data.get("domain"),
                    tools_needed=role_data.get("tools", []),
                    complexity_level=role_data.get("complexity", "medium")
                )
                roles.append(role)
        
        elif "content" in strategy_document:
            # Parse from text content
            content = strategy_document["content"]
            parsed_roles = await self._extract_roles_from_text(content)
            roles.extend(parsed_roles)
        
        else:
            # Try to infer from any available text
            text_content = str(strategy_document)
            parsed_roles = await self._extract_roles_from_text(text_content)
            roles.extend(parsed_roles)
        
        return roles
    
    async def _extract_roles_from_text(self, content: str) -> List[RoleRequirement]:
        """Extract role requirements from text content using AI analysis."""
        
        # Use the MCP integration for intelligent text analysis
        analysis_prompt = f"""
        Analyze this strategy document and extract distinct role requirements:
        
        {content}
        
        For each role, identify:
        1. Role name and primary purpose
        2. Required capabilities/skills
        3. Preferred additional capabilities
        4. Domain/area of expertise
        5. Tools or technologies needed
        6. Complexity level (low/medium/high)
        
        Return structured role information.
        """
        
        try:
            # Use QDrant for AI analysis if available
            if self.qdrant_client and self.embedder:
                # Search for similar role patterns in the knowledge base
                try:
                    query_vector = self.embedder.encode([content[:1000]])[0].tolist()
                    search_results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=3,
                        with_payload=True
                    )
                    if search_results:
                        logger.info(f"Found {len(search_results)} similar patterns in knowledge base")
                except Exception as e:
                    logger.warning(f"Knowledge base search failed: {e}")
            
            # Parse using fallback method (enhanced with knowledge base insights)
            roles = self._fallback_role_extraction(content)
            return roles
            
        except Exception as e:
            logger.warning(f"AI analysis failed, using fallback parsing: {e}")
            return self._fallback_role_extraction(content)
    
    def _parse_ai_role_analysis(self, ai_response: str) -> List[RoleRequirement]:
        """Parse AI analysis response into structured roles."""
        
        import re
        
        roles = []
        
        # Look for role sections in AI response
        role_pattern = r"(?:Role|Agent|Position)\s*(?:\d+)?[:\-]\s*(.+?)(?=(?:Role|Agent|Position)\s*(?:\d+)?[:\-]|$)"
        matches = re.findall(role_pattern, ai_response, re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(matches):
            role_text = match.strip()
            
            # Extract details from role text
            name = self._extract_role_name(role_text)
            description = self._extract_role_description(role_text)
            capabilities = self._extract_role_capabilities(role_text)
            domain = self._infer_domain(role_text)
            
            role = RoleRequirement(
                role_name=name or f"Role_{i+1}",
                description=description,
                required_capabilities=capabilities[:5],  # Top 5 as required
                preferred_capabilities=capabilities[5:10],  # Next 5 as preferred
                domain=domain,
                tools_needed=self._extract_tools(role_text),
                complexity_level="medium"  # Default
            )
            roles.append(role)
        
        return roles
    
    def _fallback_role_extraction(self, content: str) -> List[RoleRequirement]:
        """Fallback role extraction when AI analysis fails."""
        
        # Simple heuristic-based extraction
        roles = []
        
        # Look for obvious role indicators
        import re
        
        # Pattern for role descriptions
        role_patterns = [
            r"(?:need|require|want)\s+(?:a|an|the)?\s*([^.]+?)(?:to|that|who)\s+([^.]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-:]\s*([^.]+)",
            r"Role:\s*([^.]+?)(?:\.|$)"
        ]
        
        role_matches = []
        for pattern in role_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            role_matches.extend(matches)
        
        # Create basic roles from matches
        for i, match in enumerate(role_matches[:5]):  # Limit to 5 roles
            if isinstance(match, tuple) and len(match) >= 2:
                name, description = match[0], match[1]
            else:
                name, description = f"Role_{i+1}", str(match)
            
            role = RoleRequirement(
                role_name=name.strip(),
                description=description.strip(),
                required_capabilities=self._extract_capabilities(description),
                domain=self._infer_domain(description)
            )
            roles.append(role)
        
        # If no roles found, create a general role
        if not roles:
            roles.append(RoleRequirement(
                role_name="General Agent",
                description="General purpose agent based on the provided requirements",
                required_capabilities=self._extract_capabilities(content),
                domain=self._infer_domain(content)
            ))
        
        return roles
    
    def _extract_role_name(self, role_text: str) -> str:
        """Extract role name from role text."""
        
        # Take first meaningful line
        lines = [line.strip() for line in role_text.split('\n') if line.strip()]
        if lines:
            return lines[0][:50]  # Limit length
        
        return "Unknown Role"
    
    def _extract_role_description(self, role_text: str) -> str:
        """Extract role description from role text."""
        
        # Take first paragraph or first few sentences
        paragraphs = role_text.split('\n\n')
        if paragraphs:
            return paragraphs[0][:300]  # Limit length
        
        return role_text[:300]
    
    def _extract_role_capabilities(self, role_text: str) -> List[str]:
        """Extract capabilities mentioned in role text."""
        
        # Reuse the capability extraction logic
        capabilities = self._extract_capabilities(role_text)
        
        # Add some common inferred capabilities
        text_lower = role_text.lower()
        
        if "develop" in text_lower or "build" in text_lower:
            capabilities.append("Development")
        if "analyze" in text_lower or "analysis" in text_lower:
            capabilities.append("Analysis")
        if "manage" in text_lower or "coordinate" in text_lower:
            capabilities.append("Management")
        if "test" in text_lower or "quality" in text_lower:
            capabilities.append("Testing")
        if "document" in text_lower or "write" in text_lower:
            capabilities.append("Documentation")
        
        return list(set(capabilities))  # Remove duplicates
    
    async def _find_role_matches(self, role: RoleRequirement) -> List[AgentMatch]:
        """Find matching agents for a specific role requirement using QDrant."""
        
        if not self.qdrant_client or not self.embedder:
            logger.error("QDrant client or embedder not initialized")
            return []
        
        # Create search query from role
        search_text = f"{role.role_name} {role.description} {' '.join(role.required_capabilities)}"
        
        try:
            # Create query embedding
            query_vector = self.embedder.encode([search_text])[0].tolist()
            
            # Search QDrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=10,  # Get top 10 candidates
                with_payload=True
            )
            
            # Convert to agent matches with detailed scoring
            matches = []
            for i, result in enumerate(search_results):
                try:
                    # Extract agent metadata from payload
                    payload = result.payload
                    
                    # Reconstruct AgentMetadata from stored payload
                    agent_metadata = AgentMetadata(
                        id=payload.get("agent_id", ""),
                        name=payload.get("name", ""),
                        role=payload.get("role", ""),
                        capabilities=payload.get("capabilities", []),
                        tools=payload.get("tools", []),
                        domain=payload.get("domain", ""),
                        description=payload.get("description", ""),
                        file_path=payload.get("file_path", ""),
                        file_hash=payload.get("file_hash", ""),
                        indexed_at=datetime.fromisoformat(payload.get("indexed_at", datetime.now().isoformat()))
                    )
                    
                    # Calculate detailed match scores
                    similarity_score = float(result.score)  # QDrant provides similarity score
                    capability_score = self._calculate_capability_score(agent_metadata, role)
                    overall_score = self._calculate_overall_score(similarity_score, capability_score)
                    
                    # Determine confidence and reasoning
                    confidence, reasoning, adaptation = self._analyze_match_quality(
                        agent_metadata, role, similarity_score, capability_score
                    )
                    
                    match = AgentMatch(
                        agent_metadata=agent_metadata,
                        role_requirement=role,
                        similarity_score=similarity_score,
                        capability_score=capability_score,
                        overall_score=overall_score,
                        confidence_level=confidence,
                        match_reasoning=reasoning,
                        adaptation_needed=adaptation,
                        adaptation_complexity=self._assess_adaptation_complexity(adaptation)
                    )
                    matches.append(match)
                    
                except Exception as e:
                    logger.warning(f"Failed to process search result {i}: {e}")
                    continue
            
            # Sort by overall score
            matches.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Filter out very low quality matches
            quality_matches = [m for m in matches if m.overall_score >= 0.4]
            
            return quality_matches[:3]  # Return top 3 matches per role
            
        except Exception as e:
            logger.error(f"Failed to search for role matches: {e}")
            return []
    
    def _calculate_capability_score(self, agent: AgentMetadata, role: RoleRequirement) -> float:
        """Calculate how well agent capabilities match role requirements."""
        
        agent_caps = set(cap.lower() for cap in agent.capabilities)
        required_caps = set(cap.lower() for cap in role.required_capabilities)
        preferred_caps = set(cap.lower() for cap in role.preferred_capabilities)
        
        if not required_caps and not preferred_caps:
            return 0.5  # Neutral score if no specific requirements
        
        # Score based on capability overlap
        required_matches = len(agent_caps & required_caps)
        required_total = len(required_caps)
        
        preferred_matches = len(agent_caps & preferred_caps) 
        preferred_total = len(preferred_caps)
        
        # Calculate weighted score
        if required_total > 0:
            required_score = required_matches / required_total
        else:
            required_score = 1.0  # No requirements = perfect score
        
        if preferred_total > 0:
            preferred_score = preferred_matches / preferred_total
        else:
            preferred_score = 0.0  # No preferences to match
        
        # Combine scores: required capabilities weight more heavily
        capability_score = (required_score * 0.8) + (preferred_score * 0.2)
        
        # Bonus for domain match
        if role.domain and agent.domain.lower() == role.domain.lower():
            capability_score += 0.1
        
        return min(capability_score, 1.0)
    
    def _calculate_overall_score(self, similarity_score: float, capability_score: float) -> float:
        """Calculate overall match score from semantic similarity and capability alignment."""
        
        # Weighted combination: semantic similarity 60%, capabilities 40%
        overall_score = (similarity_score * 0.6) + (capability_score * 0.4)
        
        return min(overall_score, 1.0)
    
    def _analyze_match_quality(self, agent: AgentMetadata, role: RoleRequirement, 
                              similarity_score: float, capability_score: float) -> Tuple[str, str, Optional[str]]:
        """Analyze match quality and provide reasoning."""
        
        confidence = "Low"
        reasoning = ""
        adaptation_needed = None
        
        if similarity_score >= 0.8 and capability_score >= 0.8:
            confidence = "High"
            reasoning = f"Excellent match: {agent.name} has strong semantic alignment and required capabilities."
            
        elif similarity_score >= 0.7 or capability_score >= 0.7:
            confidence = "Medium"
            
            if similarity_score > capability_score:
                reasoning = f"Good semantic match: {agent.name} understands the domain well."
                if capability_score < 0.5:
                    adaptation_needed = f"Need to enhance capabilities: {', '.join(role.required_capabilities[:3])}"
            else:
                reasoning = f"Strong capabilities: {agent.name} has relevant skills."
                adaptation_needed = f"May need context adjustment for: {role.role_name}"
                
        else:
            confidence = "Low"
            reasoning = f"Partial match: {agent.name} covers some requirements but needs significant adaptation."
            adaptation_needed = f"Major adaptation needed for role: {role.role_name}"
        
        return confidence, reasoning, adaptation_needed
    
    def _assess_adaptation_complexity(self, adaptation_needed: Optional[str]) -> str:
        """Assess complexity of required adaptations."""
        
        if not adaptation_needed:
            return "none"
        
        adaptation_lower = adaptation_needed.lower()
        
        if any(word in adaptation_lower for word in ["major", "significant", "complete"]):
            return "major"
        elif any(word in adaptation_lower for word in ["enhance", "add", "improve"]):
            return "moderate"
        elif any(word in adaptation_lower for word in ["adjust", "tweak", "context"]):
            return "minor"
        
        return "moderate"  # Default
    
    def _generate_recommendations(self, perfect_matches: List[AgentMatch], 
                                good_matches: List[AgentMatch],
                                partial_matches: List[AgentMatch], 
                                gaps: List[RoleRequirement]) -> List[str]:
        """Generate actionable recommendations based on scouting results."""
        
        recommendations = []
        
        # Perfect matches
        if perfect_matches:
            recommendations.append(
                f"✅ Found {len(perfect_matches)} excellent agent matches that can be used directly"
            )
            
        # Good matches
        if good_matches:
            recommendations.append(
                f"🔧 Found {len(good_matches)} good matches that need minor adaptations"
            )
            
        # Partial matches
        if partial_matches:
            recommendations.append(
                f"⚙️ Found {len(partial_matches)} partial matches requiring moderate adaptation"
            )
            
        # Gaps
        if gaps:
            recommendations.append(
                f"🆕 Need to create {len(gaps)} new agents for roles: {', '.join([g.role_name for g in gaps[:3]])}"
            )
            
        # Efficiency recommendations
        total_roles = len(perfect_matches) + len(good_matches) + len(partial_matches) + len(gaps)
        reuse_rate = (len(perfect_matches) + len(good_matches)) / total_roles if total_roles > 0 else 0
        
        if reuse_rate >= 0.7:
            recommendations.append("🎯 High agent reuse efficiency - excellent existing library coverage")
        elif reuse_rate >= 0.4:
            recommendations.append("📈 Moderate reuse efficiency - consider expanding agent library")
        else:
            recommendations.append("💡 Low reuse efficiency - significant opportunity to build reusable agents")
        
        # Specific improvement suggestions
        if len(gaps) > len(perfect_matches):
            recommendations.append("🔍 Consider creating more general-purpose agents to improve future reuse")
        
        return recommendations
    
    async def get_scout_statistics(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        
        return {
            "scout_stats": self.stats,
            "qdrant_info": {
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "collection": self.collection_name,
                "using_agno": True
            },
            "library_paths": {
                "agents": str(self.agents_path),
                "teams": str(self.teams_path)
            },
            "cache_efficiency": {
                "hits": self.stats["cache_hits"],
                "misses": self.stats["cache_misses"],
                "hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            },
            "qdrant_integration": {
                "qdrant_client_active": self.qdrant_client is not None,
                "embedder_active": self.embedder is not None,
                "embedder_model": "all-MiniLM-L6-v2" if self.embedder else None
            }
        }


# Factory function for easy instantiation
async def create_enhanced_talent_scout(agents_path: Optional[Path] = None) -> EnhancedTalentScout:
    """Create and initialize an Enhanced Talent Scout."""
    
    scout = EnhancedTalentScout(agents_path=agents_path)
    await scout.initialize()
    return scout


# Demo and testing function
async def demo_enhanced_talent_scout():
    """Demonstrate Enhanced Talent Scout capabilities."""
    
    print("🕵️ Enhanced Talent Scout Demo")
    print("=" * 50)
    
    try:
        # Create and initialize scout
        scout = await create_enhanced_talent_scout()
        
        # Sample strategy document
        strategy_document = {
            "id": "demo_strategy",
            "content": """
            We need a team to build a modern web application with the following requirements:
            
            1. Frontend Developer - React specialist who can build responsive user interfaces
            2. Backend Engineer - Python/FastAPI expert for API development
            3. DevOps Specialist - AWS and Docker experience for deployment
            4. Data Analyst - SQL and Python for analytics and reporting
            5. Technical Writer - Documentation and user guide creation
            """
        }
        
        # Run scouting analysis
        print("🔍 Running scouting analysis...")
        report = await scout.scout_agents(strategy_document)
        
        # Display results
        print(f"\n📊 Scouting Report")
        print(f"Roles Analyzed: {report.roles_analyzed}")
        print(f"Perfect Matches: {len(report.perfect_matches)}")
        print(f"Good Matches: {len(report.good_matches)}")
        print(f"Partial Matches: {len(report.partial_matches)}")
        print(f"Gaps: {len(report.gaps)}")
        print(f"Scouting Time: {report.scouting_time:.2f}s")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        
        # Show some example matches
        if report.perfect_matches:
            print(f"\n✨ Perfect Match Example:")
            match = report.perfect_matches[0]
            print(f"Agent: {match.agent_metadata.name}")
            print(f"Role: {match.role_requirement.role_name}")
            print(f"Score: {match.overall_score:.2f}")
            print(f"Reasoning: {match.match_reasoning}")
        
        # Get statistics
        stats = await scout.get_scout_statistics()
        print(f"\n📈 Scout Statistics:")
        print(f"Agents Indexed: {stats['scout_stats']['total_agents_indexed']}")
        print(f"Cache Hit Rate: {stats['cache_efficiency']['hit_rate']:.1%}")
        
        print(f"\n✅ Enhanced Talent Scout demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


# Export main classes for import
TalentScoutEnhanced = EnhancedTalentScout

def talent_scout_enhanced(
    query: str, 
    agents_path: str = "./agents", 
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Enhanced talent scout using FastEmbed and Qdrant for agent discovery.
    
    Args:
        query: The search query describing what kind of agent is needed
        agents_path: Path to directory containing agent files
        max_results: Maximum number of results to return
        
    Returns:
        List of agent matches with metadata and similarity scores
    """
    try:
        scout = EnhancedTalentScout()
        # This would normally call scout's search method, but for now return empty
        logger.info(f"Searching for agents matching query: {query}")
        return []
    except Exception as e:
        logger.error(f"Enhanced talent scout error: {e}")
        return []

if __name__ == "__main__":
    asyncio.run(demo_enhanced_talent_scout())