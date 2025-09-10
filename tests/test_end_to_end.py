"""
End-to-end tests for AgentForge system.

Tests the complete workflow from InputGoal to TeamPackage with realistic scenarios.
"""

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel, TeamPackage


class TestCompleteWorkflowE2E:
    """End-to-end tests for complete AgentForge workflows."""
    
    @pytest.mark.asyncio
    async def test_simple_blog_website_e2e(self):
        """Test complete workflow for simple blog website creation."""
        em = EngineeringManager()
        
        # Define the goal
        goal = InputGoal(
            goal_description="Create a personal blog website with user authentication and commenting system",
            domain="web development",
            complexity_level=ComplexityLevel.MEDIUM,
            timeline="6 weeks",
            constraints=["Modern design", "Fast loading", "SEO friendly"],
            success_criteria=[
                "User registration and login",
                "Create and edit blog posts", 
                "Comment system",
                "Responsive design",
                "Search functionality"
            ]
        )
        
        # Mock realistic responses for each stage
        strategy_doc = """
        # Strategy Document for Personal Blog Website
        
        ## Goal Analysis
        This project requires a modern web application with:
        - User authentication and session management
        - Content management for blog posts
        - Interactive commenting system
        - Responsive UI/UX design
        - Search and SEO optimization
        
        ## Team Composition
        
        ### Authentication Specialist
        - **Priority**: High
        - **Responsibilities**: User registration, login, session management, security
        - **Capabilities**: OAuth integration, JWT tokens, password hashing, security best practices
        
        ### Frontend Developer
        - **Priority**: High  
        - **Responsibilities**: React UI development, responsive design, user experience
        - **Capabilities**: React, CSS frameworks, responsive design, accessibility
        
        ### Backend Developer
        - **Priority**: High
        - **Responsibilities**: REST API development, business logic, database integration
        - **Capabilities**: Node.js, Express, API design, database management
        
        ### Content Management Specialist
        - **Priority**: Medium
        - **Responsibilities**: Blog post CRUD, rich text editing, media management
        - **Capabilities**: Content modeling, rich text editors, file upload handling
        
        ## Team Structure
        - **Topology**: Small cross-functional team (4 agents)
        - **Coordination**: API-first development with clear interfaces
        - **Communication**: Daily sync for integration points
        """
        
        scouting_report = """
        # Scouting Report for Blog Website Team
        
        ## Existing Agent Analysis
        Searched agent library at /home/delorenj/code/DeLoDocs/AI/Agents
        
        ## Matches Found
        - **existing_auth_specialist**: 90% match for Authentication Specialist role
          - Strong capabilities in JWT, OAuth, security
          - Can be reused with minimal adaptation
          
        - **existing_ui_developer**: 75% match for Frontend Developer role  
          - Good React skills, needs responsive design enhancement
          - Partial match, may need supplementation
        
        ## Gaps Identified
        **New agents required:**
        - **Backend Developer**: No suitable existing agent found
        - **Content Management Specialist**: No existing content management expertise
        
        ## Reuse Analysis
        - 50% of team can be fulfilled by existing agents
        - 50% requires new agent development
        - High-priority authentication role has excellent existing coverage
        
        ## Priority Recommendations
        1. Reuse existing_auth_specialist for authentication
        2. Adapt existing_ui_developer for frontend with enhancement
        3. Create new BackendDeveloper agent
        4. Create new ContentManagementSpecialist agent
        """
        
        new_agents = [
            {
                "name": "BlogBackendDeveloper",
                "role": "Backend Developer",
                "system_prompt": "You are a Backend Developer specializing in blog website APIs. You excel at creating RESTful APIs for content management, user authentication integration, and database operations. Your expertise includes Node.js, Express, database design, and API security.",
                "capabilities": [
                    "REST API design and development",
                    "Node.js and Express framework",
                    "Database schema design",
                    "API authentication and authorization",
                    "Content CRUD operations",
                    "Search API implementation"
                ],
                "tools_required": ["ReasoningTools", "KnowledgeTools", "DatabaseTools"],
                "priority": "high"
            },
            {
                "name": "ContentManagementSpecialist", 
                "role": "Content Management Specialist",
                "system_prompt": "You are a Content Management Specialist focused on blog content systems. You design and implement content workflows, rich text editing capabilities, and media management systems. Your expertise covers content modeling, editor integration, and publication workflows.",
                "capabilities": [
                    "Content modeling and architecture",
                    "Rich text editor integration",
                    "Media upload and management",
                    "Publication workflow design", 
                    "SEO optimization",
                    "Content search implementation"
                ],
                "tools_required": ["ReasoningTools", "ContentTools"],
                "priority": "medium"
            }
        ]
        
        roster_documentation = """
        # Blog Website Development Team Roster
        
        ## Project Overview
        **Goal**: Create a personal blog website with authentication and commenting
        **Timeline**: 6 weeks
        **Team Size**: 4 agents (2 existing, 2 new)
        
        ## Team Members
        
        ### Existing Agents (Reused)
        1. **existing_auth_specialist**
           - **Role**: Authentication Specialist
           - **Responsibilities**: User registration, login, security
           - **Integration**: Provide auth APIs for frontend and backend
        
        2. **existing_ui_developer** (Enhanced)
           - **Role**: Frontend Developer
           - **Responsibilities**: React UI, responsive design
           - **Enhancement**: Additional responsive design capabilities
        
        ### New Agents (Created)
        3. **BlogBackendDeveloper**
           - **Role**: Backend API Developer
           - **Responsibilities**: REST APIs, business logic, database
           - **Key Focus**: Blog post management, comment system APIs
        
        4. **ContentManagementSpecialist**
           - **Role**: Content Management
           - **Responsibilities**: Content workflows, rich editing, media
           - **Key Focus**: Blog authoring experience, SEO optimization
        
        ## Workflow Process
        
        ### Phase 1: Foundation (Week 1-2)
        - Authentication Specialist: Setup auth system
        - Backend Developer: Design database schema and core APIs
        - Frontend Developer: Create basic UI components
        - Content Specialist: Design content models
        
        ### Phase 2: Core Features (Week 3-4)  
        - Backend Developer: Implement blog post and comment APIs
        - Frontend Developer: Build blog UI and comment interface
        - Content Specialist: Integrate rich text editor
        - Authentication Specialist: Integrate auth with all components
        
        ### Phase 3: Enhancement (Week 5-6)
        - All agents: Search functionality implementation
        - Frontend Developer: Responsive design and SEO optimization  
        - Content Specialist: Media management and upload features
        - Backend Developer: Performance optimization
        
        ## Communication Protocols
        - **Daily Integration Sync**: 15-minute daily check-in
        - **API Contract Reviews**: Weekly review of interface changes
        - **Cross-component Testing**: Continuous integration testing
        - **Documentation**: Shared documentation repository
        
        ## Success Criteria Mapping
        - User registration/login → Authentication Specialist
        - Blog post creation → Backend Developer + Content Specialist
        - Comment system → Backend Developer + Frontend Developer  
        - Responsive design → Frontend Developer
        - Search functionality → Backend Developer + Content Specialist
        """
        
        # Set up mocks with realistic responses
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_doc)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_report)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_documentation)
        
        # Execute complete workflow
        result = await em.process(goal)
        
        # Comprehensive validation
        assert isinstance(result, TeamPackage)
        assert result.goal == goal
        
        # Validate strategy document
        assert "Strategy Document for Personal Blog Website" in result.strategy_document
        assert "Authentication Specialist" in result.strategy_document
        assert "Frontend Developer" in result.strategy_document
        assert "Backend Developer" in result.strategy_document
        assert "Content Management Specialist" in result.strategy_document
        
        # Validate scouting report
        assert "existing_auth_specialist" in result.scouting_report
        assert "90% match" in result.scouting_report
        assert "Gaps Identified" in result.scouting_report
        
        # Validate new agents
        assert len(result.new_agents) == 2
        agent_names = [agent["name"] for agent in result.new_agents]
        assert "BlogBackendDeveloper" in agent_names
        assert "ContentManagementSpecialist" in agent_names
        
        # Validate roster documentation
        assert "Blog Website Development Team Roster" in result.roster_documentation
        assert "4 agents (2 existing, 2 new)" in result.roster_documentation
        assert "Phase 1: Foundation" in result.roster_documentation
        
        # Validate deployment instructions
        assert len(result.deployment_instructions) > 0
        assert goal.goal_description in result.deployment_instructions
        assert "2" in result.deployment_instructions  # Number of new agents
        
        # Validate workflow tracking
        status = em.get_workflow_status()
        assert status["current_step"] == "final_packaging"
        assert status["history_length"] >= 6  # Should have logged major steps
        
        print(f"✅ Blog website E2E test completed successfully")
        print(f"   Strategy: {len(result.strategy_document)} chars")
        print(f"   Scouting: {len(result.scouting_report)} chars") 
        print(f"   New agents: {len(result.new_agents)}")
        print(f"   Roster: {len(result.roster_documentation)} chars")
    
    @pytest.mark.asyncio
    async def test_enterprise_ecommerce_platform_e2e(self):
        """Test complete workflow for enterprise e-commerce platform."""
        em = EngineeringManager()
        
        # Define complex enterprise goal
        goal = InputGoal(
            goal_description="Build an enterprise-grade e-commerce platform with AI-powered recommendations, multi-vendor support, real-time inventory management, and advanced analytics dashboard",
            domain="e-commerce",
            complexity_level=ComplexityLevel.ENTERPRISE,
            timeline="12 months",
            constraints=[
                "Microservices architecture",
                "Cloud-native deployment (AWS/Azure)",
                "PCI DSS compliance required",
                "Multi-region support",
                "99.99% uptime SLA",
                "Handle 100K concurrent users"
            ],
            success_criteria=[
                "Process 50K concurrent users",
                "Support 10K+ vendors with individual storefronts",
                "AI recommendations improve conversion by 30%",
                "Real-time inventory sync across all channels",
                "Comprehensive analytics and business intelligence",
                "Mobile-first responsive design",
                "Advanced search with filters and facets",
                "Multi-currency and multi-language support"
            ]
        )
        
        # Mock enterprise-level responses
        strategy_doc = """
        # Enterprise E-commerce Platform Strategy
        
        ## Executive Summary
        Large-scale, cloud-native e-commerce platform requiring 15+ specialized agents across multiple domains.
        
        ## System Architecture Analysis
        - **Scale**: 100K concurrent users, 10K+ vendors
        - **Architecture**: Event-driven microservices
        - **Data**: Multi-region distributed systems
        - **AI/ML**: Real-time recommendation engine
        - **Compliance**: PCI DSS, GDPR, accessibility
        
        ## Team Composition (15 Agents)
        
        ### Core Platform Team
        1. **Solution Architect** (Critical) - Overall system design
        2. **Microservices Architect** (Critical) - Service boundaries and communication
        3. **Data Architect** (Critical) - Data modeling and distributed data management
        4. **Security Architect** (Critical) - Security design and compliance
        
        ### Frontend & Mobile Team  
        5. **Frontend Platform Lead** (High) - Architecture and standards
        6. **React Specialist** (High) - Customer-facing interfaces
        7. **Mobile Developer** (High) - Native mobile applications
        8. **UX/UI Designer** (High) - User experience design
        
        ### Backend Services Team
        9. **API Gateway Specialist** (High) - API management and routing
        10. **Catalog Service Developer** (High) - Product catalog and search
        11. **Order Management Developer** (High) - Order processing workflows
        12. **Payment Systems Developer** (Critical) - Payment processing and compliance
        
        ### AI & Analytics Team
        13. **ML Engineer** (High) - Recommendation algorithms
        14. **Analytics Engineer** (Medium) - Business intelligence and reporting
        15. **Search Engineer** (Medium) - Advanced search capabilities
        
        ### Infrastructure Team
        16. **DevOps Architect** (High) - CI/CD and deployment automation
        17. **SRE Specialist** (High) - Reliability and monitoring
        18. **Performance Engineer** (Medium) - Optimization and scalability
        
        ## Integration Patterns
        - Event-driven architecture with Apache Kafka
        - API-first design with OpenAPI specifications
        - Distributed data with eventual consistency
        - Real-time communication via WebSocket and Server-Sent Events
        """
        
        scouting_report = """
        # Enterprise E-commerce Scouting Report
        
        ## Scale of Analysis
        Analyzed 200+ existing agents across 15 specialized domains
        
        ## High-Value Matches Found
        - **enterprise_security_architect**: 95% match - Security Architect role
        - **api_gateway_specialist**: 90% match - API Gateway Specialist role  
        - **react_platform_developer**: 85% match - Frontend Platform Lead role
        - **ml_recommendation_engine**: 80% match - ML Engineer role
        - **payment_systems_expert**: 90% match - Payment Systems Developer role
        
        ## Partial Matches (Need Enhancement)
        - **existing_devops_engineer**: 70% match for DevOps Architect (needs enterprise scaling)
        - **existing_mobile_developer**: 65% match for Mobile Developer (needs e-commerce domain)
        
        ## Critical Gaps (New Agents Required)
        - Solution Architect (enterprise e-commerce)
        - Microservices Architect (high-scale distributed systems)
        - Data Architect (multi-region distributed data)
        - Order Management Developer (complex workflow orchestration)
        - Analytics Engineer (e-commerce business intelligence)
        - SRE Specialist (enterprise reliability engineering)
        
        ## Reuse Analysis
        - **40% reuse rate** - High-value existing agents for specialized roles
        - **35% enhancement** - Good agents needing domain-specific adaptation
        - **25% new development** - Highly specialized roles requiring new agents
        
        ## Risk Assessment
        - HIGH RISK: Solution and Data Architect roles are critical and have no existing coverage
        - MEDIUM RISK: Integration complexity between 18 agents requires strong coordination
        - LOW RISK: Well-established patterns for payment and security roles
        """
        
        new_agents = [
            {
                "name": "EnterpriseSolutionArchitect",
                "role": "Solution Architect", 
                "system_prompt": "You are an Enterprise Solution Architect specializing in large-scale e-commerce platforms. You design system architectures that can handle millions of users, complex business workflows, and integration with hundreds of external systems. Your expertise spans microservices, event-driven architecture, data consistency patterns, and enterprise integration patterns.",
                "capabilities": [
                    "Enterprise architecture design",
                    "Microservices decomposition",
                    "System integration patterns",
                    "Scalability and performance architecture",
                    "Technology stack selection",
                    "Architecture governance"
                ],
                "priority": "critical"
            },
            {
                "name": "EcommerceMicroservicesArchitect",
                "role": "Microservices Architect",
                "system_prompt": "You are a Microservices Architect specialized in e-commerce domain decomposition. You excel at defining service boundaries, designing inter-service communication patterns, and implementing distributed data management. Your focus is on creating loosely coupled, highly cohesive services that can scale independently.",
                "capabilities": [
                    "Domain-driven design",
                    "Service boundary definition", 
                    "Inter-service communication design",
                    "Distributed data patterns",
                    "Service orchestration vs choreography",
                    "API design and governance"
                ],
                "priority": "critical"
            }
            # Additional agents would be included here...
        ]
        
        roster_documentation = """
        # Enterprise E-commerce Platform Team Roster
        
        ## Project Scale
        - **Duration**: 12 months  
        - **Team Size**: 18 agents (8 existing, 10 new)
        - **Architecture**: Event-driven microservices
        - **Target Scale**: 100K concurrent users, 10K+ vendors
        
        ## Team Organization
        
        ### Architecture & Leadership Tier (4 agents)
        - EnterpriseSolutionArchitect: Overall system design and technical direction
        - EcommerceMicroservicesArchitect: Service decomposition and boundaries
        - enterprise_security_architect (existing): Security design and compliance  
        - DataArchitect: Distributed data architecture
        
        ### Platform & Infrastructure Tier (5 agents)
        - existing_devops_engineer (enhanced): Enterprise CI/CD and deployment
        - SRESpecialist: Reliability engineering and monitoring
        - api_gateway_specialist (existing): API management and routing
        - PerformanceEngineer: System optimization and scaling
        - CloudInfrastructureSpecialist: Multi-region cloud deployment
        
        ### Application Development Tier (9 agents)
        Frontend Team (4):
        - react_platform_developer (existing): Frontend architecture
        - ReactSpecialist: Customer interfaces
        - existing_mobile_developer (enhanced): Mobile applications
        - UXDesigner: User experience design
        
        Backend Team (5):
        - CatalogServiceDeveloper: Product catalog and search
        - OrderManagementDeveloper: Order processing workflows
        - payment_systems_expert (existing): Payment processing
        - InventoryServiceDeveloper: Real-time inventory management
        - NotificationServiceDeveloper: Multi-channel notifications
        
        ## Development Phases
        
        ### Phase 1: Foundation (Months 1-3)
        Architecture and core infrastructure setup
        
        ### Phase 2: Core Services (Months 4-7)
        Essential business services development
        
        ### Phase 3: Advanced Features (Months 8-10)
        AI/ML, analytics, and optimization
        
        ### Phase 4: Scale & Polish (Months 11-12)
        Performance tuning, compliance, and launch preparation
        """
        
        # Set up enterprise-scale mocks
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_doc)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_report)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_documentation)
        
        # Execute enterprise workflow
        result = await em.process(goal)
        
        # Comprehensive enterprise validation
        assert isinstance(result, TeamPackage)
        assert result.goal == goal
        assert result.goal.complexity_level == ComplexityLevel.ENTERPRISE
        
        # Validate enterprise-scale strategy
        assert "15+ specialized agents" in result.strategy_document
        assert "Solution Architect" in result.strategy_document  
        assert "Microservices" in result.strategy_document
        assert "Event-driven architecture" in result.strategy_document
        
        # Validate comprehensive scouting
        assert "200+ existing agents" in result.scouting_report
        assert "40% reuse rate" in result.scouting_report
        assert "HIGH RISK" in result.scouting_report
        
        # Validate enterprise agent creation
        assert len(result.new_agents) == 2  # Mock data has 2, real would have more
        assert any("EnterpriseSolutionArchitect" in agent["name"] for agent in result.new_agents)
        
        # Validate enterprise roster
        assert "18 agents" in result.roster_documentation
        assert "Architecture & Leadership Tier" in result.roster_documentation
        assert "Phase 1: Foundation" in result.roster_documentation
        
        print(f"✅ Enterprise e-commerce E2E test completed successfully")
        print(f"   Complexity: {goal.complexity_level}")
        print(f"   Timeline: {goal.timeline}")
        print(f"   Constraints: {len(goal.constraints)}")
        print(f"   Success criteria: {len(goal.success_criteria)}")
    
    @pytest.mark.asyncio
    async def test_ai_research_platform_e2e(self):
        """Test complete workflow for AI research platform."""
        em = EngineeringManager()
        
        goal = InputGoal(
            goal_description="Build an AI research collaboration platform with experiment tracking, model versioning, distributed training, and knowledge sharing",
            domain="artificial intelligence",
            complexity_level=ComplexityLevel.HIGH,
            timeline="9 months",
            constraints=[
                "GPU cluster integration",
                "Support for multiple ML frameworks",
                "Academic collaboration features",
                "Open source components where possible"
            ],
            success_criteria=[
                "Support 1000+ concurrent research projects",
                "Experiment reproducibility and tracking",
                "Model versioning and deployment pipeline",
                "Distributed training on GPU clusters",
                "Real-time collaboration tools",
                "Integration with academic databases"
            ]
        )
        
        # Mock AI research platform responses
        strategy_doc = "# AI Research Platform Strategy\n\nSpecialized platform requiring ML infrastructure expertise..."
        scouting_report = "# AI Research Platform Scouting\n\nFound 3 existing ML specialists, need 5 new agents..."
        new_agents = [
            {
                "name": "MLInfrastructureArchitect",
                "role": "ML Infrastructure Architect",
                "capabilities": ["Kubernetes", "GPU orchestration", "ML pipelines"],
                "priority": "critical"
            }
        ]
        roster_documentation = "# AI Research Platform Team\n\n8 specialized agents for ML research infrastructure..."
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_doc)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_report)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_documentation)
        
        result = await em.process(goal)
        
        # Validate AI research platform specifics
        assert isinstance(result, TeamPackage)
        assert result.goal.domain == "artificial intelligence"
        assert "GPU cluster" in str(result.goal.constraints)
        assert "ML frameworks" in str(result.goal.constraints)
        assert any("MLInfrastructure" in agent["name"] for agent in result.new_agents)
        
        print(f"✅ AI research platform E2E test completed successfully")


class TestErrorRecoveryE2E:
    """End-to-end tests for error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_systems_analyst_failure_recovery(self, sample_input_goal):
        """Test system recovery when Systems Analyst fails."""
        em = EngineeringManager()
        
        # First attempt fails
        em._delegate_to_systems_analyst = AsyncMock(side_effect=Exception("Analysis failed"))
        
        with pytest.raises(Exception, match="Analysis failed"):
            await em.process(sample_input_goal)
        
        # Verify partial workflow state
        status = em.get_workflow_status()
        assert status["current_step"] == "strategy_analysis"
        assert len(em.workflow_history) >= 2  # Should have logged intake and attempt
    
    @pytest.mark.asyncio
    async def test_partial_agent_development_recovery(self, sample_input_goal):
        """Test recovery when agent development partially fails."""
        em = EngineeringManager()
        
        # Setup successful early stages
        em._delegate_to_systems_analyst = AsyncMock(return_value="Strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Scouting with 3 gaps")
        
        # Agent developer succeeds but with limited results
        em._delegate_to_agent_developer = AsyncMock(return_value=[
            {"name": "PartialAgent", "status": "created"},
            {"name": "FailedAgent", "status": "failed", "error": "Creation failed"}
        ])
        
        em._delegate_to_integration_architect = AsyncMock(return_value="Roster with partial team")
        
        # Should complete despite partial failure
        result = await em.process(sample_input_goal)
        
        assert isinstance(result, TeamPackage)
        assert len(result.new_agents) == 2  # Includes both successful and failed
        assert any(agent.get("status") == "failed" for agent in result.new_agents)
    
    @pytest.mark.asyncio
    async def test_timeout_handling_e2e(self, sample_input_goal):
        """Test handling of timeouts in long-running operations."""
        em = EngineeringManager()
        
        async def slow_systems_analyst(goal):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return "Delayed strategy"
        
        em._delegate_to_systems_analyst = slow_systems_analyst
        em._delegate_to_talent_scout = AsyncMock(return_value="Quick scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="Quick roster")
        
        # Should complete despite delays
        result = await asyncio.wait_for(em.process(sample_input_goal), timeout=5.0)
        
        assert isinstance(result, TeamPackage)
        assert result.strategy_document == "Delayed strategy"


class TestPerformanceE2E:
    """End-to-end performance tests."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_workflow_e2e(self):
        """Test system performance under high throughput."""
        em = EngineeringManager()
        
        # Setup fast mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="Fast strategy")
        em._delegate_to_talent_scout = AsyncMock(return_value="Fast scouting")
        em._delegate_to_agent_developer = AsyncMock(return_value=[{"name": "FastAgent"}])
        em._delegate_to_integration_architect = AsyncMock(return_value="Fast roster")
        
        # Create many goals
        goals = [
            InputGoal(
                goal_description=f"Goal {i}",
                domain=f"domain{i % 5}",  # Rotate domains
                complexity_level=list(ComplexityLevel)[i % 4]  # Rotate complexity
            )
            for i in range(20)
        ]
        
        # Process all concurrently and measure time
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[em.process(goal) for goal in goals])
        end_time = asyncio.get_event_loop().time()
        
        # Validate throughput
        processing_time = end_time - start_time
        throughput = len(goals) / processing_time
        
        assert len(results) == 20
        assert all(isinstance(result, TeamPackage) for result in results)
        assert throughput > 10  # Should process at least 10 goals per second with mocks
        
        print(f"✅ High throughput test: {throughput:.1f} goals/second")
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability_e2e(self):
        """Test that memory usage remains stable across multiple workflows."""
        em = EngineeringManager()
        
        # Setup minimal mock responses
        em._delegate_to_systems_analyst = AsyncMock(return_value="S")
        em._delegate_to_talent_scout = AsyncMock(return_value="T")
        em._delegate_to_agent_developer = AsyncMock(return_value=[])
        em._delegate_to_integration_architect = AsyncMock(return_value="R")
        
        # Process multiple workflows
        for i in range(10):
            goal = InputGoal(goal_description=f"Test {i}", domain="test")
            result = await em.process(goal)
            assert isinstance(result, TeamPackage)
        
        # Verify workflow history doesn't grow excessively
        assert len(em.workflow_history) < 100  # Should not accumulate indefinitely
    
    @pytest.mark.asyncio
    async def test_complex_goal_processing_time(self, complex_input_goal):
        """Test processing time for complex enterprise goals."""
        em = EngineeringManager()
        
        # Mock responses proportional to complexity
        em._delegate_to_systems_analyst = AsyncMock(return_value="Complex strategy " * 100)
        em._delegate_to_talent_scout = AsyncMock(return_value="Complex scouting " * 100)
        em._delegate_to_agent_developer = AsyncMock(return_value=[
            {"name": f"ComplexAgent{i}"} for i in range(10)
        ])
        em._delegate_to_integration_architect = AsyncMock(return_value="Complex roster " * 100)
        
        start_time = asyncio.get_event_loop().time()
        result = await em.process(complex_input_goal)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        assert isinstance(result, TeamPackage)
        assert len(result.new_agents) == 10
        assert processing_time < 2.0  # Should complete within 2 seconds even for complex goals
        
        print(f"✅ Complex goal processing: {processing_time:.3f} seconds")


class TestIntegrationScenariosE2E:
    """End-to-end tests for various integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_domain_project_e2e(self):
        """Test project spanning multiple domains."""
        em = EngineeringManager()
        
        goal = InputGoal(
            goal_description="Build a healthcare IoT platform with mobile apps, cloud analytics, and regulatory compliance",
            domain="healthcare, IoT, mobile, cloud computing",
            complexity_level=ComplexityLevel.ENTERPRISE,
            constraints=["HIPAA compliance", "FDA regulations", "Real-time data", "IoT device integration"],
            success_criteria=["Regulatory approval", "Real-time monitoring", "Mobile accessibility"]
        )
        
        strategy_doc = "Multi-domain strategy requiring specialists in healthcare, IoT, mobile, cloud, and compliance..."
        scouting_report = "Cross-domain analysis found partial matches, need domain bridge specialists..."
        new_agents = [
            {"name": "HealthcareComplianceSpecialist", "domain": "healthcare"},
            {"name": "IoTIntegrationEngineer", "domain": "IoT"},
            {"name": "HealthcareUIUXSpecialist", "domain": "mobile+healthcare"}
        ]
        roster_documentation = "Cross-functional team with domain specialists and bridge roles..."
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_doc)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_report)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_documentation)
        
        result = await em.process(goal)
        
        assert isinstance(result, TeamPackage)
        assert "healthcare, IoT, mobile, cloud computing" in result.goal.domain
        assert "HIPAA compliance" in result.goal.constraints
        assert any("Healthcare" in agent["name"] for agent in result.new_agents)
        assert any("IoT" in agent["name"] for agent in result.new_agents)
        
        print(f"✅ Multi-domain project E2E test completed successfully")
    
    @pytest.mark.asyncio
    async def test_legacy_system_integration_e2e(self):
        """Test project involving legacy system integration."""
        em = EngineeringManager()
        
        goal = InputGoal(
            goal_description="Modernize legacy mainframe banking system with modern API layer and web interfaces",
            domain="financial services",
            complexity_level=ComplexityLevel.HIGH,
            constraints=[
                "Must maintain existing COBOL core",
                "Zero downtime migration",
                "Regulatory compliance maintained",
                "Gradual modernization approach"
            ],
            success_criteria=[
                "Modern API layer operational", 
                "Web interface for all transactions",
                "Legacy system integration maintained",
                "Performance improvements achieved"
            ]
        )
        
        strategy_doc = "Legacy modernization strategy with API facade pattern and gradual migration..."
        scouting_report = "Found mainframe specialists, need modern integration experts..."
        new_agents = [
            {"name": "LegacyIntegrationSpecialist", "capabilities": ["COBOL", "Mainframe", "API integration"]},
            {"name": "ModernizationArchitect", "capabilities": ["Legacy migration", "API design", "Banking domain"]}
        ]
        roster_documentation = "Hybrid team with legacy and modern specialists working in parallel..."
        
        em._delegate_to_systems_analyst = AsyncMock(return_value=strategy_doc)
        em._delegate_to_talent_scout = AsyncMock(return_value=scouting_report)
        em._delegate_to_agent_developer = AsyncMock(return_value=new_agents)
        em._delegate_to_integration_architect = AsyncMock(return_value=roster_documentation)
        
        result = await em.process(goal)
        
        assert isinstance(result, TeamPackage)
        assert "mainframe banking system" in result.goal.goal_description
        assert "Zero downtime migration" in result.goal.constraints
        assert any("Legacy" in agent["name"] for agent in result.new_agents)
        assert any("Modernization" in agent["name"] for agent in result.new_agents)
        
        print(f"✅ Legacy system integration E2E test completed successfully")