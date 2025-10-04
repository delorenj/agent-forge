"""
Enhanced Talent Scout Demonstration Script

This script demonstrates the capabilities of the Enhanced Talent Scout with QDrant integration:
- Agent library indexing and scanning
- Semantic similarity search
- Intelligent agent matching
- Adaptation recommendations
- Comprehensive scouting reports

Usage:
    python demo_enhanced_talent_scout.py [--mode demo|interactive|benchmark]
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

from agents.talent_scout import (
    TalentScout,
    TalentScoutInput,
    StrategyDocument,
    RoleRequirement,
    AgentMetadata,
    AgentLibraryScanner,
    QDrantManager
)


class TalentScoutDemo:
    """Demonstration orchestrator for the Enhanced Talent Scout."""
    
    def __init__(self):
        self.scout = TalentScout()
        self.temp_dir = None
        self.sample_agents_created = False
    
    def create_sample_agent_library(self) -> str:
        """Create a sample agent library for demonstration."""
        if self.sample_agents_created:
            return self.temp_dir
        
        self.temp_dir = tempfile.mkdtemp(prefix="agent_forge_demo_")
        temp_path = Path(self.temp_dir)
        
        # Sample agent definitions
        sample_agents = [
            {
                "filename": "react_frontend_developer.md",
                "content": """# React Frontend Developer Agent

**Role:** Frontend Developer
**Description:** Expert React developer specializing in modern, responsive web applications with TypeScript and state management
**Capabilities:** React, JavaScript, TypeScript, HTML5, CSS3, Sass, Redux, Context API, React Router, Webpack
**Tools:** Create React App, Next.js, Vite, ESLint, Prettier, Jest, React Testing Library, Storybook
**Domain:** web development  
**Complexity:** high
**Tags:** frontend, react, typescript, ui, responsive, spa

## System Prompt
You are an expert React frontend developer with 8+ years of experience building scalable web applications. 
You excel at component architecture, state management, performance optimization, and modern development practices.

### Key Strengths:
- Component-based architecture design
- Advanced React patterns (hooks, context, HOCs)
- TypeScript integration and type safety
- Responsive design and cross-browser compatibility
- Performance optimization and code splitting
- Testing strategies and implementation
"""
            },
            {
                "filename": "nodejs_backend_developer.md", 
                "content": """# Node.js Backend Developer Agent

**Role:** Backend Developer
**Description:** Senior Node.js developer specializing in scalable APIs, microservices, and cloud-native applications
**Capabilities:** Node.js, JavaScript, TypeScript, Express.js, Fastify, PostgreSQL, MongoDB, Redis, Docker, Kubernetes
**Tools:** npm, yarn, PM2, Docker, Kubernetes, Jest, Supertest, Swagger, Postman
**Domain:** web development
**Complexity:** high
**Tags:** backend, nodejs, api, microservices, database, cloud

## System Prompt
You are a senior Node.js backend developer with expertise in building high-performance, scalable server applications.
You have deep knowledge of API design, database optimization, security best practices, and cloud deployment.

### Core Competencies:
- RESTful API and GraphQL development
- Database design and optimization
- Authentication and authorization systems
- Microservices architecture
- Performance monitoring and optimization
- DevOps and containerization
"""
            },
            {
                "filename": "python_data_scientist.md",
                "content": """# Python Data Scientist Agent

**Role:** Data Scientist
**Description:** Expert data scientist specializing in machine learning, statistical analysis, and data visualization
**Capabilities:** Python, R, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Matplotlib, Seaborn, Jupyter
**Tools:** Anaconda, Jupyter Lab, Git, Docker, MLflow, Apache Airflow, Tableau, Power BI
**Domain:** data science
**Complexity:** high
**Tags:** python, machine-learning, statistics, visualization, analytics

## System Prompt
You are an expert data scientist with extensive experience in machine learning, statistical modeling, and data analysis.
You excel at extracting insights from complex datasets and building predictive models.

### Expertise Areas:
- Statistical analysis and hypothesis testing
- Machine learning model development and deployment
- Data preprocessing and feature engineering
- Data visualization and storytelling
- A/B testing and experimental design
- Big data processing and analytics
"""
            },
            {
                "filename": "devops_engineer.md",
                "content": """# DevOps Engineer Agent

**Role:** DevOps Engineer  
**Description:** Senior DevOps engineer focused on CI/CD, infrastructure automation, and cloud platform management
**Capabilities:** Docker, Kubernetes, AWS, Azure, GCP, Terraform, Ansible, Jenkins, GitLab CI, Prometheus, Grafana
**Tools:** Docker, Kubernetes, Terraform, Ansible, Jenkins, GitLab, Prometheus, Grafana, ELK Stack
**Domain:** infrastructure
**Complexity:** high
**Tags:** devops, kubernetes, cloud, automation, monitoring, cicd

## System Prompt
You are an experienced DevOps engineer specializing in modern infrastructure automation, containerization, and cloud platforms.
You focus on reliability, scalability, and security in distributed systems.

### Key Areas:
- Infrastructure as Code (IaC) with Terraform
- Container orchestration with Kubernetes
- CI/CD pipeline design and implementation
- Monitoring and observability solutions
- Cloud architecture and security
- Site Reliability Engineering (SRE) practices
"""
            },
            {
                "filename": "ui_ux_designer.md",
                "content": """# UI/UX Designer Agent

**Role:** UI/UX Designer
**Description:** Creative UI/UX designer with expertise in user-centered design, prototyping, and design systems
**Capabilities:** User Research, Wireframing, Prototyping, Visual Design, Interaction Design, Usability Testing, Design Systems
**Tools:** Figma, Sketch, Adobe XD, InVision, Miro, Principle, Framer, Adobe Creative Suite
**Domain:** design
**Complexity:** medium
**Tags:** ui, ux, design, prototyping, user-research, figma

## System Prompt
You are a talented UI/UX designer with a strong focus on user-centered design principles and modern design practices.
You excel at creating intuitive, accessible, and visually appealing digital experiences.

### Design Philosophy:
- User-first approach to design decisions
- Iterative design process with continuous feedback
- Accessibility and inclusive design principles
- Data-driven design optimization
- Collaborative design system development
- Cross-platform consistency
"""
            },
            {
                "filename": "qa_automation_engineer.md",
                "content": """# QA Automation Engineer Agent

**Role:** QA Engineer
**Description:** Quality assurance specialist focusing on test automation, performance testing, and quality processes
**Capabilities:** Selenium, Cypress, Playwright, Jest, JUnit, TestNG, API Testing, Performance Testing, Test Planning
**Tools:** Selenium WebDriver, Cypress, Playwright, Postman, JMeter, Jenkins, TestRail, JIRA
**Domain:** quality assurance
**Complexity:** medium
**Tags:** qa, automation, testing, selenium, cypress, performance

## System Prompt
You are an experienced QA automation engineer dedicated to ensuring software quality through comprehensive testing strategies.
You specialize in building robust test frameworks and maintaining high code quality standards.

### Testing Expertise:
- Test automation framework development
- API testing and validation
- Performance and load testing
- Cross-browser and mobile testing  
- Test data management
- Quality metrics and reporting
"""
            },
            {
                "filename": "mobile_developer.md", 
                "content": """# Mobile Developer Agent

**Role:** Mobile Developer
**Description:** Cross-platform mobile developer specializing in React Native and Flutter applications
**Capabilities:** React Native, Flutter, Dart, JavaScript, TypeScript, iOS Development, Android Development, Mobile UI/UX
**Tools:** React Native, Flutter, Xcode, Android Studio, Expo, Firebase, App Store Connect, Google Play Console
**Domain:** mobile development
**Complexity:** high
**Tags:** mobile, react-native, flutter, ios, android, cross-platform

## System Prompt  
You are a skilled mobile developer with expertise in building high-quality cross-platform mobile applications.
You focus on performance, user experience, and platform-specific best practices.

### Mobile Expertise:
- Cross-platform development strategies
- Native module integration
- Mobile performance optimization
- App store deployment processes
- Mobile security best practices
- Offline functionality and data sync
"""
            },
            {
                "filename": "security_specialist.md",
                "content": """# Security Specialist Agent

**Role:** Security Specialist
**Description:** Cybersecurity expert focusing on application security, penetration testing, and security architecture
**Capabilities:** Penetration Testing, Security Architecture, OWASP, Vulnerability Assessment, Incident Response, Compliance
**Tools:** Burp Suite, OWASP ZAP, Nessus, Metasploit, Wireshark, Kali Linux, Splunk, Security Frameworks
**Domain:** cybersecurity
**Complexity:** high
**Tags:** security, penetration-testing, vulnerability, compliance, owasp

## System Prompt
You are a cybersecurity specialist with extensive experience in application security, threat modeling, and security architecture.
You help organizations build secure systems and respond to security incidents.

### Security Focus Areas:
- Application security assessment and testing
- Security architecture design and review
- Threat modeling and risk assessment
- Incident response and forensics
- Compliance and regulatory requirements
- Security awareness and training
"""
            }
        ]
        
        # Create agent files
        for agent_data in sample_agents:
            file_path = temp_path / agent_data["filename"]
            file_path.write_text(agent_data["content"])
        
        print(f"✅ Created {len(sample_agents)} sample agents in: {self.temp_dir}")
        self.sample_agents_created = True
        return self.temp_dir
    
    def create_sample_strategy(self) -> StrategyDocument:
        """Create a sample strategy document for demonstration."""
        return StrategyDocument(
            title="Modern E-Commerce Platform Development Strategy",
            goal_description="Build a comprehensive e-commerce platform with modern architecture, scalable backend, responsive frontend, and robust security",
            domain="web development",
            complexity_level="high",
            timeline="6 months",
            constraints=["Budget: $500k", "Team size: 8-10 people", "Launch deadline: Q3 2024"],
            roles=[
                RoleRequirement(
                    role_id="frontend-lead",
                    role_name="Frontend Team Lead",
                    description="Lead frontend developer responsible for React-based e-commerce UI with advanced features",
                    required_capabilities=["React", "TypeScript", "Redux", "Responsive Design", "Team Leadership"],
                    preferred_capabilities=["Next.js", "GraphQL", "Performance Optimization", "E-commerce UX"],
                    domain="web development",
                    complexity_level="high",
                    priority="critical"
                ),
                RoleRequirement(
                    role_id="backend-architect",
                    role_name="Backend Architect", 
                    description="Senior backend developer to design and implement scalable e-commerce APIs and microservices",
                    required_capabilities=["Node.js", "Microservices", "Database Design", "API Development", "System Architecture"],
                    preferred_capabilities=["Event-Driven Architecture", "Message Queues", "Caching Strategies", "Payment Integration"],
                    domain="web development", 
                    complexity_level="high",
                    priority="critical"
                ),
                RoleRequirement(
                    role_id="mobile-developer",
                    role_name="Mobile App Developer",
                    description="Cross-platform mobile developer for e-commerce mobile application",
                    required_capabilities=["React Native", "Mobile UI/UX", "API Integration", "App Store Deployment"],
                    preferred_capabilities=["Flutter", "Push Notifications", "Offline Support", "Mobile Analytics"],
                    domain="mobile development",
                    complexity_level="high",
                    priority="high"
                ),
                RoleRequirement(
                    role_id="devops-engineer",
                    role_name="DevOps Engineer",
                    description="Infrastructure and deployment specialist for cloud-native e-commerce platform",
                    required_capabilities=["Kubernetes", "Docker", "CI/CD", "Cloud Platforms", "Monitoring"],
                    preferred_capabilities=["Terraform", "Service Mesh", "Security Hardening", "Cost Optimization"],
                    domain="infrastructure",
                    complexity_level="high",
                    priority="high"
                ),
                RoleRequirement(
                    role_id="ux-designer",
                    role_name="UX/UI Designer", 
                    description="User experience designer for e-commerce customer journey optimization",
                    required_capabilities=["User Research", "Wireframing", "Prototyping", "E-commerce UX"],
                    preferred_capabilities=["A/B Testing", "Conversion Optimization", "Mobile Design", "Accessibility"],
                    domain="design",
                    complexity_level="medium",
                    priority="medium"
                ),
                RoleRequirement(
                    role_id="qa-lead",
                    role_name="QA Automation Lead",
                    description="Quality assurance lead for comprehensive e-commerce testing strategy",
                    required_capabilities=["Test Automation", "API Testing", "Performance Testing", "Test Strategy"],
                    preferred_capabilities=["E-commerce Testing", "Load Testing", "Security Testing", "Mobile Testing"],
                    domain="quality assurance",
                    complexity_level="medium",
                    priority="medium"
                ),
                RoleRequirement(
                    role_id="data-analyst",
                    role_name="E-commerce Data Analyst",
                    description="Data analyst for customer behavior analysis and business intelligence",
                    required_capabilities=["Python", "SQL", "Data Analysis", "Business Intelligence", "Visualization"],
                    preferred_capabilities=["E-commerce Analytics", "Customer Segmentation", "Predictive Modeling", "A/B Testing Analysis"],
                    domain="data science",
                    complexity_level="medium", 
                    priority="low"
                ),
                RoleRequirement(
                    role_id="security-specialist",
                    role_name="Security Specialist",
                    description="Cybersecurity expert for e-commerce platform security architecture",
                    required_capabilities=["Security Architecture", "Penetration Testing", "Compliance", "Threat Modeling"],
                    preferred_capabilities=["E-commerce Security", "PCI DSS", "Payment Security", "GDPR Compliance"],
                    domain="cybersecurity",
                    complexity_level="high",
                    priority="high"
                )
            ]
        )
    
    async def demo_agent_indexing(self):
        """Demonstrate agent library indexing process."""
        print("\n🔍 DEMO: Agent Library Indexing")
        print("=" * 50)
        
        # Create sample library
        library_path = self.create_sample_agent_library()
        
        # Initialize QDrant
        print("Initializing QDrant vector database...")
        await self.scout.initialize()
        
        # Index agents
        print(f"Indexing agents from: {library_path}")
        start_time = time.time()
        
        indexing_stats = await self.scout.index_agent_libraries([library_path], force_reindex=True)
        
        end_time = time.time()
        indexing_time = (end_time - start_time) * 1000
        
        # Display results
        print(f"\n✅ Indexing Complete!")
        print(f"📊 Libraries scanned: {indexing_stats['libraries_scanned']}")
        print(f"🤖 Agents found: {indexing_stats['agents_found']}")  
        print(f"💾 Agents indexed: {indexing_stats['agents_indexed']}")
        print(f"❌ Errors: {indexing_stats['errors']}")
        print(f"⏱️  Processing time: {indexing_time:.1f}ms")
        
        # Get collection stats
        collection_stats = await self.scout.qdrant.get_collection_stats()
        print(f"\n📈 QDrant Collection Stats:")
        print(f"   Total agents in DB: {collection_stats.get('total_agents', 'N/A')}")
        print(f"   Vector dimension: {collection_stats.get('vector_dimension', 'N/A')}")
        print(f"   Distance metric: {collection_stats.get('distance_metric', 'N/A')}")
    
    async def demo_semantic_search(self):
        """Demonstrate semantic search capabilities."""
        print("\n🎯 DEMO: Semantic Search")
        print("=" * 50)
        
        search_queries = [
            "React developer with TypeScript experience",
            "Backend engineer skilled in microservices and APIs", 
            "Mobile developer for cross-platform applications",
            "Security expert with penetration testing skills",
            "Data scientist with machine learning expertise"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching: '{query}'")
            
            results = await self.scout.qdrant.search_similar_agents(
                query_text=query,
                limit=3,
                score_threshold=0.6
            )
            
            if results:
                print(f"   Found {len(results)} matches:")
                for agent, score in results:
                    print(f"   • {agent.name} ({agent.role}) - Score: {score:.3f}")
                    print(f"     Domain: {agent.domain} | Capabilities: {', '.join(agent.capabilities[:3])}...")
            else:
                print("   No matches found above threshold")
    
    async def demo_complete_scouting(self):
        """Demonstrate complete scouting workflow."""
        print("\n🕵️ DEMO: Complete Scouting Workflow")
        print("=" * 50)
        
        # Create strategy
        strategy = self.create_sample_strategy()
        print(f"📋 Strategy: {strategy.title}")
        print(f"🎯 Goal: {strategy.goal_description}")
        print(f"👥 Total roles required: {len(strategy.roles)}")
        
        # Create scout input
        library_path = self.create_sample_agent_library()
        scout_input = TalentScoutInput(
            goal="Demo complete scouting workflow",
            strategy_document=strategy,
            agent_libraries=[library_path],
            force_reindex=False
        )
        
        # Process strategy
        print("\n🔄 Processing strategy document...")
        start_time = time.time()
        
        result = await self.scout.process(scout_input)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Display results
        print(f"\n📊 SCOUTING RESULTS")
        print(f"Status: {result.status}")
        print(f"Processing time: {processing_time:.1f}ms")
        
        report = result.scouting_report
        
        print(f"\n📈 COVERAGE ANALYSIS")
        print(f"Total roles: {report.total_roles}")
        print(f"Filled roles: {report.filled_roles}")
        print(f"Vacant roles: {report.vacant_roles}")
        print(f"Overall coverage: {report.overall_coverage:.1%}")
        print(f"Reuse efficiency: {report.reuse_efficiency:.1%}")
        
        # Show matches
        if report.matches:
            print(f"\n✅ SUCCESSFUL MATCHES ({len(report.matches)})")
            for i, match in enumerate(report.matches, 1):
                print(f"{i}. {match.role_requirement.role_name}")
                print(f"   → Matched with: {match.agent.name}")
                print(f"   → Overall score: {match.overall_score:.3f} ({match.match_confidence} confidence)")
                print(f"   → Reasoning: {match.match_reasoning}")
                if match.adaptation_needed:
                    print(f"   → ⚠️  Adaptation needed: {'; '.join(match.adaptation_suggestions)}")
                print()
        
        # Show vacant positions
        if report.vacant_positions:
            print(f"\n❌ VACANT POSITIONS ({len(report.vacant_positions)})")
            for i, vacant in enumerate(report.vacant_positions, 1):
                print(f"{i}. {vacant.role_requirement.role_name}")
                print(f"   → Gap analysis: {vacant.gap_analysis}")
                print(f"   → Creation recommendations:")
                for rec in vacant.creation_recommendations:
                    print(f"     • {rec}")
                if vacant.closest_matches:
                    print(f"   → Closest matches for inspiration:")
                    for match in vacant.closest_matches[:2]:
                        print(f"     • {match.agent.name} (score: {match.overall_score:.3f})")
                print()
        
        return result
    
    async def demo_performance_benchmark(self):
        """Demonstrate performance characteristics."""
        print("\n⚡ DEMO: Performance Benchmark")
        print("=" * 50)
        
        # Create larger sample library for performance testing
        print("Creating extended agent library for performance testing...")
        
        extended_agents = []
        for i in range(20):
            agent_content = f"""# Performance Test Agent {i}
            
**Role:** Test Role {i % 5}
**Description:** Generated agent {i} for performance testing with various capabilities and complex descriptions
**Capabilities:** {', '.join([f'skill-{j}' for j in range(i % 8 + 1)])}
**Tools:** {', '.join([f'tool-{j}' for j in range(i % 5 + 1)])}
**Domain:** domain-{i % 4}
**Complexity:** {['low', 'medium', 'high'][i % 3]}
**Tags:** {', '.join([f'tag-{j}' for j in range(i % 6)])}

## System Prompt
This is a comprehensive system prompt for agent {i} with detailed instructions and multiple paragraphs
to simulate realistic agent documentation. The agent has specialized knowledge in area {i} and can
perform complex tasks related to {', '.join([f'domain-{j}' for j in range(i % 3 + 1)])}.
"""
            
            file_path = Path(self.temp_dir) / f"perf_agent_{i}.md"
            file_path.write_text(agent_content)
            extended_agents.append(str(file_path))
        
        print(f"Created {len(extended_agents)} additional agents for testing")
        
        # Benchmark indexing
        print("\n📊 Benchmarking indexing performance...")
        start_time = time.time()
        
        indexing_stats = await self.scout.index_agent_libraries([self.temp_dir], force_reindex=True)
        
        indexing_time = (time.time() - start_time) * 1000
        
        # Benchmark search
        print("📊 Benchmarking search performance...")
        search_queries = [
            "Senior developer with leadership skills",
            "Data processing and analytics expert", 
            "Frontend specialist with modern frameworks",
            "Infrastructure automation and deployment",
            "Quality assurance and testing professional"
        ]
        
        search_times = []
        total_results = 0
        
        for query in search_queries:
            start_time = time.time()
            results = await self.scout.qdrant.search_similar_agents(
                query_text=query,
                limit=5,
                score_threshold=0.5
            )
            search_time = (time.time() - start_time) * 1000
            search_times.append(search_time)
            total_results += len(results)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY")
        print(f"Total agents indexed: {indexing_stats['agents_indexed']}")
        print(f"Indexing time: {indexing_time:.1f}ms")
        print(f"Average indexing time per agent: {indexing_time / max(indexing_stats['agents_indexed'], 1):.1f}ms")
        print(f"Search queries executed: {len(search_queries)}")
        print(f"Average search time: {avg_search_time:.1f}ms")
        print(f"Total search results: {total_results}")
        print(f"Average results per query: {total_results / len(search_queries):.1f}")
    
    async def interactive_mode(self):
        """Interactive demonstration mode."""
        print("\n🎮 INTERACTIVE MODE")
        print("=" * 50)
        
        # Initialize
        library_path = self.create_sample_agent_library()
        await self.scout.initialize()
        await self.scout.index_agent_libraries([library_path])
        
        print("✅ Enhanced Talent Scout initialized and ready!")
        print("\nAvailable commands:")
        print("  1. search <query>     - Search for agents")
        print("  2. scout <role_name>  - Find matches for a specific role")
        print("  3. stats              - Show collection statistics")
        print("  4. list               - List available sample roles")
        print("  5. help               - Show this help")
        print("  6. quit               - Exit interactive mode")
        
        # Sample roles for quick testing
        sample_roles = {
            "frontend": RoleRequirement(
                role_id="frontend",
                role_name="Frontend Developer",
                description="React developer",
                required_capabilities=["React", "JavaScript"],
                domain="web development",
                complexity_level="medium"
            ),
            "backend": RoleRequirement(
                role_id="backend", 
                role_name="Backend Developer",
                description="API developer",
                required_capabilities=["Node.js", "API Development"],
                domain="web development",
                complexity_level="high"
            ),
            "mobile": RoleRequirement(
                role_id="mobile",
                role_name="Mobile Developer", 
                description="Cross-platform mobile apps",
                required_capabilities=["React Native", "Mobile Development"],
                domain="mobile development",
                complexity_level="medium"
            )
        }
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command.startswith("search "):
                    query = command[7:]
                    results = await self.scout.qdrant.search_similar_agents(query, limit=5)
                    
                    if results:
                        print(f"Found {len(results)} matches for '{query}':")
                        for agent, score in results:
                            print(f"  • {agent.name} ({agent.role}) - Score: {score:.3f}")
                    else:
                        print(f"No matches found for '{query}'")
                
                elif command.startswith("scout "):
                    role_key = command[6:]
                    if role_key in sample_roles:
                        role = sample_roles[role_key]
                        matches = await self.scout.find_matches_for_role(role, limit=3)
                        
                        if matches:
                            print(f"Found {len(matches)} matches for {role.role_name}:")
                            for match in matches:
                                print(f"  • {match.agent.name} - Score: {match.overall_score:.3f}")
                                print(f"    Confidence: {match.match_confidence}")
                                if match.adaptation_needed:
                                    print(f"    Adaptations: {'; '.join(match.adaptation_suggestions)}")
                        else:
                            print(f"No suitable matches found for {role.role_name}")
                    else:
                        print(f"Unknown role. Available: {', '.join(sample_roles.keys())}")
                
                elif command == "stats":
                    stats = await self.scout.qdrant.get_collection_stats()
                    print("Collection Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                
                elif command == "list":
                    print("Available sample roles:")
                    for key, role in sample_roles.items():
                        print(f"  {key}: {role.role_name} - {role.description}")
                
                elif command in ["help", "h"]:
                    print("Available commands:")
                    print("  search <query>     - Search for agents")
                    print("  scout <role_name>  - Find matches for a specific role") 
                    print("  stats              - Show collection statistics")
                    print("  list               - List available sample roles")
                    print("  help               - Show this help")
                    print("  quit               - Exit interactive mode")
                
                elif command in ["quit", "q", "exit"]:
                    print("👋 Goodbye!")
                    break
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main demonstration entry point."""
    import sys
    
    mode = "demo"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    demo = TalentScoutDemo()
    
    print("🔥 Enhanced Talent Scout Demonstration")
    print("=" * 60)
    print("Advanced semantic agent discovery with QDrant vector database")
    print("=" * 60)
    
    try:
        if mode == "interactive":
            await demo.interactive_mode()
        
        elif mode == "benchmark":
            await demo.demo_agent_indexing()
            await demo.demo_performance_benchmark()
        
        else:  # Default demo mode
            await demo.demo_agent_indexing()
            await demo.demo_semantic_search() 
            result = await demo.demo_complete_scouting()
            
            # Save results to file
            output_file = "scouting_demo_results.json"
            with open(output_file, 'w') as f:
                f.write(result.scouting_report.model_dump_json(indent=2))
            
            print(f"\n💾 Detailed results saved to: {output_file}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Enhanced Talent Scout demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())