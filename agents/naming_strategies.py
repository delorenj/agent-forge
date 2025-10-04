"""
Naming Strategies for AgentForge CLI

This module provides various naming strategies for automatically generating
agent names based on different approaches:
- Domain-based naming (e.g., DocumentationExpert, WebDeveloper)
- Real person names (e.g., Bill Thomlinson, Sarah Chen)
- Custom rules from configuration files
"""

import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import yaml
from pydantic import BaseModel, Field


class NamingStrategy(ABC):
    """Abstract base class for agent naming strategies."""
    
    @abstractmethod
    def generate_name(self, role: str, domain: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a name based on the role, domain, and context."""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> str:
        """Return the type of naming strategy."""
        pass


class DomainNamingStrategy(NamingStrategy):
    """
    Domain-based naming strategy that creates descriptive names
    like DocumentationExpert, WebDeveloper, DataAnalyst, etc.
    """
    
    def __init__(self):
        self.domain_prefixes = {
            "web development": ["Web", "Frontend", "Backend", "Fullstack"],
            "data science": ["Data", "ML", "Analytics", "AI"],
            "mobile development": ["Mobile", "iOS", "Android", "App"],
            "devops": ["DevOps", "Cloud", "Infrastructure", "Platform"],
            "security": ["Security", "Cyber", "Penetration", "Compliance"],
            "design": ["UI", "UX", "Visual", "Product"],
            "testing": ["QA", "Test", "Automation", "Performance"],
            "documentation": ["Documentation", "Technical", "Content"],
            "management": ["Project", "Product", "Team", "Scrum"],
            "architecture": ["System", "Solution", "Enterprise", "Cloud"],
            "general": ["Technical", "Senior", "Lead", "Principal"]
        }
        
        self.role_suffixes = {
            "developer": "Developer",
            "engineer": "Engineer", 
            "analyst": "Analyst",
            "architect": "Architect",
            "manager": "Manager",
            "tester": "Tester",
            "designer": "Designer",
            "researcher": "Researcher",
            "writer": "Writer",
            "reviewer": "Reviewer",
            "specialist": "Specialist",
            "expert": "Expert",
            "consultant": "Consultant",
            "lead": "Lead",
            "coordinator": "Coordinator"
        }
    
    def generate_name(self, role: str, domain: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a domain-based name like 'WebDeveloper' or 'DataAnalyst'."""
        # Clean and normalize inputs
        role_clean = role.lower().strip()
        domain_clean = domain.lower().strip()
        
        # Get domain prefix
        domain_prefix = self._get_domain_prefix(domain_clean)
        
        # Get role suffix
        role_suffix = self._get_role_suffix(role_clean)
        
        # Handle special cases from context
        if context:
            # Check for specific technologies or specializations
            if "capabilities" in context:
                capabilities = [cap.lower() for cap in context["capabilities"]]
                
                # Technology-specific prefixes
                if any(tech in capabilities for tech in ["react", "vue", "angular"]):
                    domain_prefix = "Frontend"
                elif any(tech in capabilities for tech in ["node", "express", "django", "flask"]):
                    domain_prefix = "Backend"
                elif any(tech in capabilities for tech in ["tensorflow", "pytorch", "scikit"]):
                    domain_prefix = "ML"
                elif any(tech in capabilities for tech in ["kubernetes", "docker", "aws", "azure"]):
                    domain_prefix = "Cloud"
        
        # Combine prefix and suffix
        name = f"{domain_prefix}{role_suffix}"
        
        # Handle edge cases where name might be too generic
        if name in ["TechnicalExpert", "GeneralSpecialist"]:
            name = f"{domain_prefix}Consultant"
        
        return name
    
    def _get_domain_prefix(self, domain: str) -> str:
        """Get appropriate domain prefix."""
        # Direct mapping
        if domain in self.domain_prefixes:
            prefixes = self.domain_prefixes[domain]
            return prefixes[0]  # Use first/primary prefix
        
        # Fuzzy matching for variations
        for key, prefixes in self.domain_prefixes.items():
            if any(word in domain for word in key.split()):
                return prefixes[0]
        
        # Keyword-based inference
        if any(word in domain for word in ["web", "frontend", "ui", "react", "vue"]):
            return "Web"
        elif any(word in domain for word in ["data", "analytics", "ml", "ai", "machine"]):
            return "Data"
        elif any(word in domain for word in ["mobile", "app", "ios", "android"]):
            return "Mobile"
        elif any(word in domain for word in ["cloud", "devops", "infrastructure"]):
            return "Cloud"
        elif any(word in domain for word in ["security", "cyber"]):
            return "Security"
        
        return "Technical"
    
    def _get_role_suffix(self, role: str) -> str:
        """Get appropriate role suffix."""
        # Direct mapping
        for key, suffix in self.role_suffixes.items():
            if key in role:
                return suffix
        
        # Pattern matching
        if any(word in role for word in ["dev", "program", "code"]):
            return "Developer"
        elif any(word in role for word in ["test", "qa", "quality"]):
            return "Tester"
        elif any(word in role for word in ["design", "ui", "ux"]):
            return "Designer"
        elif any(word in role for word in ["manage", "pm", "product"]):
            return "Manager"
        elif any(word in role for word in ["architect", "system", "solution"]):
            return "Architect"
        elif any(word in role for word in ["analyst", "analyze", "research"]):
            return "Analyst"
        
        return "Expert"
    
    def get_strategy_type(self) -> str:
        return "domain"


class RealNamingStrategy(NamingStrategy):
    """
    Real person naming strategy that generates realistic names
    like 'Bill Thomlinson', 'Sarah Chen', 'Alex Rodriguez', etc.
    """
    
    def __init__(self):
        self.first_names = {
            "male": [
                "Alex", "Bill", "Chris", "David", "Eric", "Frank", "George", "Henry",
                "Ian", "Jack", "Kevin", "Luke", "Mike", "Nathan", "Oliver", "Paul",
                "Quinn", "Robert", "Sam", "Tom", "Victor", "William", "Xavier", "Zach"
            ],
            "female": [
                "Alice", "Beth", "Carol", "Diana", "Emma", "Fiona", "Grace", "Hannah",
                "Iris", "Jane", "Kate", "Lisa", "Mary", "Nina", "Olivia", "Priya",
                "Rachel", "Sarah", "Tina", "Uma", "Victoria", "Wendy", "Xylia", "Zoe"
            ],
            "neutral": [
                "Adrian", "Blake", "Cameron", "Drew", "Elliot", "Finley", "Gray",
                "Harper", "Indigo", "Jordan", "Kit", "Lane", "Morgan", "Nova",
                "Oakley", "Parker", "Quinn", "River", "Sage", "Taylor"
            ]
        }
        
        self.last_names = [
            "Anderson", "Brown", "Chen", "Davis", "Evans", "Fisher", "Garcia",
            "Harris", "Johnson", "Kim", "Lee", "Miller", "Nelson", "O'Connor",
            "Parker", "Rodriguez", "Smith", "Taylor", "Thompson", "Wilson",
            "Zhang", "Patel", "Martinez", "Williams", "Jones", "Kumar", "Singh",
            "Ahmed", "Nakamura", "Schmidt", "Rossi", "Dubois", "Kowalski",
            "Petrov", "Andersson", "Nielsen", "O'Brien", "MacDonald", "Müller"
        ]
        
        self.name_pools = []
        self._initialize_name_pools()
    
    def _initialize_name_pools(self):
        """Initialize combined name pools for random selection."""
        all_first_names = []
        for gender_names in self.first_names.values():
            all_first_names.extend(gender_names)
        
        # Create all possible combinations
        for first in all_first_names:
            for last in self.last_names:
                self.name_pools.append(f"{first} {last}")
        
        # Shuffle for better randomness
        random.shuffle(self.name_pools)
    
    def generate_name(self, role: str, domain: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a realistic person name."""
        # For consistency in testing/demos, we can use a seed based on role+domain
        # This ensures the same role+domain combo gets the same name
        seed_string = f"{role}_{domain}".lower().replace(" ", "_")
        seed = hash(seed_string) % len(self.name_pools)
        
        # Use deterministic selection for consistency
        name = self.name_pools[seed]
        
        # Optional: Add professional titles based on seniority/context
        if context and context.get("seniority") == "senior":
            # Could prefix with "Dr." or "Prof." for senior roles
            pass
        
        return name
    
    def get_strategy_type(self) -> str:
        return "real"


class CustomRulesStrategy(NamingStrategy):
    """
    Custom rules-based naming strategy that loads naming rules
    from configuration files (JSON, YAML, or Markdown).
    """
    
    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        self.rules = custom_rules or {}
        self.templates = []
        self.mappings = {}
        self.patterns = {}
        self._process_rules()
    
    @classmethod
    def from_file(cls, rules_file: Path) -> 'CustomRulesStrategy':
        """Create CustomRulesStrategy from a rules file."""
        try:
            content = rules_file.read_text(encoding='utf-8')
            
            # Try to parse based on file extension
            if rules_file.suffix.lower() == '.json':
                rules = json.loads(content)
            elif rules_file.suffix.lower() in ['.yaml', '.yml']:
                rules = yaml.safe_load(content)
            elif rules_file.suffix.lower() in ['.md', '.txt']:
                # Parse markdown/text format
                rules = cls._parse_text_rules(content)
            else:
                raise ValueError(f"Unsupported rules file format: {rules_file.suffix}")
            
            return cls(rules)
            
        except Exception as e:
            raise ValueError(f"Failed to load rules from {rules_file}: {e}")
    
    @staticmethod
    def _parse_text_rules(content: str) -> Dict[str, Any]:
        """Parse rules from markdown/text content."""
        rules = {}
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Section headers
            if line.startswith('#'):
                current_section = line.lstrip('#').strip().lower()
                if current_section not in rules:
                    rules[current_section] = {}
                continue
            
            # Key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if current_section:
                    rules[current_section][key] = value
                else:
                    rules[key] = value
        
        return rules
    
    def _process_rules(self):
        """Process loaded rules into usable formats."""
        # Handle fixed name (for single agent scenarios)
        if "fixed_name" in self.rules:
            self.fixed_name = self.rules["fixed_name"]
        
        # Handle naming templates
        if "templates" in self.rules:
            if isinstance(self.rules["templates"], list):
                self.templates = self.rules["templates"]
            elif isinstance(self.rules["templates"], str):
                self.templates = [self.rules["templates"]]
        
        # Handle role/domain mappings
        if "mappings" in self.rules:
            self.mappings = self.rules["mappings"]
        
        # Handle pattern-based rules
        if "patterns" in self.rules:
            self.patterns = self.rules["patterns"]
        
        # Default templates if none provided
        if not self.templates:
            self.templates = [
                "{role_title} {domain_prefix}",
                "{domain_prefix} {role_title}",
                "Senior {role_title}",
                "{role_title} Expert"
            ]
    
    def generate_name(self, role: str, domain: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate name based on custom rules."""
        # Fixed name override (for single agent scenarios)
        if hasattr(self, 'fixed_name'):
            return self.fixed_name
        
        # Try exact mapping first
        mapping_key = f"{role.lower()}_{domain.lower()}"
        if mapping_key in self.mappings:
            return self.mappings[mapping_key]
        
        # Try role-only mapping
        if role.lower() in self.mappings:
            return self.mappings[role.lower()]
        
        # Try domain-only mapping
        if domain.lower() in self.mappings:
            return self.mappings[domain.lower()]
        
        # Use pattern matching
        for pattern, template in self.patterns.items():
            if re.search(pattern, f"{role} {domain}", re.IGNORECASE):
                return self._apply_template(template, role, domain, context)
        
        # Fall back to templates
        template = random.choice(self.templates)
        return self._apply_template(template, role, domain, context)
    
    def _apply_template(self, template: str, role: str, domain: str, context: Optional[Dict[str, Any]]) -> str:
        """Apply a naming template with variable substitution."""
        # Prepare variables for template
        variables = {
            "role": role,
            "domain": domain,
            "role_title": self._title_case(role),
            "domain_prefix": self._get_domain_prefix(domain),
            "role_clean": re.sub(r'[^a-zA-Z0-9]', '', role),
            "domain_clean": re.sub(r'[^a-zA-Z0-9]', '', domain)
        }
        
        # Add context variables if available
        if context:
            variables.update(context)
        
        # Apply template
        try:
            name = template.format(**variables)
            return self._clean_name(name)
        except KeyError as e:
            # Fall back to simple format if template variable is missing
            return f"{self._title_case(role)} {self._get_domain_prefix(domain)}"
    
    def _title_case(self, text: str) -> str:
        """Convert text to title case, handling common abbreviations."""
        # Common abbreviations that should stay uppercase
        abbreviations = ["AI", "ML", "UI", "UX", "API", "DB", "QA", "CI", "CD", "DevOps"]
        
        words = text.split()
        result = []
        
        for word in words:
            if word.upper() in abbreviations:
                result.append(word.upper())
            else:
                result.append(word.capitalize())
        
        return " ".join(result)
    
    def _get_domain_prefix(self, domain: str) -> str:
        """Get a clean domain prefix."""
        domain_mappings = {
            "web development": "Web",
            "data science": "Data",
            "mobile development": "Mobile", 
            "devops": "DevOps",
            "security": "Security",
            "design": "Design"
        }
        
        return domain_mappings.get(domain.lower(), domain.title())
    
    def _clean_name(self, name: str) -> str:
        """Clean up the generated name."""
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def get_strategy_type(self) -> str:
        return "custom"


def create_naming_strategy(
    strategy_type: Optional[str] = None,
    rules_file: Optional[str] = None,
    manual_name: Optional[str] = None,
    custom_rules: Optional[Dict[str, Any]] = None
) -> NamingStrategy:
    """
    Factory function to create appropriate naming strategy.
    
    Args:
        strategy_type: Type of strategy ("domain" or "real")
        rules_file: Path to custom rules file
        manual_name: Manual name (creates custom strategy with fixed name)
        custom_rules: Direct custom rules dictionary
    
    Returns:
        Appropriate NamingStrategy instance
    """
    # Manual name takes precedence
    if manual_name:
        return CustomRulesStrategy({"fixed_name": manual_name})
    
    # Custom rules file
    if rules_file:
        rules_path = Path(rules_file)
        return CustomRulesStrategy.from_file(rules_path)
    
    # Direct custom rules
    if custom_rules:
        return CustomRulesStrategy(custom_rules)
    
    # Strategy type
    if strategy_type:
        if strategy_type.lower() == "domain":
            return DomainNamingStrategy()
        elif strategy_type.lower() == "real":
            return RealNamingStrategy()
    
    # Default to domain-based naming
    return DomainNamingStrategy()


# Example usage and testing
if __name__ == "__main__":
    # Test domain naming strategy
    print("=== Domain Naming Strategy ===")
    domain_strategy = DomainNamingStrategy()
    
    test_cases = [
        ("developer", "web development"),
        ("analyst", "data science"),
        ("architect", "cloud infrastructure"),
        ("tester", "mobile development"),
        ("designer", "user experience")
    ]
    
    for role, domain in test_cases:
        name = domain_strategy.generate_name(role, domain)
        print(f"{role} + {domain} = {name}")
    
    print("\n=== Real Naming Strategy ===")
    real_strategy = RealNamingStrategy()
    
    for role, domain in test_cases:
        name = real_strategy.generate_name(role, domain)
        print(f"{role} + {domain} = {name}")
    
    print("\n=== Custom Rules Strategy ===")
    custom_rules = {
        "templates": [
            "{role_title} Specialist",
            "Senior {role_title}",
            "{domain_prefix} {role_title}"
        ],
        "mappings": {
            "developer_web development": "Full Stack Developer",
            "tester": "QA Engineer"
        }
    }
    
    custom_strategy = CustomRulesStrategy(custom_rules)
    
    for role, domain in test_cases:
        name = custom_strategy.generate_name(role, domain)
        print(f"{role} + {domain} = {name}")