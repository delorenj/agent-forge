# AgentForge CLI Implementation Plan

## 📋 Implementation Roadmap

This document outlines the step-by-step implementation plan for the AgentForge CLI based on the architectural specification in `CLI_ARCHITECTURE.md`.

## 🎯 Priority Matrix

### Phase 1: Foundation (Week 1-2)
**Goal**: Get basic CLI working with simple query mode

| Task | Priority | Effort | Dependencies |
|------|----------|---------|--------------|
| Project setup & TypeScript config | HIGH | 1d | None |
| Commander.js integration | HIGH | 2d | Project setup |
| Basic Python subprocess bridge | HIGH | 3d | Commander.js |
| Simple query processing (Input 1) | HIGH | 2d | Python bridge |

### Phase 2: Context Processing (Week 3-4)  
**Goal**: File context and agent library integration

| Task | Priority | Effort | Dependencies |
|------|----------|---------|--------------|
| File context processor | HIGH | 3d | Foundation |
| Agent library scanner | HIGH | 2d | File processor |
| Context-aware mode (Input 2) | HIGH | 2d | Library scanner |
| Manual control mode (Input 3) | MEDIUM | 1d | Context processing |

### Phase 3: Advanced Features (Week 5-6)
**Goal**: Complete all acceptance criteria

| Task | Priority | Effort | Dependencies |
|------|----------|---------|--------------|
| Force creation mode (Input 4) | HIGH | 2d | Context processing |
| Domain naming strategy (Input 5) | HIGH | 2d | Force creation |
| Real person naming strategy | MEDIUM | 1d | Domain naming |
| Custom rules engine (Input 6) | MEDIUM | 3d | Naming strategies |

### Phase 4: Polish & Testing (Week 7-8)
**Goal**: Production-ready CLI with comprehensive testing

| Task | Priority | Effort | Dependencies |
|------|----------|---------|--------------|
| Rich terminal UI (Ink/React) | MEDIUM | 4d | All features |
| Comprehensive error handling | HIGH | 2d | Features complete |
| Test suite (unit + integration) | HIGH | 3d | Error handling |
| Documentation & examples | LOW | 2d | Testing |

## 🏗️ Detailed Implementation Steps

### Step 1: Project Foundation

#### 1.1 Initialize TypeScript Project
```bash
# Create package.json with proper dependencies
npm init -y
npm install --save typescript @types/node commander chalk cross-spawn zod
npm install --save-dev @types/jest jest ts-jest ts-node nodemon
npm install --save ink react @types/react
```

#### 1.2 Configure TypeScript (`tsconfig.json`)
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "CommonJS",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

#### 1.3 Create Basic CLI Entry Point (`src/index.ts`)
```typescript
#!/usr/bin/env node

import { Command } from 'commander';
import { CLIOptions, parseCommand } from './cli/parser';
import { AgentForgeEngine } from './engine/agentforge';

async function main() {
  const program = new Command();
  
  program
    .name('agentforge')
    .description('AgentForge CLI - Meta-agent system for creating specialized agent teams')
    .version('1.0.0');

  // Simple query mode
  program
    .argument('[query]', 'Direct query for agent team creation')
    .option('-f, --file <path>', 'Context file path')
    .option('--agents <path>', 'Agent library directory')
    .option('-n <count>', 'Number of agents to create', parseInt)
    .option('--name <name>', 'Manual agent name')
    .option('--force', 'Skip existing agent checks')
    .option('-o, --output <path>', 'Output directory')
    .option('--auto-name-strategy <strategy>', 'Naming strategy (domain|real)')
    .option('--auto-name-rules <path>', 'Custom naming rules file')
    .action(async (query, options) => {
      const parsedOptions = parseCommand(query, options);
      const engine = new AgentForgeEngine();
      await engine.execute(parsedOptions);
    });

  await program.parseAsync();
}

main().catch((error) => {
  console.error('AgentForge CLI Error:', error.message);
  process.exit(1);
});
```

### Step 2: Command Parsing & Validation

#### 2.1 Create Parser (`src/cli/parser.ts`)
```typescript
import { z } from 'zod';

const CLIOptionsSchema = z.object({
  query: z.string().optional(),
  file: z.string().optional(),
  agents: z.string().optional(),
  count: z.number().min(1).max(20).optional(),
  name: z.string().optional(), 
  force: z.boolean().default(false),
  output: z.string().optional(),
  autoNameStrategy: z.enum(['domain', 'real']).optional(),
  autoNameRules: z.string().optional(),
});

export type CLIOptions = z.infer<typeof CLIOptionsSchema>;

export interface ParsedCommand {
  mode: 'simple' | 'context' | 'manual' | 'force' | 'strategy' | 'custom';
  options: CLIOptions;
  validationResult: ValidationResult;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function parseCommand(query: string, options: any): ParsedCommand {
  // Validate options
  const validationResult = CLIOptionsSchema.safeParse({
    query,
    ...options,
  });
  
  if (!validationResult.success) {
    return {
      mode: 'simple',
      options: options as CLIOptions,
      validationResult: {
        valid: false,
        errors: validationResult.error.errors.map(e => e.message),
        warnings: [],
      },
    };
  }

  const parsedOptions = validationResult.data;
  
  // Determine mode based on options
  const mode = determineMode(parsedOptions);
  
  return {
    mode,
    options: parsedOptions,
    validationResult: {
      valid: true,
      errors: [],
      warnings: generateWarnings(parsedOptions),
    },
  };
}

function determineMode(options: CLIOptions): ParsedCommand['mode'] {
  if (options.autoNameRules) return 'custom';
  if (options.autoNameStrategy) return 'strategy';
  if (options.force) return 'force';
  if (options.name && options.count === 1) return 'manual';
  if (options.file || options.agents) return 'context';
  return 'simple';
}
```

### Step 3: Python Integration Bridge

#### 3.1 Create Integration Bridge (`src/integration/bridge.ts`)
```typescript
import { spawn, ChildProcess } from 'child_process';
import { join } from 'path';
import { CLIOptions } from '../cli/parser';

export interface PythonBackendRequest {
  mode: string;
  query?: string;
  fileContext?: FileContext;
  agentLibraryPath?: string;
  forceCreate?: boolean;
  agentCount?: number;
  namingStrategy?: NamingStrategy;
  outputPath?: string;
}

export interface PythonBackendResponse {
  success: boolean;
  agents: AgentDefinition[];
  strategyDocument: string;
  scoutingReport: string;
  rosterDocumentation: string;
  errors: string[];
  metadata: {
    processingTime: number;
    agentsCreated: number;
    agentsReused: number;
  };
}

export class PythonIntegrationBridge {
  private pythonPath: string;
  private agentforgePath: string;

  constructor(pythonPath = 'python', agentforgePath = './') {
    this.pythonPath = pythonPath;
    this.agentforgePath = agentforgePath;
  }

  async execute(options: CLIOptions): Promise<PythonBackendResponse> {
    const request = this.buildRequest(options);
    const pythonScript = join(this.agentforgePath, 'cli_integration.py');
    
    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, [pythonScript], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: this.agentforgePath,
      });

      let stdout = '';
      let stderr = '';

      process.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const response = JSON.parse(stdout) as PythonBackendResponse;
            resolve(response);
          } catch (error) {
            reject(new Error(`Failed to parse Python response: ${error}`));
          }
        } else {
          reject(new Error(`Python process failed with code ${code}: ${stderr}`));
        }
      });

      // Send request to Python process
      process.stdin?.write(JSON.stringify(request));
      process.stdin?.end();
    });
  }

  private buildRequest(options: CLIOptions): PythonBackendRequest {
    return {
      mode: this.determineMode(options),
      query: options.query,
      fileContext: options.file ? { filePath: options.file } : undefined,
      agentLibraryPath: options.agents,
      forceCreate: options.force,
      agentCount: options.count,
      namingStrategy: options.autoNameStrategy ? {
        type: options.autoNameStrategy,
        customRulesPath: options.autoNameRules,
      } : undefined,
      outputPath: options.output,
    };
  }

  private determineMode(options: CLIOptions): string {
    if (options.autoNameRules) return 'custom_rules';
    if (options.autoNameStrategy) return 'auto_naming';
    if (options.force) return 'force_create';
    if (options.name) return 'manual_name';
    if (options.file) return 'file_context';
    return 'simple_query';
  }
}
```

### Step 4: Python CLI Integration Script

#### 4.1 Create Python Bridge Script (`cli_integration.py`)
```python
#!/usr/bin/env python3
"""
CLI Integration Bridge for AgentForge
Receives JSON requests from TypeScript CLI and processes them through AgentForge
"""

import json
import sys
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

# Import AgentForge components
from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel
from agents.systems_analyst import SystemsAnalyst
from agents.agent_developer import AgentDeveloper
from agents.integration_architect import IntegrationArchitect

class CLIIntegrationBridge:
    """Bridge between TypeScript CLI and Python AgentForge backend."""
    
    def __init__(self):
        self.engineering_manager = EngineeringManager()
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process CLI request and return formatted response."""
        try:
            mode = request.get('mode', 'simple_query')
            
            if mode == 'simple_query':
                return await self._handle_simple_query(request)
            elif mode == 'file_context':
                return await self._handle_file_context(request)
            elif mode == 'force_create':
                return await self._handle_force_create(request)
            elif mode == 'manual_name':
                return await self._handle_manual_name(request)
            elif mode == 'auto_naming':
                return await self._handle_auto_naming(request)
            elif mode == 'custom_rules':
                return await self._handle_custom_rules(request)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            return {
                'success': False,
                'agents': [],
                'strategyDocument': '',
                'scoutingReport': '',
                'rosterDocumentation': '',
                'errors': [str(e)],
                'metadata': {
                    'processingTime': 0,
                    'agentsCreated': 0,
                    'agentsReused': 0,
                }
            }
    
    async def _handle_simple_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple query mode."""
        query = request.get('query', '')
        
        # Create InputGoal from query
        goal = InputGoal(
            goal_description=query,
            domain='general',
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        # Process through AgentForge
        result = await self.engineering_manager.process(goal)
        
        return self._format_response(result, request)
    
    async def _handle_file_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file context mode."""
        file_path = request.get('fileContext', {}).get('filePath', '')
        agents_path = request.get('agentLibraryPath')
        
        # Read and analyze file context
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            raise FileNotFoundError(f"Context file not found: {file_path}")
        
        # Create InputGoal with file context
        goal = InputGoal(
            goal_description=f"Process requirements from file: {file_path}",
            domain='file_analysis',
            complexity_level=self._determine_complexity(content),
            existing_resources={'file_content': content, 'agents_path': agents_path}
        )
        
        result = await self.engineering_manager.process(goal)
        
        return self._format_response(result, request)
    
    async def _handle_force_create(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle force create mode - skip existing agent checks."""
        file_path = request.get('fileContext', {}).get('filePath', '')
        agent_count = request.get('agentCount', 3)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        goal = InputGoal(
            goal_description=f"Create exactly {agent_count} new agents based on: {content[:200]}...",
            domain='forced_creation',
            complexity_level=ComplexityLevel.HIGH,
            existing_resources={
                'file_content': content,
                'force_create': True,
                'agent_count': agent_count
            }
        )
        
        result = await self.engineering_manager.process(goal)
        
        return self._format_response(result, request)
    
    def _determine_complexity(self, content: str) -> ComplexityLevel:
        """Determine complexity level based on content analysis."""
        if len(content) > 10000 or 'enterprise' in content.lower():
            return ComplexityLevel.ENTERPRISE
        elif len(content) > 5000 or 'complex' in content.lower():
            return ComplexityLevel.HIGH
        elif len(content) > 2000:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _format_response(self, result, request: Dict[str, Any]) -> Dict[str, Any]:
        """Format AgentForge result for CLI consumption."""
        return {
            'success': True,
            'agents': [
                {
                    'name': agent.get('name', 'UnnamedAgent'),
                    'role': agent.get('role', 'General'),
                    'specialization': agent.get('specialization', ''),
                    'prompt': agent.get('prompt', ''),
                    'filePath': agent.get('filePath', ''),
                }
                for agent in result.new_agents
            ],
            'strategyDocument': result.strategy_document,
            'scoutingReport': result.scouting_report,
            'rosterDocumentation': result.roster_documentation,
            'errors': [],
            'metadata': {
                'processingTime': 0,  # TODO: Add timing
                'agentsCreated': len(result.new_agents),
                'agentsReused': len(result.existing_agents),
            }
        }

async def main():
    """Main CLI integration entry point."""
    try:
        # Read JSON request from stdin
        request_data = sys.stdin.read()
        request = json.loads(request_data)
        
        # Process request
        bridge = CLIIntegrationBridge()
        response = await bridge.process_request(request)
        
        # Output JSON response
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        error_response = {
            'success': False,
            'agents': [],
            'strategyDocument': '',
            'scoutingReport': '',
            'rosterDocumentation': '',
            'errors': [str(e)],
            'metadata': {
                'processingTime': 0,
                'agentsCreated': 0,
                'agentsReused': 0,
            }
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 5: Main Engine Implementation

#### 5.1 Create AgentForge Engine (`src/engine/agentforge.ts`)
```typescript
import { CLIOptions, ParsedCommand } from '../cli/parser';
import { PythonIntegrationBridge, PythonBackendResponse } from '../integration/bridge';
import { FileContextProcessor } from '../context/processor';
import { AgentNamingEngine } from '../naming/engine';
import { OutputManager } from '../output/manager';
import { UIManager } from '../ui/manager';

export class AgentForgeEngine {
  private bridge: PythonIntegrationBridge;
  private contextProcessor: FileContextProcessor;
  private namingEngine: AgentNamingEngine;
  private outputManager: OutputManager;
  private uiManager: UIManager;

  constructor() {
    this.bridge = new PythonIntegrationBridge();
    this.contextProcessor = new FileContextProcessor();
    this.namingEngine = new AgentNamingEngine();
    this.outputManager = new OutputManager();
    this.uiManager = new UIManager();
  }

  async execute(command: ParsedCommand): Promise<void> {
    // Show initial UI
    await this.uiManager.showWelcome();
    
    try {
      // Validate command
      if (!command.validationResult.valid) {
        throw new Error(`Validation failed: ${command.validationResult.errors.join(', ')}`);
      }

      // Show progress
      const steps = this.getStepsForMode(command.mode);
      await this.uiManager.startProgress(steps);

      // Process file context if needed
      if (command.options.file) {
        await this.uiManager.updateProgress('Processing file context...');
        const context = await this.contextProcessor.processFile(command.options.file);
        // Add context to options
      }

      // Execute through Python backend
      await this.uiManager.updateProgress('Creating agent team...');
      const response = await this.bridge.execute(command.options);

      if (!response.success) {
        throw new Error(`Backend failed: ${response.errors.join(', ')}`);
      }

      // Apply naming strategies
      if (command.options.autoNameStrategy || command.options.name) {
        await this.uiManager.updateProgress('Applying naming strategies...');
        await this.applyNamingStrategies(response, command.options);
      }

      // Save agents to output directory
      await this.uiManager.updateProgress('Saving agents...');
      await this.outputManager.saveAgents(response.agents, command.options.output);

      // Show success
      await this.uiManager.showSuccess(response);

    } catch (error) {
      await this.uiManager.showError(error as Error);
      throw error;
    }
  }

  private getStepsForMode(mode: string): string[] {
    const baseSteps = ['Validating options', 'Initializing backend'];
    
    switch (mode) {
      case 'simple':
        return [...baseSteps, 'Processing query', 'Creating agents', 'Saving results'];
      case 'context':
        return [...baseSteps, 'Processing file context', 'Loading agent library', 'Creating team', 'Saving results'];
      case 'force':
        return [...baseSteps, 'Processing context', 'Force creating agents', 'Applying names', 'Saving results'];
      default:
        return [...baseSteps, 'Processing request', 'Saving results'];
    }
  }

  private async applyNamingStrategies(response: PythonBackendResponse, options: CLIOptions): Promise<void> {
    // Apply naming strategies to agents
    if (options.name && response.agents.length === 1) {
      response.agents[0].name = options.name;
    } else if (options.autoNameStrategy) {
      for (const agent of response.agents) {
        const newName = await this.namingEngine.generateName(
          { type: options.autoNameStrategy },
          agent.role
        );
        agent.name = newName;
      }
    }
  }
}
```

## 🧪 Testing Implementation Plan

### Test Structure
```
tests/
├── unit/                   # Unit tests for individual components
│   ├── cli/
│   │   ├── parser.test.ts
│   │   └── validator.test.ts
│   ├── context/
│   │   └── processor.test.ts
│   ├── naming/
│   │   └── engine.test.ts
│   └── integration/
│       └── bridge.test.ts
├── integration/            # Integration tests
│   ├── python-bridge.test.ts
│   └── end-to-end.test.ts
└── acceptance/             # Acceptance criteria tests
    ├── input-1.test.ts     # Simple query
    ├── input-2.test.ts     # File + agents folder
    ├── input-3.test.ts     # Single agent + manual name
    ├── input-4.test.ts     # Force create + output dir
    ├── input-5.test.ts     # Auto-name strategy
    └── input-6.test.ts     # Custom rules
```

### Key Test Cases
1. **All 6 acceptance criteria commands execute successfully**
2. **File path validation and error handling**
3. **Python backend communication and error recovery**
4. **Naming strategy validation and application**
5. **Output directory creation and file saving**
6. **Cross-platform compatibility**

## 📦 Package Configuration

### Final `package.json`
```json
{
  "name": "agentforge-cli",
  "version": "1.0.0",
  "description": "CLI for AgentForge meta-agent system",
  "main": "dist/index.js",
  "bin": {
    "agentforge": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "ts-node src/index.ts",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:acceptance": "jest tests/acceptance",
    "lint": "eslint src/**/*.ts",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "commander": "^9.4.1",
    "chalk": "^4.1.2",
    "cross-spawn": "^7.0.3",
    "zod": "^3.21.4",
    "ink": "^3.2.0",
    "react": "^17.0.2",
    "cli-progress": "^3.12.0",
    "cosmiconfig": "^7.0.1"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^18.0.0",
    "@types/react": "^17.0.0",
    "@types/cross-spawn": "^6.0.2",
    "@types/cli-progress": "^3.11.0",
    "jest": "^29.0.0",
    "ts-jest": "^29.0.0",
    "ts-node": "^10.9.0",
    "@types/jest": "^29.0.0",
    "eslint": "^8.0.0",
    "@typescript-eslint/eslint-plugin": "^5.0.0",
    "@typescript-eslint/parser": "^5.0.0"
  },
  "engines": {
    "node": ">=16.0.0"
  },
  "keywords": ["cli", "agents", "ai", "agentforge", "automation"],
  "author": "AgentForge Team",
  "license": "MIT"
}
```

This implementation plan provides a comprehensive roadmap for building the AgentForge CLI that meets all 8 acceptance criteria while maintaining high code quality and user experience standards.