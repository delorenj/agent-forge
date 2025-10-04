# Task

**Main Goal**:
Implement a CLI interface that passes the provided acceptance criteria

Secondary Goals:
  - Swap out lancedb for QDrant. I have a local instance of qdrant running on port 6333, api key is 'touchmyflappyfoldyholds'
  - Implement the Talent Scout agent

## Test Inputs

**Input 1**
*Simple query*
```sh
agentforge "I need a team fine tuned to convert python scripts to idiomatic rust scripts"
```

**Input 2**
*File for context*
```sh
agentforge -f /path/to/prd.md --agents /path/to/agents/folder/
```

**Input 3**
*Only make a single agent and manually name it*
```sh
agentforge -f /path/to/task.md -n1 --name "Billy Cheemo"
```

**Input 4**
*Force create all agents - skip the cross-check for existing ones and output to  `./agents/` folder*
```sh
agentforge -f /path/to/task.md --force -n3 -o ./agents/
```

**Input 5**
*Name Strategy*:
- domain (i.e. DocumentationExpert)
- real (i.e. random real name Bill Thomlinson)
```sh
agentforge -f /path/to/task.md --auto-name-strategy "[domain|real]"
```

**Input 6**
*Name Strategy*
- custom rule file
```sh
agentforge -f /path/to/task.md --auto-name-rules /path/to/naming-rules.md
```

## Acceptance Criteria
1. All 6 command above are run without error
2. Agents are created successfully with appropriate specializations based on the input context
3. Agent names follow the specified naming strategy (manual, domain-based, real person, or custom rules)
4. Agents are output to the correct directory when `-o` flag is specified
5. The `--force` flag successfully creates agents without checking for existing ones
6. File context is properly parsed and used when `-f` flag is provided
7. The agents folder is properly utilized when `--agents` flag is specified
8. Generated agents are functional and can be imported/used by the system 
