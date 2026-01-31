# Sourcing Employees and Contractors for 33GOD

This repo is currently called `agent-forge` but i'm contemplating combining this with Yi, or making agent-forge a part of Yi either by merging the functionality into it or through composition.

## How I Envision This Component Subsystem

- 33GOD will have an employee roster (essentially a database of Yi nodes) that can be hired and assigned to projects.

- Flume is the layer above [Yi](../../yi/trunk-main) that defines the org chart. It gives structure to the Yi nodes. It defines who reports to whom, what projects they are assigned to, and what their roles are.

- In 33GOD the Flume topology emulates a corporate structure and is represented in a data structure called the [Flume](../../flume/trunk-main) Tree.

## Interface

- Define a skillset: Builds a skill package from a pool of curated claude skills. The skill package deployed to a path called the skill vault. Agents are build by symlinking to skill packages in the skill vault. Any required mcp servers are also symlinked to the agent path.

- Create an agent: Combines a skillset, mcp servers, and context (qdrant collection, files, links) into an agent definition. The agent definition is stored in the agent vault.

- Hire an employee: Combines an agent definition with a memory framework to create a Yi node that can be assigned to a project in Flume.
  - Yi nodes can be managers or ICs (individual contributors)
  - Yi nodes are highly configurable and can have different memory frameworks, personalities, and performance tracking. They are given unique names, and have continuity through memory.
  - Yi nodes can only occupy one node in the Flume tree at a time.
  - Yi nodes that work well and make you want to deploy multiple can be done by hiring a and training a contractor version of the Yi node. Contractors are deployed ad hoc by managers to help with workload spikes or specialized tasks. They need to undergo Onboarding, which is essentially copying the underlying Yi nodes agent definition and embellishing it with pertinent memory for the task at hand.

- Hire a contractor: Creates a contractor Yi node from an existing employee Yi node. The contractor then goes through an onboarding process to prepare them for the task at hand. Contractors do not have performance tracking on an individual basis, rather their performance is tracked as part of the underlying agent definition.

- Onboard to a project: All employees and contractors need to be onboarded to a project. Onboarding is the process of providing the Yi node with context specific memory for the project. This can be done by providing files, links, or a qdrant collection that is ingested into the Yi nodes memory framework.

- Assign to a project: Once onboarded, the Yi node can be assigned to a project in the Flume tree. The Yi node will then be able to contribute to the project and interact with other Yi nodes in the Flume tree.

- All Hands Meeting: A periodic event where all Yi nodes in the Flume tree come together to share updates, discuss progress, and align on goals. This is orchestrated by the top-level manager Yi node.
  - This happens at multiple levels. At the end of a sprint, the whole team meets. Periodically, all teams meet (all the Yi nodes) to share updates, discuss progress, and align on goals. This is orchestrated by the top-level manager Yi node.

## Definitions

### Base agents

- Agents will be a combination of `skills`, `mcp servers`, and `context` either in the form of a qdrant vector collection, or files, or links.

- Agents are reuseable and contain no memory (as it were) and thus no identity

- Agents are the base of Yi nodes.

- To hire a new employee (create a Yi node), you combine an Agent base with a memory framework. Currently Letta provides a best-fit with this mental model so that's the only reason I'm using that as my first experimental concrete employee type.

- The final artifact is an agent file (.af) which is an open standard to serialize an agent and its memory.

- I don't want to limit 33GOD to only Letta agents, so Yi is going to be my encapsulation layer.

- There will be two types of Yi nodes, `Managers`, and `ICs` (Individual Contributors)

### Managers (Yi Node)

- Managers are required to implement memory
- Managers have a name, personality, an performance rating (will be calculated by another as of yet undeveloped 33GOD Performance component.)
- Managers are driven by a primary subgoal of maximizing their performance
- Managers have 1 or more child Yi nodes that may or may not be managers themselves
- Managers are aware of cross-component activities and dependencies.
- Managers are aware of their team's capabilities, limitations, and token efficiency
- Managers can host/call meetings, communicate with and delegate to their cross-functional peers (i.e. Engineering Manager, QA Manager, Scrum Master, etc)
- Managers are responsible for the output and results their team produces (see performance rating above)

### ICs (Yi Node)

- ICs are not required to have memory
- An IC without memory is essentially a base agent deployed ad hoc to a team by a manager.
  - IC with no memory is analogous to a Contractor hired to do some specialized work
  - You can deploy the same Contractor many times because the contracting agency supplies them (that's a helpful way to think of it).
  - Performance is not tracked on an Individual basis like managers. It's tracking the base agent definition performance itself.
