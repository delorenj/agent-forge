# Gemini Code Assistant Context

## Project Overview

This project, "AgentForge," is a meta-agent system for building specialized agent teams. It is a Python-based command-line tool that allows users to define a goal, and AgentForge will then assemble a team of AI agents to accomplish that goal. The system is designed to be modular, with different agents responsible for different parts of the process (e.g., Systems Analyst, Talent Scout, Agent Developer, Integration Architect). It uses the `agno` framework and can connect to a Qdrant vector database for knowledge storage and retrieval.

The core of the system is the `EngineeringManager` (orchestrator), which manages the entire workflow from initial goal intake to final deliverable. It delegates tasks to specialized agents and coordinates their interactions.

## Building and Running

The project uses `pyproject.toml` to manage dependencies. The primary entry point for the CLI is `cli.py`, which uses the `typer` library. The main application logic is in `main.py` and `orchestrator.py`.

*   **Installation:** To install the necessary dependencies, run the following command in your terminal:
    ```bash
    pip install -e .
    ```
    or if you are using `uv`:
    ```bash
    uv pip install -e .
    ```

*   **Running the CLI:** The command-line interface provides various options for interacting with AgentForge. To see the available commands and options, run:
    ```bash
    agentforge --help
    ```

*   **Running the main application:** The main application can be run in interactive mode or with an example goal.
    *   Interactive mode:
        ```bash
        python main.py
        ```
    *   Run with an example goal:
        ```bash
        python main.py --example
        ```

## Development Conventions

*   **Dependencies:** The project uses `pydantic` for data validation and settings management, and `rich` for enhanced command-line output.
*   **Structure:** The codebase is organized into several directories:
    *   `agents`: Contains the different agent implementations.
    *   `core`: Contains the core logic of the application, including the orchestrator.
    *   `docs`: Contains documentation for the project.
    *   `scripts`: Contains scripts for demonstrating the functionality of different agents.
    *   `tests`: Contains tests for the project.
*   **Asynchronous Operations:** The project uses `asyncio` for handling asynchronous operations, which is crucial for managing multiple agents and I/O-bound tasks efficiently.
*   **Separation of Concerns:** There is a clear separation of concerns between the different agent modules, with each agent having a specific role and responsibility.
*   **Orchestration:** The project uses a central `EngineeringManager` (orchestrator) to manage the workflow, which promotes a clean and maintainable architecture.
