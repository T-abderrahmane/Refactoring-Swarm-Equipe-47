# Requirements Document

## Introduction

The Refactoring Swarm is a multi-agent system designed to automatically refactor poorly written Python code into clean, functional, and well-tested code. The system orchestrates collaboration between specialized agents to analyze, fix, and validate Python codebases without human intervention.

## Glossary

- **Refactoring_Swarm**: The complete multi-agent system that processes buggy Python code
- **Auditor_Agent**: Agent responsible for code analysis and refactoring plan generation
- **Fixer_Agent**: Agent that implements code corrections based on the refactoring plan
- **Judge_Agent**: Agent that validates fixes by executing unit tests
- **Orchestrator**: Main coordination component that manages agent workflow
- **Sandbox**: Isolated directory where code modifications are performed
- **Target_Directory**: Input directory containing the buggy Python code to be refactored
- **Experiment_Logger**: Component that records all agent interactions for analysis
- **Self_Healing_Loop**: Iterative process where failed fixes are sent back for correction

## Requirements

### Requirement 1

**User Story:** As a software engineering researcher, I want the system to automatically refactor buggy Python code, so that I can study autonomous software maintenance capabilities.

#### Acceptance Criteria

1. WHEN the system receives a target directory with Python files, THE Refactoring_Swarm SHALL process all Python files in the directory
2. THE Refactoring_Swarm SHALL produce clean, functional Python code that passes unit tests
3. THE Refactoring_Swarm SHALL improve the Pylint quality score of the processed code
4. THE Refactoring_Swarm SHALL complete processing within 10 iterations maximum to prevent infinite loops
5. THE Refactoring_Swarm SHALL operate without human intervention during the refactoring process

### Requirement 2

**User Story:** As a system administrator, I want the system to be launched via command line with specific arguments, so that it can be integrated into automated workflows.

#### Acceptance Criteria

1. WHEN launched with --target_dir argument, THE Refactoring_Swarm SHALL process the specified directory
2. THE Refactoring_Swarm SHALL validate that the target directory exists before processing
3. THE Refactoring_Swarm SHALL exit cleanly with appropriate status codes upon completion
4. THE Refactoring_Swarm SHALL display progress information during execution

### Requirement 3

**User Story:** As a researcher, I want all agent interactions to be logged in a structured format, so that I can analyze the system's behavior and performance.

#### Acceptance Criteria

1. THE Experiment_Logger SHALL record every LLM interaction with input prompts and output responses
2. THE Experiment_Logger SHALL categorize actions using ActionType enumeration (ANALYSIS, GENERATION, DEBUG, FIX)
3. THE Experiment_Logger SHALL save all data to logs/experiment_data.json in valid JSON format
4. THE Experiment_Logger SHALL include timestamps, agent names, and model information for each interaction
5. THE Experiment_Logger SHALL ensure log data integrity throughout the execution process

### Requirement 4

**User Story:** As a security-conscious developer, I want the system to restrict file operations to designated areas, so that it cannot modify files outside the intended scope.

#### Acceptance Criteria

1. THE Refactoring_Swarm SHALL only write files within the sandbox directory
2. THE Refactoring_Swarm SHALL prevent agents from accessing files outside the target directory and sandbox
3. THE Refactoring_Swarm SHALL validate file paths before any write operations
4. THE Refactoring_Swarm SHALL reject attempts to modify system files or directories

### Requirement 5

**User Story:** As a software quality analyst, I want the system to use specialized agents for different tasks, so that each aspect of refactoring is handled by an expert component.

#### Acceptance Criteria

1. THE Auditor_Agent SHALL analyze Python code and generate comprehensive refactoring plans
2. THE Fixer_Agent SHALL implement code corrections based on the auditor's recommendations
3. THE Judge_Agent SHALL execute unit tests to validate code functionality
4. THE Orchestrator SHALL coordinate the workflow between all agents
5. WHEN tests fail, THE Judge_Agent SHALL send error logs back to the Fixer_Agent for correction

### Requirement 6

**User Story:** As a system integrator, I want the system to use standardized tools for code analysis and testing, so that results are consistent and reliable.

#### Acceptance Criteria

1. THE Auditor_Agent SHALL use Pylint for static code analysis
2. THE Judge_Agent SHALL use pytest for unit test execution
3. THE Refactoring_Swarm SHALL integrate with Google Gemini API for LLM capabilities
4. THE Refactoring_Swarm SHALL use LangGraph or CrewAI for agent orchestration
5. THE Refactoring_Swarm SHALL follow the existing project structure and coding guidelines

### Requirement 7

**User Story:** As a performance monitor, I want the system to implement a self-healing mechanism, so that failed fixes can be automatically corrected through iteration.

#### Acceptance Criteria

1. WHEN unit tests fail, THE Judge_Agent SHALL provide detailed error information to the Fixer_Agent
2. THE Self_Healing_Loop SHALL allow up to 10 correction attempts before terminating
3. THE Orchestrator SHALL track iteration count and prevent infinite loops
4. THE Self_Healing_Loop SHALL log each iteration attempt with failure reasons
5. WHEN all tests pass, THE Judge_Agent SHALL confirm successful completion