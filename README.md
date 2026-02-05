# Refactoring Swarm - AI Multi-Agent System

A sophisticated multi-agent system that automatically refactors Python code using AI. Built with LangGraph, LangChain, and Google's Gemini AI.

## 🎯 Overview

The Refactoring Swarm is an autonomous AI system that takes poorly written Python code and transforms it into clean, well-documented, tested code. It uses a multi-agent architecture with specialized agents working together in a self-healing loop.

## 🏗️ Architecture

The system consists of 4 main components:

### 1. **Analyzer Agent** 🔍
- Discovers Python files in the target directory
- Runs static analysis using Pylint
- Uses Gemini AI to create a comprehensive refactoring plan
- Identifies bugs, code smells, and quality issues

### 2. **Fixer Agent** 🔧
- Reads the refactoring plan
- Uses Gemini AI to generate fixed code
- Applies fixes file by file
- Maintains code functionality while improving quality

### 3. **Judge Agent** ⚖️
- Runs pytest to validate code quality
- Uses Gemini AI to analyze test results
- Decides whether fixes were successful
- Triggers additional fix iterations if needed

### 4. **Orchestrator** 🎭
- Coordinates the workflow between all agents using LangGraph
- Manages state transitions
- Implements the self-healing loop (Analyzer → Fixer → Judge → Fixer...)
- Controls iteration limits and termination

## 🔄 Workflow

```
START → Analyzer → Fixer → Judge
                    ↑        ↓
                    └────────┘
                  (if tests fail)
                      ↓
                     END
                  (if tests pass or max iterations)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Google Gemini API key
- Git

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd refactoring-swarm
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

5. Verify setup:
```bash
python check_setup.py
```

### Usage

Basic usage:
```bash
python main_new.py --target_dir ./my_code
```

With custom iteration limit:
```bash
python main_new.py --target_dir ./my_code --max_iterations 5
```

Run on sandbox directly (files modified in place):
```bash
python main_new.py --target_dir ./sandbox --no_copy
```

## 📊 Features

- ✅ **Automatic Code Analysis**: Uses Pylint for comprehensive static analysis
- ✅ **AI-Powered Fixes**: Gemini 2.0 Flash for intelligent code generation
- ✅ **Self-Healing Loop**: Automatically retries fixes if tests fail
- ✅ **Safe Sandbox**: Files are copied to sandbox before modification
- ✅ **Complete Logging**: All LLM interactions logged to `logs/experiment_data.json`
- ✅ **Test Validation**: Pytest integration for quality assurance
- ✅ **Iteration Control**: Configurable max iterations to prevent infinite loops
- ✅ **Progress Tracking**: Real-time console output of agent activities

## 📝 Logging

All LLM interactions are automatically logged to `logs/experiment_data.json` following this format:

```json
{
  "id": "unique-id",
  "timestamp": "2025-01-11T12:00:00.000000",
  "agent": "Analyzer",
  "model": "gemini-2.0-flash-exp",
  "action": "CODE_ANALYSIS",
  "details": "...",
  "status": "SUCCESS"
}
```

Action types:
- `CODE_ANALYSIS`: Code analysis and planning
- `FIX`: Code fixing and refactoring
- `DEBUG`: Test result analysis
- `CODE_GEN`: Code generation (if used)

## 🛠️ Project Structure

```
refactoring-swarm/
├── main_new.py                 # Main entry point (new system)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── check_setup.py             # Environment validation
├── src/
│   ├── agents/
│   │   ├── analyzer.py        # Analyzer agent
│   │   ├── fixer_agent.py     # Fixer agent
│   │   └── judge_agent.py     # Judge agent
│   ├── orchestrator/
│   │   └── refactoring_workflow.py  # LangGraph workflow
│   ├── models/
│   │   └── graph_state.py     # State definitions
│   ├── tools/
│   │   └── refactoring_tools.py  # File ops & testing tools
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── llm_call.py        # LLM wrapper (existing)
├── sandbox/                   # Working directory for refactoring
└── logs/
    └── experiment_data.json   # LLM interaction logs
```

## 🔧 Tools

The system includes these LangChain tools:

- `read_python_file`: Read Python files from sandbox
- `write_python_file`: Write fixed Python files
- `list_python_files`: Discover all Python files
- `run_pylint`: Run static analysis
- `run_pytest`: Execute tests
- `get_file_info`: Get file metadata

All tools enforce sandbox security - no writes outside `./sandbox`.

## 📈 Configuration

### Environment Variables (.env)
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### CLI Arguments
```
--target_dir: Directory to refactor (default: ./sandbox)
--max_iterations: Max fix-test cycles (default: 10)
--no_copy: Skip sandbox copy (DANGEROUS - modifies in place)
```

## 🧪 Testing

Test the system with the included example:

```bash
python main_new.py --target_dir ./sandbox
```

This will refactor `sandbox/test_code.py` which contains intentional issues.

## 🔒 Security

- All file operations are restricted to the `./sandbox` directory
- Paths are validated to prevent directory traversal
- API keys are loaded from `.env` (never committed)
- Files are copied to sandbox before modification (unless `--no_copy`)

## 📚 Technologies Used

- **LangGraph**: Agent workflow orchestration
- **LangChain**: Agent framework and tools
- **Google Gemini 2.0 Flash**: LLM for code analysis and generation
- **Pylint**: Static code analysis
- **Pytest**: Testing framework
- **Pydantic**: Data validation and state management

## 🤝 Contributing

This is an academic project for the IGL Lab at ESI. Please follow the coding guidelines in `docs/coding_guidelines.md`.

## 📄 License

Academic project - ESI / IGL Module 2025-2026

## 🙏 Acknowledgments

- Instructor: BATATA Sofiane
- National School of Computer Science (ESI)
- IGL Module Practical Session

## 📞 Support

For issues or questions, please refer to the project documentation in the `docs/` directory.

---

Built with ❤️ by team 47
