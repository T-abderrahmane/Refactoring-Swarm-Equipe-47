import argparse
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import log_experiment, ActionType
from src.orchestrator.orchestrator import RefactoringOrchestrator
from src.exceptions import RefactoringError

load_dotenv()

def progress_callback(update_info):
    """
    Progress callback function for displaying real-time updates.
    
    Args:
        update_info: Dictionary containing progress information
    """
    session_id = update_info.get("session_id", "unknown")
    iteration = update_info.get("iteration", 0)
    phase = update_info.get("phase", "unknown")
    progress = update_info.get("progress", {})
    
    completion = progress.get("completion_percentage", 0)
    current_iteration = progress.get("current_iteration", 0)
    max_iterations = progress.get("max_iterations", 10)
    
    # Display progress bar
    bar_length = 30
    filled_length = int(bar_length * completion / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print(f"\r🔄 Phase: {phase.upper()} | Progress: [{bar}] {completion:.1f}% | Iteration: {current_iteration}/{max_iterations}", end="", flush=True)

def validate_target_directory(target_dir):
    """
    Validate the target directory and its contents.
    
    Args:
        target_dir: Path to the target directory
        
    Returns:
        Tuple of (is_valid, error_message, python_files_count)
    """
    if not os.path.exists(target_dir):
        return False, f"Directory {target_dir} does not exist", 0
    
    if not os.path.isdir(target_dir):
        return False, f"Path {target_dir} is not a directory", 0
    
    # Check for Python files
    python_files = []
    for root, dirs, files in os.walk(target_dir):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in [
            '.git', '__pycache__', '.venv', 'venv', '.pytest_cache',
            '.refactoring_sandbox', 'node_modules', '.tox'
        ]]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))
    
    if not python_files:
        return False, f"No Python files found in {target_dir}", 0
    
    return True, "", len(python_files)

def display_startup_info(target_dir, python_files_count):
    """
    Display startup information and system status.
    
    Args:
        target_dir: Target directory path
        python_files_count: Number of Python files found
    """
    print("=" * 60)
    print("🤖 REFACTORING SWARM - AUTONOMOUS CODE IMPROVEMENT")
    print("=" * 60)
    print(f"📁 Target Directory: {os.path.abspath(target_dir)}")
    print(f"🐍 Python Files Found: {python_files_count}")
    print(f"⚙️  Max Iterations: 10")
    print(f"🧠 LLM Model: gemini-1.5-flash")
    print("=" * 60)

def display_results(results):
    """
    Display comprehensive execution results.
    
    Args:
        results: Results dictionary from orchestrator execution
    """
    print("\n" + "=" * 60)
    print("📊 EXECUTION RESULTS")
    print("=" * 60)
    
    # Overall status
    status_emoji = "✅" if results["success"] else "❌"
    print(f"{status_emoji} Status: {results['status'].upper()}")
    print(f"⏱️  Execution Time: {results['execution_time']:.2f} seconds")
    print(f"🔄 Total Iterations: {results['total_iterations']}/{results['max_iterations']}")
    
    # File processing results
    print(f"📄 Files Processed: {results['files_processed']}/{results['total_files']}")
    
    # Test results
    if results.get("test_results"):
        test_results = results["test_results"]
        test_emoji = "✅" if test_results["passed"] else "❌"
        print(f"{test_emoji} Tests: {test_results['total_tests'] - test_results['failed_tests']}/{test_results['total_tests']} passed")
        if test_results["failed_tests"] > 0:
            print(f"⚠️  Failed Tests: {test_results['failed_tests']}")
    
    # Refactoring summary
    if results.get("refactoring_summary"):
        refactoring = results["refactoring_summary"]
        print(f"🔧 Refactoring Plans: {refactoring['plans_generated']}")
        print(f"🐛 Issues Found: {refactoring['total_issues']}")
        if refactoring.get("critical_issues", 0) > 0:
            print(f"🚨 Critical Issues: {refactoring['critical_issues']}")
    
    # Performance metrics
    if results.get("performance"):
        perf = results["performance"]
        print(f"⚡ Performance: {perf['files_per_second']:.2f} files/sec")
    
    # Error information
    if results.get("has_errors") and results.get("error_message"):
        print(f"❌ Error: {results['error_message']}")
    
    print("=" * 60)

def main():
    """
    Main entry point for the Refactoring Swarm system.
    
    Handles command line arguments, validates inputs, initializes the orchestrator,
    and executes the refactoring workflow with progress reporting.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Refactoring Swarm - Autonomous Python Code Improvement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --target_dir ./my_project
  python main.py --target_dir /path/to/buggy/code

The system will:
1. Analyze Python files for code quality issues
2. Generate and apply refactoring fixes
3. Validate fixes through automated testing
4. Iterate until all tests pass or max iterations reached
        """
    )
    
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True,
        help="Directory containing Python files to refactor"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of refactoring iterations (default: 10)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use for refactoring (default: gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--sandbox_dir",
        type=str,
        help="Custom sandbox directory for file operations (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose progress reporting"
    )
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # Handle argument parsing errors gracefully
        if e.code != 0:
            log_experiment(
                agent_name="System",
                model_used="N/A",
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Command line argument parsing",
                    "output_response": "Invalid arguments provided"
                },
                status="FAILURE"
            )
        sys.exit(e.code)
    
    # Validate target directory
    print("🔍 Validating target directory...")
    is_valid, error_message, python_files_count = validate_target_directory(args.target_dir)
    
    if not is_valid:
        print(f"❌ Validation Error: {error_message}")
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Directory validation for {args.target_dir}",
                "output_response": f"Validation failed: {error_message}"
            },
            status="FAILURE"
        )
        sys.exit(1)
    
    # Display startup information
    display_startup_info(args.target_dir, python_files_count)
    
    # Log system startup
    log_experiment(
        agent_name="System",
        model_used=args.model,
        action=ActionType.ANALYSIS,
        details={
            "input_prompt": f"System startup with target directory: {args.target_dir}",
            "output_response": f"Found {python_files_count} Python files, using model {args.model}"
        },
        status="SUCCESS"
    )
    
    # Initialize orchestrator
    print("🚀 Initializing Refactoring Swarm...")
    try:
        orchestrator = RefactoringOrchestrator(
            model=args.model,
            max_iterations=args.max_iterations,
            enable_monitoring=True
        )
        
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Orchestrator initialization",
                "output_response": f"Initialized with model {args.model}, max_iterations {args.max_iterations}"
            },
            status="SUCCESS"
        )
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {str(e)}")
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Orchestrator initialization",
                "output_response": f"Initialization failed: {str(e)}"
            },
            status="FAILURE"
        )
        sys.exit(1)
    
    # Execute refactoring workflow
    print("🔄 Starting refactoring workflow...")
    print("   (This may take several minutes depending on codebase size)")
    print()
    
    start_time = time.time()
    
    try:
        # Set up progress callback if verbose mode is enabled
        callback = progress_callback if args.verbose else None
        
        # Execute the workflow
        results = orchestrator.execute_refactoring(
            target_directory=args.target_dir,
            sandbox_directory=args.sandbox_dir,
            progress_callback=callback
        )
        
        # Clear progress line if it was displayed
        if args.verbose:
            print()  # New line after progress bar
        
        # Display results
        display_results(results)
        
        # Log final results
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Workflow execution completed",
                "output_response": f"Success: {results['success']}, Status: {results['status']}, Time: {results['execution_time']:.2f}s"
            },
            status="SUCCESS" if results["success"] else "FAILURE"
        )
        
        # Exit with appropriate code
        exit_code = 0 if results["success"] else 1
        
        if results["success"]:
            print("🎉 Refactoring completed successfully!")
        else:
            print("⚠️  Refactoring completed with issues. Check logs for details.")
        
        sys.exit(exit_code)
        
    except RefactoringError as e:
        print(f"\n❌ Refactoring Error: {str(e)}")
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Workflow execution",
                "output_response": f"Refactoring error: {str(e)}"
            },
            status="FAILURE"
        )
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Workflow execution",
                "output_response": "Execution interrupted by user (Ctrl+C)"
            },
            status="FAILURE"
        )
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Unexpected Error: {str(e)}")
        print(f"⏱️  Execution time before error: {execution_time:.2f} seconds")
        
        log_experiment(
            agent_name="System",
            model_used=args.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Workflow execution",
                "output_response": f"Unexpected error after {execution_time:.2f}s: {str(e)}"
            },
            status="FAILURE"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()