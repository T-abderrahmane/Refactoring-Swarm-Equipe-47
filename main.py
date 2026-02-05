"""
Main entry point for the Refactoring Swarm multi-agent system.

This script provides the CLI interface for running the refactoring workflow.
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from src.orchestrator.refactoring_workflow import run_refactoring_workflow
from src.utils.logger import log_llm_interaction, ActionType

# Load environment variables
load_dotenv()


def validate_environment():
    """
    Validate that the environment is properly configured.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY not found in environment. Please add it to .env file."
    
    # Check for sandbox directory
    sandbox_dir = Path("./sandbox")
    if not sandbox_dir.exists():
        print(f"📁 Creating sandbox directory: {sandbox_dir}")
        sandbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for logs directory
    logs_dir = Path("./logs")
    if not logs_dir.exists():
        print(f"📁 Creating logs directory: {logs_dir}")
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    return True, ""


def validate_target_directory(target_dir: str):
    """
    Validate the target directory exists and contains Python files.
    
    Args:
        target_dir: Path to target directory
        
    Returns:
        Tuple of (is_valid, error_message, python_file_count)
    """
    path = Path(target_dir)
    
    if not path.exists():
        return False, f"Directory {target_dir} does not exist", 0
    
    if not path.is_dir():
        return False, f"Path {target_dir} is not a directory", 0
    
    # Count Python files
    python_files = list(path.rglob("*.py"))
    
    if not python_files:
        return False, f"No Python files found in {target_dir}", 0
    
    return True, "", len(python_files)


def setup_sandbox(target_dir: str):
    """
    Copy target directory to sandbox for safe processing.
    
    Args:
        target_dir: Source directory to copy
        
    Returns:
        Path to sandbox directory
    """
    import shutil
    
    sandbox_dir = Path("./sandbox")
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear sandbox
    for item in sandbox_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    # Copy target directory to sandbox
    source_path = Path(target_dir)
    
    if source_path.resolve() == sandbox_dir.resolve():
        print("📂 Target is already sandbox directory, skipping copy")
        return str(sandbox_dir)
    
    print(f"📂 Copying {target_dir} to sandbox...")
    
    for item in source_path.rglob("*"):
        if item.is_file():
            # Calculate relative path
            rel_path = item.relative_to(source_path)
            dest_path = sandbox_dir / rel_path
            
            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, dest_path)
    
    print("✅ Copied to sandbox")
    
    return str(sandbox_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Refactoring Swarm - AI-powered code refactoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --target_dir ./my_code
  python main.py --target_dir ./sandbox --max_iterations 5
        """
    )
    
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./sandbox",
        help="Directory containing Python code to refactor (default: ./sandbox)"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of fix-test iterations (default: 10)"
    )
    
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="Don't copy to sandbox (use target_dir directly - DANGEROUS)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*20 + "REFACTORING SWARM")
    print(" "*15 + "AI Multi-Agent Code Refactoring")
    print("="*70 + "\n")
    
    # Validate environment
    print("🔍 Validating environment...")
    env_valid, env_error = validate_environment()
    if not env_valid:
        print(f"❌ Environment error: {env_error}")
        sys.exit(1)
    print("✅ Environment valid\n")
    
    # Validate target directory
    print(f"🔍 Validating target directory: {args.target_dir}")
    is_valid, error_msg, py_count = validate_target_directory(args.target_dir)
    
    if not is_valid:
        print(f"❌ {error_msg}")
        sys.exit(1)
    
    print(f"✅ Found {py_count} Python file(s)\n")
    
    # Setup sandbox (unless --no_copy)
    if not args.no_copy and args.target_dir != "./sandbox":
        sandbox_path = setup_sandbox(args.target_dir)
    else:
        sandbox_path = args.target_dir
        if args.no_copy:
            print("⚠️  WARNING: Running without sandbox copy - files will be modified in place!\n")
    
    # Log system startup
    log_llm_interaction(
        agent_name="System",
        model_used="N/A",
        action=ActionType.ANALYSIS,
        input_prompt=f"Starting refactoring workflow for {sandbox_path}",
        output_response=f"Target: {sandbox_path}, Max iterations: {args.max_iterations}",
        status="SUCCESS"
    )
    
    # Run the refactoring workflow
    try:
        final_state = run_refactoring_workflow(
            target_directory=sandbox_path,
            max_iterations=args.max_iterations
        )
        
        # Print final summary
        print("\n" + "="*70)
        print(" "*25 + "FINAL SUMMARY")
        print("="*70)
        print(f"📊 Status: {final_state.get('final_status', 'UNKNOWN')}")
        print(f"🔄 Iterations completed: {final_state.get('iteration', 0)}/{args.max_iterations}")
        print(f"📁 Files analyzed: {len(final_state.get('python_files', []))}")
        print(f"✏️  Files modified: {len(final_state.get('files_modified', []))}")
        print(f"✅ Tests passed: {final_state.get('tests_passed', False)}")
        
        if final_state.get('errors'):
            print(f"⚠️  Errors encountered: {len(final_state.get('errors', []))}")
            for error in final_state.get('errors', [])[:3]:  # Show first 3 errors
                print(f"   • {error}")
        
        print("="*70 + "\n")
        
        # Exit code based on success
        if final_state.get('tests_passed'):
            print("🎉 SUCCESS! All tests passed.")
            sys.exit(0)
        else:
            print("⚠️  INCOMPLETE: Some issues may remain.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

    
