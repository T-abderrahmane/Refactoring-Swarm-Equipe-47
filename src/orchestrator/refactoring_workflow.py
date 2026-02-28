"""
Orchestrator - Main LangGraph workflow that coordinates all agents.

This orchestrator:
1. Defines the workflow graph with all agent nodes
2. Manages state transitions between agents
3. Implements the self-healing loop (Analyzer -> Fixer -> Judge -> Fixer...)
4. Controls iteration limits and termination conditions
"""

from langgraph.graph import StateGraph, END
from src.models.graph_state import RefactoringState
from src.agents.analyzer import analyzer_node
from src.agents.fixer_agent import fixer_node
from src.agents.tester_agent import tester_node
from src.agents.judge_agent import judge_node


def should_continue_fixing(state: RefactoringState) -> str:
    """
    Determine if we should continue fixing or end.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name or END
    """
    # Check if we should continue
    should_continue = state.get("should_continue", False)
    tests_passed = state.get("tests_passed", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    
    if tests_passed:
        # All tests passed, we're done!
        return END
    elif iteration >= max_iterations:
        # Max iterations reached, stop
        return END
    elif should_continue:
        # Continue fixing
        return "fixer"
    else:
        # Stop for other reasons
        return END


def create_refactoring_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for the refactoring swarm.
    
    The workflow follows this pattern:
    1. START -> Analyzer: Analyze code and create plan
    2. Analyzer -> Fixer: Apply fixes based on plan
    3. Fixer -> Tester: Generate missing test scripts
    4. Tester -> Judge: Run tests and evaluate
    5. Judge -> Fixer (if tests fail and iterations remain)
    6. Judge -> END (if tests pass or max iterations reached)
    
    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph
    workflow = StateGraph(RefactoringState)
    
    # Add nodes for each agent
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("judge", judge_node)
    
    # Define edges (workflow transitions)
    # Start -> Analyzer
    workflow.set_entry_point("analyzer")
    
    # Analyzer -> Fixer (always)
    workflow.add_edge("analyzer", "fixer")
    
    # Fixer -> Tester (always)
    workflow.add_edge("fixer", "tester")
    
    # Tester -> Judge (always)
    workflow.add_edge("tester", "judge")
    
    # Judge -> Fixer or END (conditional)
    workflow.add_conditional_edges(
        "judge",
        should_continue_fixing,
        {
            "fixer": "fixer",
            END: END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_refactoring_workflow(target_directory: str, max_iterations: int = 10):
    """
    Run the complete refactoring workflow.
    
    Args:
        target_directory: Directory containing code to refactor
        max_iterations: Maximum number of fix-test iterations
        
    Returns:
        Final state of the workflow
    """
    print("\n" + "="*60)
    print("🚀 REFACTORING SWARM - Starting Workflow")
    print("="*60)
    print(f"📂 Target: {target_directory}")
    print(f"🔄 Max iterations: {max_iterations}")
    print("="*60 + "\n")
    
    # Create the workflow
    app = create_refactoring_workflow()
    
    # Initialize state
    initial_state: RefactoringState = {
        "target_directory": target_directory,
        "python_files": [],
        "analysis_complete": False,
        "analysis_report": "",
        "issues_found": [],
        "refactoring_plan": "",
        "files_modified": [],
        "fix_attempts": 0,
        "current_file": None,
        "fix_report": "",
        "generated_test_files": [],
        "tests_passed": False,
        "test_report": "",
        "test_failures": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "final_status": "STARTING",
        "errors": []
    }
    
    # Run the workflow
    try:
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*60)
        print("🏁 WORKFLOW COMPLETE")
        print("="*60)
        print(f"📊 Final Status: {final_state.get('final_status', 'UNKNOWN')}")
        print(f"🔄 Iterations: {final_state.get('iteration', 0)}/{max_iterations}")
        print(f"📁 Files Modified: {len(final_state.get('files_modified', []))}")
        print(f"✅ Tests Passed: {final_state.get('tests_passed', False)}")
        
        if final_state.get('errors'):
            print(f"⚠️  Errors: {len(final_state.get('errors', []))}")
        
        print("="*60 + "\n")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ WORKFLOW ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the workflow
    import sys
    
    target = sys.argv[1] if len(sys.argv) > 1 else "./sandbox"
    final_state = run_refactoring_workflow(target)
    
    # Print summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Status: {final_state.get('final_status')}")
    print(f"Tests Passed: {final_state.get('tests_passed')}")
    print(f"Iterations: {final_state.get('iteration')}")
