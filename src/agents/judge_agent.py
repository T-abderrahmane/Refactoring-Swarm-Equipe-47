"""
Judge Agent - Runs tests and evaluates if code is ready.

This agent is responsible for:
1. Running pytest on the fixed code
2. Evaluating test results
3. Deciding whether to pass or send back to fixer
4. Creating detailed test reports
"""

import os
import time
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models.graph_state import RefactoringState
from src.tools.refactoring_tools import run_pytest
from src.utils.logger import log_llm_interaction, ActionType


def get_llm():
    """Get configured Gemini LLM instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )


def judge_node(state: RefactoringState) -> Dict:
    """
    Judge agent node for LangGraph.
    
    Runs tests and determines if the refactoring is successful.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with test results and decision
    """
    print(f"\n{'='*60}")
    print("⚖️  JUDGE AGENT: Evaluating code quality...")
    print(f"{'='*60}")
    
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    
    print(f"📋 Iteration: {iteration + 1}/{max_iterations}")
    
    # Step 1: Run pytest
    print("\n🧪 Running tests...")
    test_output = run_pytest.invoke({"test_path": "."})
    
    # Determine if tests passed
    tests_passed = "PASSED" in test_output and "FAILED" not in test_output
    tests_passed = tests_passed or "passed" in test_output.lower()
    
    # Check for test failures
    test_failures = []
    if not tests_passed and test_output:
        # Parse test failures (basic parsing)
        if "FAILED" in test_output or "ERROR" in test_output:
            test_failures.append({
                "file": "unknown",
                "description": "Tests failed - see test output for details"
            })
    
    print(f"{'✅' if tests_passed else '❌'} Tests {'PASSED' if tests_passed else 'FAILED'}")
    
    # Step 2: Use LLM to analyze test results
    print("\n🤖 Analyzing test results with Gemini...")
    
    system_prompt = """You are an expert software testing analyst.

Your task is to analyze pytest output and provide a clear assessment of:
1. Whether all tests passed
2. What specific failures occurred (if any)
3. Recommendations for fixes (if tests failed)
4. Overall code quality assessment

Be concise and actionable in your analysis."""

    user_prompt = f"""Here is the pytest output from running tests:

```
{test_output}
```

Please provide:
1. A summary of the test results
2. Specific issues that need to be fixed (if any)
3. Your overall assessment"""

    try:
        llm = get_llm()
        
        # Call LLM to analyze test results with retry logic
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        def call_llm_with_retry():
            time.sleep(2)  # Rate limiting: 2 second delay before each call
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            return llm.invoke(messages)
        
        response = call_llm_with_retry()
        analysis = response.content
        
        # Log the LLM interaction
        log_llm_interaction(
            agent_name="Judge",
            model_used="gemini-2.5-flash",
            action=ActionType.DEBUG,
            input_prompt=f"{system_prompt}\n\n{user_prompt}",
            output_response=analysis,
            status="SUCCESS"
        )
        
        print("✅ Test analysis completed")
        
    except Exception as e:
        error_msg = f"Failed to analyze test results: {str(e)}"
        print(f"❌ {error_msg}")
        analysis = f"Error during analysis: {str(e)}"
        
        # Log failed interaction
        log_llm_interaction(
            agent_name="Judge",
            model_used="gemini-2.5-flash",
            action=ActionType.DEBUG,
            input_prompt=f"{system_prompt}\n\n{user_prompt}",
            output_response=str(e),
            status="FAILURE"
        )
    
    # Step 3: Create test report
    test_report = f"""
=== TEST REPORT ===

Iteration: {iteration + 1}/{max_iterations}
Tests Passed: {tests_passed}

LLM Analysis:
{analysis}

Raw Test Output:
{test_output[:1000]}...
"""
    
    # Step 4: Make decision
    should_continue = not tests_passed and iteration + 1 < max_iterations
    
    if tests_passed:
        final_status = "SUCCESS: All tests passed!"
        print("\n🎉 All tests passed! Refactoring complete.")
    elif iteration + 1 >= max_iterations:
        final_status = f"INCOMPLETE: Max iterations ({max_iterations}) reached"
        print(f"\n⚠️  Max iterations reached. Some issues may remain.")
    else:
        final_status = "IN_PROGRESS: Tests failed, continuing fixes"
        print(f"\n🔄 Tests failed. Sending back to fixer...")
    
    print(f"\n📋 Judge Summary:")
    print(f"  • Tests passed: {tests_passed}")
    print(f"  • Should continue: {should_continue}")
    print(f"  • Status: {final_status}")
    print(f"{'='*60}\n")
    
    return {
        "tests_passed": tests_passed,
        "test_report": test_report,
        "test_failures": test_failures,
        "should_continue": should_continue,
        "final_status": final_status,
        "iteration": iteration + 1
    }
