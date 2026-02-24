"""
Fixer Agent - Reads the refactoring plan and modifies code to fix issues.

This agent is responsible for:
1. Reading the refactoring plan from the Analyzer
2. Identifying which files need to be modified
3. Using LLM to generate fixed code
4. Writing the corrected code back to files
"""

import os
import time
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models.graph_state import RefactoringState
from src.tools.refactoring_tools import read_python_file, write_python_file
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


def fixer_node(state: RefactoringState) -> Dict:
    """
    Fixer agent node for LangGraph.
    
    Applies fixes to code based on the refactoring plan.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with fix results
    """
    print(f"\n{'='*60}")
    print("🔧 FIXER AGENT: Starting code fixes...")
    print(f"{'='*60}")
    
    refactoring_plan = state.get("refactoring_plan", "")
    python_files = state.get("python_files", [])
    test_failures = state.get("test_failures", [])
    iteration = state.get("iteration", 0)
    
    if not refactoring_plan and not test_failures:
        print("❌ No refactoring plan or test failures provided")
        return {
            "errors": ["Fixer called without refactoring plan or test failures"]
        }
    
    print(f"📋 Iteration: {iteration + 1}")
    print(f"📁 Files to process: {len(python_files)}")
    
    # If we have test failures, focus on those
    if test_failures:
        print(f"⚠️  Fixing {len(test_failures)} test failures")
    
    files_modified = []
    fix_reports = []
    
    # Process each file (skip test files — only fix source code)
    for file_path in python_files:
        if os.path.basename(file_path).startswith("test_"):
            print(f"\n  • Skipping test file: {file_path}")
            continue
        print(f"\n  • Processing: {file_path}")
        
        # Read current file content
        current_content = read_python_file.invoke({"file_path": file_path})
        
        if "Error" in current_content:
            print(f"    ❌ Could not read file: {current_content}")
            continue
        
        # Prepare context for LLM
        context = f"Refactoring Plan:\n{refactoring_plan}\n\n"
        
        if test_failures:
            # Add test failure context
            relevant_failures = [f for f in test_failures if f.get("file") == file_path]
            if relevant_failures:
                context += "Test Failures to Fix:\n"
                for failure in relevant_failures:
                    context += f"- {failure.get('description', 'Unknown error')}\n"
                context += "\n"
        
        # Create prompt for fixing
        system_prompt = """You are an expert Python developer specializing in code refactoring and bug fixes.

Your task is to fix the provided Python code according to the refactoring plan and any test failures.

Guidelines:
1. Fix all syntax errors and bugs first (highest priority)
2. Add missing docstrings to functions and classes
3. Remove unused imports and variables
4. Improve code readability and structure
5. Follow PEP 8 style guidelines
6. Ensure the code is well-documented

IMPORTANT: Return ONLY the complete fixed Python code, nothing else.
Do not include explanations, comments about changes, or markdown formatting.
Just return the raw Python code that should replace the current file."""

        user_prompt = f"""{context}

Current file: {file_path}

Current code:
```python
{current_content}
```

Please provide the fixed version of this code."""

        try:
            llm = get_llm()
            
            # Call LLM to fix code with retry logic
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
            fixed_code = response.content
            
            # Clean up the response (remove markdown code blocks if present)
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
            
            # Log the LLM interaction
            log_llm_interaction(
                agent_name="Fixer",
                model_used="gemini-2.5-flash",
                action=ActionType.FIX,
                input_prompt=f"{system_prompt}\n\n{user_prompt}",
                output_response=fixed_code,
                status="SUCCESS"
            )
            
            # Write fixed code back to file
            write_result = write_python_file.invoke({
                "file_path": file_path,
                "content": fixed_code
            })
            
            if "Success" in write_result:
                print(f"    ✅ Fixed and saved")
                files_modified.append(file_path)
                fix_reports.append(f"Fixed {file_path}")
            else:
                print(f"    ❌ Failed to write: {write_result}")
                
        except Exception as e:
            error_msg = f"Failed to fix {file_path}: {str(e)}"
            print(f"    ❌ {error_msg}")
            
            # Log failed interaction
            log_llm_interaction(
                agent_name="Fixer",
                model_used="gemini-2.5-flash",
                action=ActionType.FIX,
                input_prompt=f"{system_prompt}\n\n{user_prompt}",
                output_response=str(e),
                status="FAILURE"
            )
            
            fix_reports.append(error_msg)
    
    # Create fix report
    fix_report = f"""
=== FIX REPORT ===

Iteration: {iteration + 1}
Files Modified: {len(files_modified)}
Success: {len(files_modified) > 0}

Modified Files:
{chr(10).join(f'  • {f}' for f in files_modified)}
"""
    
    print(f"\n📋 Fix Summary:")
    print(f"  • Files modified: {len(files_modified)}")
    print(f"{'='*60}\n")
    
    return {
        "files_modified": files_modified,
        "fix_report": fix_report,
        "fix_attempts": state.get("fix_attempts", 0) + 1
    }
