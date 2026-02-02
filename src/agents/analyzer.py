"""
Analyzer Agent - Reads code, runs static analysis, and produces a refactoring plan.

This agent is responsible for:
1. Discovering Python files in the target directory
2. Running pylint on each file
3. Analyzing the results
4. Creating a comprehensive refactoring plan
"""

import os
import time
from pathlib import Path
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models.graph_state import RefactoringState
from src.tools.refactoring_tools import list_python_files, read_python_file, run_pylint
from src.utils.logger import log_llm_interaction, ActionType
import json


# Initialize Gemini model
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


def analyzer_node(state: RefactoringState) -> Dict:
    """
    Analyzer agent node for LangGraph.
    
    Analyzes Python code and creates a refactoring plan.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with analysis results
    """
    print(f"\n{'='*60}")
    print("🔍 ANALYZER AGENT: Starting code analysis...")
    print(f"{'='*60}")
    
    target_dir = state.get("target_directory", "./sandbox")
    
    # Step 1: Discover Python files
    print(f"📂 Scanning directory: {target_dir}")
    files_result = list_python_files.invoke({"directory": "."})
    
    if "Error" in files_result or "No Python files" in files_result:
        print(f"❌ Error discovering files: {files_result}")
        return {
            "analysis_complete": False,
            "errors": [f"File discovery failed: {files_result}"]
        }
    
    python_files = [f.strip() for f in files_result.split("\n") if f.strip()]
    print(f"✅ Found {len(python_files)} Python files")
    
    # Step 2: Run pylint on each file and collect results
    print("\n🔍 Running static analysis (pylint)...")
    analysis_results = []
    issues = []
    
    for file_path in python_files:
        print(f"  • Analyzing: {file_path}")
        pylint_output = run_pylint.invoke({"file_path": file_path})
        
        # Read file content for context
        file_content = read_python_file.invoke({"file_path": file_path})
        
        analysis_results.append({
            "file": file_path,
            "pylint_output": pylint_output,
            "content_preview": file_content[:500] if len(file_content) < 500 else file_content[:500] + "..."
        })
        
        # Parse issues from pylint output
        if "rated at" in pylint_output.lower() or "error" in pylint_output.lower():
            issues.append({
                "file": file_path,
                "description": "Code quality issues detected by pylint",
                "severity": "medium"
            })
    
    # Step 3: Use LLM to analyze results and create refactoring plan
    print("\n🤖 Generating refactoring plan with Gemini...")
    
    # Prepare prompt for LLM
    system_prompt = """You are an expert Python code analyzer and refactoring planner.
    
Your task is to:
1. Analyze the pylint output for each file
2. Identify the most critical issues (bugs, code smells, missing documentation, etc.)
3. Create a prioritized refactoring plan

Focus on:
- Syntax errors and bugs (highest priority)
- Missing or incorrect documentation
- Code style issues
- Unused imports or variables
- Complex or unclear code

Provide a clear, actionable plan with specific file names and line numbers where possible.
Format your response as a structured plan."""

    analysis_summary = "\n\n".join([
        f"File: {r['file']}\n{r['pylint_output']}"
        for r in analysis_results
    ])
    
    user_prompt = f"""Here are the analysis results for {len(python_files)} Python files:

{analysis_summary}

Please create a comprehensive refactoring plan that prioritizes the most critical issues."""

    try:
        llm = get_llm()
        
        # Call LLM with retry logic
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
        refactoring_plan = response.content
        
        # Log the LLM interaction
        log_llm_interaction(
            agent_name="Analyzer",
            model_used="gemini-2.5-flash",
            action=ActionType.ANALYSIS,
            input_prompt=f"{system_prompt}\n\n{user_prompt}",
            output_response=refactoring_plan,
            status="SUCCESS"
        )
        
        print("✅ Refactoring plan generated successfully")
        
        # Create analysis report
        analysis_report = f"""
=== CODE ANALYSIS REPORT ===

Files Analyzed: {len(python_files)}
Issues Found: {len(issues)}

REFACTORING PLAN:
{refactoring_plan}

FILES:
{', '.join(python_files)}
"""
        
        print(f"\n📋 Analysis Summary:")
        print(f"  • Files analyzed: {len(python_files)}")
        print(f"  • Issues found: {len(issues)}")
        print(f"{'='*60}\n")
        
        return {
            "python_files": python_files,
            "analysis_complete": True,
            "analysis_report": analysis_report,
            "issues_found": issues,
            "refactoring_plan": refactoring_plan
        }
        
    except Exception as e:
        error_msg = f"LLM analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Log failed interaction
        log_llm_interaction(
            agent_name="Analyzer",
            model_used="gemini-2.5-flash",
            action=ActionType.ANALYSIS,
            input_prompt=f"{system_prompt}\n\n{user_prompt}",
            output_response=str(e),
            status="FAILURE"
        )
        
        return {
            "analysis_complete": False,
            "errors": [error_msg]
        }
