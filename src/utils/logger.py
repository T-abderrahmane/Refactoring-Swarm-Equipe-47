"""
Logger module for recording LLM interactions in the Refactoring Swarm system.

⚠️ IMPORTANT: According to the requirements, we should ONLY log LLM interactions,
not every internal operation. This module provides functions to log interactions
with the LLM (Gemini) for scientific analysis.
"""

import json
import os
import uuid
from datetime import datetime
from enum import Enum

# Path to the log file
LOG_FILE = os.path.join("logs", "experiment_data.json")


class ActionType(str, Enum):
    """
    Enumeration of possible action types for standardizing analysis.
    According to the requirements, we should ONLY log LLM interactions.
    """
    ANALYSIS = "CODE_ANALYSIS"  # LLM analyzing code
    GENERATION = "CODE_GEN"     # LLM generating new code/tests/docs
    DEBUG = "DEBUG"             # LLM analyzing execution errors
    FIX = "FIX"                 # LLM suggesting/generating fixes


def log_llm_interaction(agent_name: str, model_used: str, action: ActionType, 
                       input_prompt: str, output_response: str, status: str = "SUCCESS"):
    """
    Record ONLY LLM interactions for scientific analysis.
    
    ⚠️ IMPORTANT: This function should ONLY be called when actually interacting with an LLM.
    Do NOT call this for internal operations like file reading, tool execution, etc.
    
    According to the requirements document:
    "Golden Rule: Each significant interaction with the LLM (analyze, generate, correct) must be recorded."
    
    Args:
        agent_name (str): Name of the agent/node calling the LLM (e.g., "Auditor", "Fixer")
        model_used (str): LLM model used (e.g., "gemini-2.5-flash")
        action (ActionType): Type of LLM action (use ActionType enum)
        input_prompt (str): The exact text sent to the LLM
        output_response (str): The raw response received from the LLM
        status (str): "SUCCESS" or "FAILURE"

    Raises:
        ValueError: If action type is invalid
    """
    
    # Validate action type
    valid_actions = [a.value for a in ActionType]
    if isinstance(action, ActionType):
        action_str = action.value
    elif action in valid_actions:
        action_str = action
    else:
        raise ValueError(f"❌ Invalid action: '{action}'. Use ActionType class (e.g., ActionType.FIX).")

    # Validate required fields
    if not input_prompt or not output_response:
        raise ValueError(
            f"❌ Logging Error (Agent: {agent_name}): "
            f"'input_prompt' and 'output_response' are REQUIRED for LLM interaction logging."
        )

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "model": model_used,
        "action": action_str,
        "details": {
            "input_prompt": input_prompt,
            "output_response": output_response
        },
        "status": status
    }

    # Read existing logs
    data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Log file {LOG_FILE} was corrupted. Created new log list.")
            data = []

    data.append(entry)
    
    # Write logs
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Legacy function for backward compatibility
def log_experiment(agent_name: str, model_used: str, action: ActionType, details: dict, status: str):
    """
    Legacy function for backward compatibility.
    
    ⚠️ DEPRECATED: Use log_llm_interaction() instead.
    This function is kept only for backward compatibility and will extract
    input_prompt and output_response from the details dict.
    
    NOTE: If this is called for non-LLM operations (missing input_prompt/output_response),
    it will silently skip logging as per requirements to ONLY log LLM interactions.
    """
    input_prompt = details.get("input_prompt", "")
    output_response = details.get("output_response", "")
    
    # Only log if this appears to be an actual LLM interaction
    if input_prompt and output_response:
        log_llm_interaction(
            agent_name=agent_name,
            model_used=model_used,
            action=action,
            input_prompt=input_prompt,
            output_response=output_response,
            status=status
        )
