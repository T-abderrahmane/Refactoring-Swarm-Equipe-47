"""
LLM call wrapper with automatic logging of LLM interactions.

This module provides a centralized function for calling the LLM
and automatically logging the interaction for scientific analysis.
"""

from src.utils.logger import log_llm_interaction, ActionType
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def call_llm(agent_name, model, action, prompt, llm_fn):
    """
    Call the LLM and automatically log the interaction.
    
    This is the ONLY place where LLM interactions should be logged.
    All calls to the LLM should go through this function.
    
    Args:
        agent_name: Name of the agent/node calling the LLM
        model: LLM model being used
        action: ActionType indicating what the LLM is doing
        prompt: The prompt to send to the LLM
        llm_fn: The actual LLM function to call
        
    Returns:
        The LLM's response
    """
    try:
        response = llm_fn(prompt)

        # Log the LLM interaction
        log_llm_interaction(
            agent_name=agent_name,
            model_used=model,
            action=action,
            input_prompt=prompt,
            output_response=response,
            status="SUCCESS"
        )

        return response
        
    except Exception as e:
        # Log failed LLM interaction
        log_llm_interaction(
            agent_name=agent_name,
            model_used=model,
            action=action,
            input_prompt=prompt,
            output_response=f"Error: {str(e)}",
            status="FAILURE"
        )
        raise
