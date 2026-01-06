from src.utils.logger import log_experiment, ActionType
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def call_llm(agent_name, model, action, prompt, llm_fn):
    response = llm_fn(prompt)

    log_experiment(
        agent_name=agent_name,
        model_used=model,
        action=action,
        details={
            "input_prompt": prompt,
            "output_response": response
        },
        status="SUCCESS"
    )

    return response
