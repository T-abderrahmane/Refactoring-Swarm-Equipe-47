"""
LangGraph workflow definition for the Refactoring Swarm system.

This module defines the agent workflow using LangGraph for orchestration,
managing state transitions and conditional routing based on test results.
"""

from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.core import AgentState, AgentStatus, RefactoringPlan, TestResult
from ..models.state_manager import StateManager, StateValidationError
from ..agents.auditor import AuditorAgent
from ..agents.fixer import FixerAgent
from ..agents.judge import JudgeAgent
from ..security.sandbox import SandboxManager
from ..utils.logger import log_experiment, ActionType
from ..exceptions import RefactoringError
from .error_handler import WorkflowErrorHandler


class WorkflowState(Dict[str, Any]):
    """
    State dictionary for the LangGraph workflow.
    
    Contains all necessary information for agent coordination and execution.
    """
    # Core state
    agent_state: AgentState
    
    # Agent instances
    auditor: Optional[AuditorAgent]
    fixer: Optional[FixerAgent]
    judge: Optional[JudgeAgent]
    
    # Managers
    state_manager: Optional[StateManager]
    sandbox_manager: Optional[SandboxManager]
    
    # Workflow control
    max_iterations: int
    current_iteration: int
    should_continue: bool
    
    # Error handling
    error_handler: Optional[WorkflowErrorHandler]
    
    # Results and feedback
    refactoring_plans: List[RefactoringPlan]
    test_results: Optional[TestResult]
    feedback_loop_state: Optional[Dict[str, Any]]
    
    # Error handling
    last_error: str
    error_count: int


class RefactoringWorkflow:
    """
    Main workflow orchestrator using LangGraph for agent coordination.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash", llm_fn=None, 
                 max_iterations: int = 10):
        """
        Initialize the refactoring workflow.
        
        Args:
            model: LLM model to use for all agents
            llm_fn: Function to call the LLM (for dependency injection)
            max_iterations: Maximum number of iterations for self-healing loop
        """
        self.model = model
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        
        # Initialize error handler
        self.error_handler = WorkflowErrorHandler()
        
        # Initialize workflow graph
        self.workflow_graph: Optional[StateGraph] = None
        self.memory_saver = MemorySaver()
        
        # Workflow configuration
        self.workflow_config = {
            "configurable": {
                "thread_id": "refactoring_session",
                "checkpoint_ns": "refactoring_workflow"
            }
        }
    
    def create_workflow(self) -> StateGraph:
        """
        Create and configure the LangGraph workflow.
        
        Returns:
            Configured LangGraph workflow
        """
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent and control flow
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("audit", self._audit_node)
        workflow.add_node("fix", self._fix_node)
        workflow.add_node("test", self._test_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Define state transitions
        workflow.add_edge("initialize", "audit")
        workflow.add_edge("audit", "fix")
        workflow.add_edge("fix", "test")
        workflow.add_edge("test", "evaluate")
        
        # Conditional routing from evaluate node
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_routing,
            {
                "continue": "fix",  # Continue fixing if tests failed and under iteration limit
                "complete": "finalize",  # Complete if tests passed
                "terminate": "finalize",  # Terminate if max iterations reached
                "error": "error_handler"  # Handle errors
            }
        )
        
        # Final edges
        workflow.add_edge("finalize", END)
        workflow.add_edge("error_handler", END)
        
        # Compile workflow with checkpointing
        self.workflow_graph = workflow.compile(checkpointer=self.memory_saver)
        
        return self.workflow_graph
    
    def _initialize_node(self, state: WorkflowState) -> WorkflowState:
        """
        Initialize the workflow with agents and managers.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            # Log initialization
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Initializing refactoring workflow",
                    "output_response": f"Starting workflow for {len(state.get('target_files', []))} files"
                },
                status="SUCCESS"
            )
            
            # Initialize sandbox manager
            sandbox_manager = SandboxManager(
                target_directory=state["agent_state"].target_directory,
                sandbox_directory=state["agent_state"].sandbox_directory
            )
            sandbox_manager.setup_sandbox()
            
            # Initialize agents
            auditor = AuditorAgent(model=self.model, llm_fn=self.llm_fn)
            fixer = FixerAgent(sandbox_manager, model=self.model, llm_fn=self.llm_fn)
            judge = JudgeAgent(model=self.model, llm_fn=self.llm_fn)
            
            # Initialize state manager
            state_manager = StateManager()
            
            # Initialize feedback loop state
            feedback_loop_state = judge.create_feedback_loop_state(self.max_iterations)
            
            # Update workflow state
            state.update({
                "auditor": auditor,
                "fixer": fixer,
                "judge": judge,
                "state_manager": state_manager,
                "sandbox_manager": sandbox_manager,
                "error_handler": self.error_handler,
                "max_iterations": self.max_iterations,
                "current_iteration": 0,
                "should_continue": True,
                "refactoring_plans": [],
                "test_results": None,
                "feedback_loop_state": feedback_loop_state,
                "last_error": "",
                "error_count": 0
            })
            
            # Transition agent state to analyzing
            state["agent_state"] = state_manager.transition_state(
                state["agent_state"], AgentStatus.ANALYZING
            )
            
            return state
            
        except Exception as e:
            # Use error handler for comprehensive error handling
            context = {
                "phase": "initialization",
                "target_directory": state.get("agent_state", {}).get("target_directory", "unknown"),
                "critical_operation": True
            }
            
            if "error_handler" in state and state["error_handler"]:
                state = state["error_handler"].handle_error(e, context, state)
            else:
                # Fallback error handling
                state["last_error"] = f"Initialization failed: {str(e)}"
                state["error_count"] = state.get("error_count", 0) + 1
                state["should_continue"] = False
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Workflow initialization failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _audit_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the audit phase using the Auditor Agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with refactoring plans
        """
        try:
            auditor = state["auditor"]
            agent_state = state["agent_state"]
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": f"Starting audit phase for {len(agent_state.current_files)} files",
                    "output_response": "Audit phase initiated"
                },
                status="SUCCESS"
            )
            
            # Analyze all files in the target directory
            refactoring_plans = []
            for file_path in agent_state.current_files:
                try:
                    plan = auditor.analyze_file(file_path)
                    refactoring_plans.append(plan)
                except Exception as e:
                    # Log individual file analysis errors but continue
                    log_experiment(
                        agent_name="Orchestrator",
                        model=self.model,
                        action=ActionType.ANALYSIS,
                        details={
                            "input_prompt": f"Failed to analyze {file_path}",
                            "output_response": f"Error: {str(e)}"
                        },
                        status="FAILURE"
                    )
                    continue
            
            if not refactoring_plans:
                raise RefactoringError("No files could be analyzed successfully")
            
            # Update state with refactoring plans
            state["refactoring_plans"] = refactoring_plans
            
            # Update agent state
            # For simplicity, use the first plan as the primary plan
            primary_plan = refactoring_plans[0] if refactoring_plans else None
            state["agent_state"] = state["state_manager"].update_refactoring_plan(
                state["agent_state"], primary_plan
            )
            
            # Transition to fixing state
            state["agent_state"] = state["state_manager"].transition_state(
                state["agent_state"], AgentStatus.FIXING
            )
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Audit phase completed",
                    "output_response": f"Generated {len(refactoring_plans)} refactoring plans"
                },
                status="SUCCESS"
            )
            
            return state
            
        except Exception as e:
            # Use error handler for comprehensive error handling
            context = {
                "phase": "audit",
                "files_being_processed": agent_state.current_files,
                "iteration": state.get("current_iteration", 0)
            }
            
            state = state["error_handler"].handle_error(e, context, state)
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Audit phase failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _fix_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the fix phase using the Fixer Agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state after applying fixes
        """
        try:
            fixer = state["fixer"]
            refactoring_plans = state["refactoring_plans"]
            current_iteration = state["current_iteration"]
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.FIX,
                details={
                    "input_prompt": f"Starting fix phase - iteration {current_iteration + 1}",
                    "output_response": f"Processing {len(refactoring_plans)} refactoring plans"
                },
                status="SUCCESS"
            )
            
            # Apply fixes for each refactoring plan
            successful_fixes = 0
            for plan in refactoring_plans:
                try:
                    # Use incremental fixing strategy
                    success = fixer.apply_fixes(plan, use_incremental=True)
                    if success:
                        successful_fixes += 1
                except Exception as e:
                    # Log individual fix errors but continue
                    log_experiment(
                        agent_name="Orchestrator",
                        model=self.model,
                        action=ActionType.FIX,
                        details={
                            "input_prompt": f"Failed to apply fixes for {plan.file_path}",
                            "output_response": f"Error: {str(e)}"
                        },
                        status="FAILURE"
                    )
                    continue
            
            if successful_fixes == 0:
                raise RefactoringError("No fixes could be applied successfully")
            
            # Transition to testing state
            state["agent_state"] = state["state_manager"].transition_state(
                state["agent_state"], AgentStatus.TESTING
            )
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.FIX,
                details={
                    "input_prompt": "Fix phase completed",
                    "output_response": f"Applied fixes to {successful_fixes}/{len(refactoring_plans)} files"
                },
                status="SUCCESS"
            )
            
            return state
            
        except Exception as e:
            # Use error handler for comprehensive error handling
            context = {
                "phase": "fix",
                "refactoring_plans": len(refactoring_plans),
                "iteration": current_iteration,
                "successful_fixes": successful_fixes if 'successful_fixes' in locals() else 0
            }
            
            state = state["error_handler"].handle_error(e, context, state)
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.FIX,
                details={
                    "input_prompt": "Fix phase failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _test_node(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the test phase using the Judge Agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with test results
        """
        try:
            judge = state["judge"]
            agent_state = state["agent_state"]
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Starting test phase",
                    "output_response": "Test execution initiated"
                },
                status="SUCCESS"
            )
            
            # Run tests in the sandbox directory
            test_directory = agent_state.sandbox_directory
            test_result = judge.run_tests(test_directory)
            
            # Update state with test results
            state["test_results"] = test_result
            state["agent_state"] = state["state_manager"].update_test_results(
                state["agent_state"], test_result
            )
            
            # Update feedback loop state
            feedback_loop_state = state["feedback_loop_state"]
            state["feedback_loop_state"] = judge.process_feedback_loop_iteration(
                feedback_loop_state, test_result
            )
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Test phase completed",
                    "output_response": f"Tests: {test_result.total_tests}, Failed: {test_result.failed_tests}, Passed: {test_result.passed}"
                },
                status="SUCCESS" if test_result.passed else "FAILURE"
            )
            
            return state
            
        except Exception as e:
            # Use error handler for comprehensive error handling
            context = {
                "phase": "test",
                "test_directory": agent_state.sandbox_directory,
                "iteration": state.get("current_iteration", 0)
            }
            
            state = state["error_handler"].handle_error(e, context, state)
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Test phase failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _evaluate_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate test results and determine next action.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with evaluation results
        """
        try:
            test_result = state["test_results"]
            current_iteration = state["current_iteration"]
            max_iterations = state["max_iterations"]
            
            # Increment iteration count
            state["current_iteration"] = current_iteration + 1
            state["agent_state"] = state["state_manager"].increment_iteration(
                state["agent_state"]
            )
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Evaluating results - iteration {state['current_iteration']}",
                    "output_response": f"Tests passed: {test_result.passed if test_result else False}"
                },
                status="SUCCESS"
            )
            
            # Determine next action based on test results and iteration count
            if test_result and test_result.passed:
                # All tests passed - complete successfully
                state["should_continue"] = False
                state["agent_state"] = state["state_manager"].transition_state(
                    state["agent_state"], AgentStatus.COMPLETE
                )
                
                log_experiment(
                    agent_name="Orchestrator",
                    model=self.model,
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": "All tests passed",
                        "output_response": f"Workflow completed successfully after {state['current_iteration']} iterations"
                    },
                    status="SUCCESS"
                )
                
            elif state["current_iteration"] >= max_iterations:
                # Max iterations reached - terminate
                state["should_continue"] = False
                state["agent_state"] = state["state_manager"].transition_state(
                    state["agent_state"], AgentStatus.FAILED, 
                    f"Maximum iterations ({max_iterations}) reached"
                )
                
                log_experiment(
                    agent_name="Orchestrator",
                    model=self.model,
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": "Maximum iterations reached",
                        "output_response": f"Terminating after {max_iterations} iterations"
                    },
                    status="FAILURE"
                )
                
            else:
                # Continue with next iteration
                state["should_continue"] = True
                
                # Generate feedback for the next iteration
                if test_result:
                    judge = state["judge"]
                    feedback = judge.generate_detailed_feedback_for_iteration(
                        test_result, state["current_iteration"]
                    )
                    
                    log_experiment(
                        agent_name="Orchestrator",
                        model=self.model,
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": f"Continuing to iteration {state['current_iteration'] + 1}",
                            "output_response": f"Generated feedback for {test_result.failed_tests} failed tests"
                        },
                        status="SUCCESS"
                    )
            
            return state
            
        except Exception as e:
            # Use error handler for comprehensive error handling
            context = {
                "phase": "evaluate",
                "iteration": state.get("current_iteration", 0),
                "test_passed": test_result.passed if test_result else False
            }
            
            state = state["error_handler"].handle_error(e, context, state)
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Evaluation phase failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """
        Finalize the workflow and clean up resources.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final workflow state
        """
        try:
            agent_state = state["agent_state"]
            test_result = state.get("test_results")
            
            # Generate completion report
            if state.get("feedback_loop_state"):
                judge = state["judge"]
                completion_report = judge.generate_loop_completion_report(
                    state["feedback_loop_state"]
                )
                
                log_experiment(
                    agent_name="Orchestrator",
                    model=self.model,
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": "Generating completion report",
                        "output_response": completion_report[:500] + "..." if len(completion_report) > 500 else completion_report
                    },
                    status="SUCCESS"
                )
            
            # Clean up sandbox if needed
            if state.get("sandbox_manager"):
                try:
                    state["sandbox_manager"].cleanup_sandbox()
                except Exception as e:
                    # Log cleanup error but don't fail the workflow
                    log_experiment(
                        agent_name="Orchestrator",
                        model=self.model,
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": "Sandbox cleanup failed",
                            "output_response": f"Warning: {str(e)}"
                        },
                        status="FAILURE"
                    )
            
            # Perform graceful cleanup using error handler
            if state.get("error_handler"):
                state["error_handler"].perform_graceful_cleanup(state)
            
            # Final status log
            final_status = "SUCCESS" if agent_state.status == AgentStatus.COMPLETE else "FAILURE"
            final_message = f"Workflow completed with status: {agent_state.status.value}"
            
            if test_result:
                final_message += f" - Final test results: {test_result.total_tests - test_result.failed_tests}/{test_result.total_tests} passed"
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Workflow finalization",
                    "output_response": final_message
                },
                status=final_status
            )
            
            return state
            
        except Exception as e:
            state["last_error"] = f"Finalization failed: {str(e)}"
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Workflow finalization failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return state
    
    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """
        Handle workflow errors and attempt recovery.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state after error handling
        """
        try:
            last_error = state.get("last_error", "Unknown error")
            error_count = state.get("error_count", 0)
            
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Handling workflow error (count: {error_count})",
                    "output_response": f"Error: {last_error}"
                },
                status="FAILURE"
            )
            
            # Update agent state to failed
            if state.get("state_manager") and state.get("agent_state"):
                state["agent_state"] = state["state_manager"].transition_state(
                    state["agent_state"], AgentStatus.FAILED, last_error
                )
            
            # Stop workflow execution
            state["should_continue"] = False
            
            # Attempt cleanup
            if state.get("sandbox_manager"):
                try:
                    state["sandbox_manager"].cleanup_sandbox()
                except Exception as cleanup_error:
                    log_experiment(
                        agent_name="Orchestrator",
                        model=self.model,
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": "Error cleanup failed",
                            "output_response": f"Cleanup error: {str(cleanup_error)}"
                        },
                        status="FAILURE"
                    )
            
            return state
            
        except Exception as e:
            # If error handling itself fails, just log and return
            log_experiment(
                agent_name="Orchestrator",
                model=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Error handler failed",
                    "output_response": f"Critical error: {str(e)}"
                },
                status="FAILURE"
            )
            
            state["should_continue"] = False
            return state
    
    def _should_continue_routing(self, state: WorkflowState) -> Literal["continue", "complete", "terminate", "error"]:
        """
        Determine the next route based on current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next route to take in the workflow
        """
        # Check for errors first
        if state.get("error_count", 0) > 0 or not state.get("should_continue", True):
            if state.get("last_error"):
                return "error"
            else:
                return "terminate"
        
        # Check test results
        test_result = state.get("test_results")
        if test_result and test_result.passed:
            return "complete"
        
        # Check iteration limit
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        
        if current_iteration >= max_iterations:
            return "terminate"
        
        # Continue with next iteration
        return "continue"