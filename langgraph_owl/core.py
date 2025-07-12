"""
LangGraph implementation of the OWL multi-agent system.
This module provides the core multi-agent orchestration using LangGraph.
"""

from typing import Dict, List, Optional, Tuple, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
import asyncio
import logging
import json
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the multi-agent system"""
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    current_round: int
    max_rounds: int
    task_completed: bool
    user_agent_response: Optional[str]
    assistant_agent_response: Optional[str]
    tool_calls: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    chat_history: List[Dict[str, Any]]


class AgentRole(Enum):
    """Agent roles in the system"""
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LangGraphOwlConfig:
    """Configuration for the LangGraph OWL system"""
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_rounds: int = 15
    tools: List[BaseTool] = field(default_factory=list)
    streaming: bool = True
    verbose: bool = True


class LangGraphOwlSystem:
    """
    LangGraph implementation of the OWL multi-agent system.
    
    This class implements the core multi-agent orchestration logic using LangGraph,
    providing the same functionality as the original OWL system with streaming capabilities.
    """
    
    def __init__(self, config: LangGraphOwlConfig):
        self.config = config
        self.user_model = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        )
        self.assistant_model = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        )
        self.tools = config.tools
        
        # Bind tools to assistant model
        if self.tools:
            self.assistant_model = self.assistant_model.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
        
        # Memory for checkpointing
        self.memory = MemorySaver()
        
        # Compile the graph with checkpointing
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("user_agent", self._user_agent_node)
        graph.add_node("assistant_agent", self._assistant_agent_node)
        graph.add_node("check_completion", self._check_completion_node)
        
        # Add edges
        graph.add_edge(START, "user_agent")
        graph.add_edge("user_agent", "assistant_agent")
        graph.add_edge("assistant_agent", "check_completion")
        
        # Conditional edges from check_completion
        graph.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "user_agent",
                "end": END
            }
        )
        
        return graph
    
    def _get_user_system_prompt(self, task: str) -> str:
        """Get the system prompt for the user agent"""
        return f"""
===== RULES OF USER =====
Never forget you are a user and I am an assistant. Never flip roles! You will always instruct me. We share a common interest in collaborating to successfully complete a task.
I must help you to complete a difficult task.
You must instruct me based on my expertise and your needs to solve the task step by step. The format of your instruction is: `Instruction: [YOUR INSTRUCTION]`, where "Instruction" describes a sub-task or question.
You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
You should instruct me not ask me questions.

Please note that the task may be very complicated. Do not attempt to solve the task by single step. You must instruct me to find the answer step by step.
Here are some tips that will help you to give more valuable instructions about our task to me:
<tips>
- I have various tools to use, such as search toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will solve the task step-by-step, and give me instructions just like that. For example, one may first use google search to get some initial information and the target url, then retrieve the content of the url, or do some web browser interaction to find the answer.
- Although the task is complex, the answer does exist. If you can't find the answer using the current scheme, try to re-plan and use other ways to find the answer, e.g. using other tools or methods that can achieve similar results.
- Always remind me to verify my final answer about the overall task. This work can be done by using multiple tools(e.g., screenshots, webpage analysis, etc.), or something else.
- If I have written code, please remind me to run the code and get the result.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc.
- If the question mentions youtube video, in most cases you have to process the content of the mentioned video.
- For downloading files, you can either use the web browser simulation toolkit or write codes (for example, the github content can be downloaded via https://raw.githubusercontent.com/...).
- Flexibly write codes to solve some problems, such as excel relevant tasks.
</tips>

Now, here is the overall task: <task>{task}</task>. Never forget our task!

Now you must start to instruct me to solve the task step-by-step. Do not add anything else other than your instruction!
Keep giving me instructions until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task.
        """
    
    def _get_assistant_system_prompt(self, task: str) -> str:
        """Get the system prompt for the assistant agent"""
        return f"""
===== RULES OF ASSISTANT =====
Never forget you are an assistant and I am a user. Never flip roles! Never instruct me! You have to utilize your available tools to solve the task I assigned.
We share a common interest in collaborating to successfully complete a complex task.
You must help me to complete the task.

Here is our overall task: {task}. Never forget our task!

I must instruct you based on your expertise and my needs to complete the task. An instruction is typically a sub-task or question.

You must leverage your available tools, try your best to solve the problem, and explain your solutions.
Unless I say the task is completed, you should always start with:
Solution: [YOUR_SOLUTION]
[YOUR_SOLUTION] should be specific, including detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it. Also, bear in mind that the code execution environment does not support interactive input.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc.
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>
        """
    
    def _user_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """User agent node that provides instructions"""
        task = state["task"]
        current_round = state["current_round"]
        
        # Create system message for user agent
        system_prompt = self._get_user_system_prompt(task)
        system_message = SystemMessage(content=system_prompt)
        
        # Get conversation history
        messages = [system_message] + state["messages"]
        
        # If this is the first round, start with initial prompt
        if current_round == 0:
            initial_prompt = "Now please give me instructions to solve our overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task."
            messages.append(HumanMessage(content=initial_prompt))
        
        # Get user agent response
        try:
            response = self.user_model.invoke(messages)
            user_content = response.content
            
            logger.info(f"Round #{current_round} - User Agent: {user_content}")
            
            # Add auxiliary information if task is not done
            if user_content and "TASK_DONE" not in user_content:
                user_content += f"""

Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
<auxiliary_information>
{task}
</auxiliary_information>
If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
"""
            else:
                # Task is done, ask for final answer
                user_content += f"""

Now please make a final answer of the original task based on our conversation: <task>{task}</task>
"""
            
            return {
                "messages": [HumanMessage(content=user_content)],
                "user_agent_response": user_content,
                "current_round": current_round + 1
            }
            
        except Exception as e:
            logger.error(f"Error in user agent: {e}")
            return {
                "messages": [HumanMessage(content=f"Error in user agent: {e}")],
                "user_agent_response": f"Error: {e}",
                "current_round": current_round + 1
            }
    
    def _assistant_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Assistant agent node that executes tasks using tools"""
        task = state["task"]
        current_round = state["current_round"]
        
        # Create system message for assistant agent
        system_prompt = self._get_assistant_system_prompt(task)
        system_message = SystemMessage(content=system_prompt)
        
        # Get the latest user message
        user_message = state["messages"][-1]
        
        # Prepare messages for assistant
        messages = [system_message, user_message]
        
        try:
            # Get assistant response (with potential tool calls)
            response = self.assistant_model.invoke(messages)
            assistant_content = response.content
            
            logger.info(f"Round #{current_round} - Assistant Agent: {assistant_content}")
            
            # Handle tool calls if present
            tool_calls = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_calls.append({
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("name", ""),
                        "args": tool_call.get("args", {}),
                        "output": "Tool executed"  # Placeholder - actual tool execution would go here
                    })
            
            # Add guidance for next instruction if task is not done
            user_agent_response = state.get("user_agent_response", "")
            if user_agent_response and "TASK_DONE" not in user_agent_response:
                assistant_content += f"""

Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{task}</task>
Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
If I have written codes, remind me to run the codes.
If you think our task is done, reply with `TASK_DONE` to end our conversation.
"""
            
            return {
                "messages": [AIMessage(content=assistant_content)],
                "assistant_agent_response": assistant_content,
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            logger.error(f"Error in assistant agent: {e}")
            return {
                "messages": [AIMessage(content=f"Error in assistant agent: {e}")],
                "assistant_agent_response": f"Error: {e}",
                "tool_calls": []
            }
    
    def _check_completion_node(self, state: AgentState) -> Dict[str, Any]:
        """Check if the task is completed"""
        user_response = state.get("user_agent_response", "")
        current_round = state["current_round"]
        max_rounds = state["max_rounds"]
        
        # Check completion conditions
        task_completed = (
            (user_response and "TASK_DONE" in user_response) or
            current_round >= max_rounds
        )
        
        # Update chat history
        chat_history = state.get("chat_history", [])
        chat_history.append({
            "round": current_round,
            "user": user_response,
            "assistant": state.get("assistant_agent_response", ""),
            "tool_calls": state.get("tool_calls", [])
        })
        
        return {
            "task_completed": task_completed,
            "chat_history": chat_history
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the conversation should continue"""
        if state.get("task_completed", False):
            return "end"
        return "continue"
    
    def run(self, task: str, thread_id: str = "default") -> Tuple[str, List[Dict], Dict]:
        """
        Run the multi-agent system synchronously
        
        Args:
            task: The task to be solved
            thread_id: Thread ID for checkpointing
            
        Returns:
            Tuple of (final_answer, chat_history, token_usage)
        """
        # Initialize state
        initial_state = AgentState(
            messages=[],
            task=task,
            current_round=0,
            max_rounds=self.config.max_rounds,
            task_completed=False,
            user_agent_response=None,
            assistant_agent_response=None,
            tool_calls=[],
            token_usage={"completion_tokens": 0, "prompt_tokens": 0},
            chat_history=[]
        )
        
        # Run the graph
        config = RunnableConfig(configurable={"thread_id": thread_id})
        final_state = self.compiled_graph.invoke(initial_state, config=config)
        
        # Extract results
        chat_history = final_state.get("chat_history", [])
        final_answer = ""
        if chat_history:
            final_answer = chat_history[-1].get("assistant", "")
        
        token_usage = final_state.get("token_usage", {})
        
        return final_answer, chat_history, token_usage
    
    async def arun(self, task: str, thread_id: str = "default") -> Tuple[str, List[Dict], Dict]:
        """
        Run the multi-agent system asynchronously
        
        Args:
            task: The task to be solved
            thread_id: Thread ID for checkpointing
            
        Returns:
            Tuple of (final_answer, chat_history, token_usage)
        """
        # Initialize state
        initial_state = AgentState(
            messages=[],
            task=task,
            current_round=0,
            max_rounds=self.config.max_rounds,
            task_completed=False,
            user_agent_response=None,
            assistant_agent_response=None,
            tool_calls=[],
            token_usage={"completion_tokens": 0, "prompt_tokens": 0},
            chat_history=[]
        )
        
        # Run the graph asynchronously
        config = RunnableConfig(configurable={"thread_id": thread_id})
        final_state = await self.compiled_graph.ainvoke(initial_state, config=config)
        
        # Extract results
        chat_history = final_state.get("chat_history", [])
        final_answer = ""
        if chat_history:
            final_answer = chat_history[-1].get("assistant", "")
        
        token_usage = final_state.get("token_usage", {})
        
        return final_answer, chat_history, token_usage
    
    def stream(self, task: str, thread_id: str = "default"):
        """
        Stream the multi-agent system execution
        
        Args:
            task: The task to be solved
            thread_id: Thread ID for checkpointing
            
        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = AgentState(
            messages=[],
            task=task,
            current_round=0,
            max_rounds=self.config.max_rounds,
            task_completed=False,
            user_agent_response=None,
            assistant_agent_response=None,
            tool_calls=[],
            token_usage={"completion_tokens": 0, "prompt_tokens": 0},
            chat_history=[]
        )
        
        # Stream the graph execution
        config = RunnableConfig(configurable={"thread_id": thread_id})
        
        for state_update in self.compiled_graph.stream(initial_state, config=config):
            yield state_update
    
    async def astream(self, task: str, thread_id: str = "default"):
        """
        Stream the multi-agent system execution asynchronously
        
        Args:
            task: The task to be solved
            thread_id: Thread ID for checkpointing
            
        Yields:
            State updates during execution
        """
        # Initialize state
        initial_state = AgentState(
            messages=[],
            task=task,
            current_round=0,
            max_rounds=self.config.max_rounds,
            task_completed=False,
            user_agent_response=None,
            assistant_agent_response=None,
            tool_calls=[],
            token_usage={"completion_tokens": 0, "prompt_tokens": 0},
            chat_history=[]
        )
        
        # Stream the graph execution asynchronously
        config = RunnableConfig(configurable={"thread_id": thread_id})
        
        async for state_update in self.compiled_graph.astream(initial_state, config=config):
            yield state_update


def create_owl_system(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_rounds: int = 15,
    tools: Optional[List[BaseTool]] = None,
    streaming: bool = True,
    verbose: bool = True
) -> LangGraphOwlSystem:
    """
    Factory function to create a LangGraph OWL system
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature for the model
        max_rounds: Maximum number of conversation rounds
        tools: List of tools to provide to the assistant
        streaming: Whether to enable streaming
        verbose: Whether to enable verbose logging
        
    Returns:
        LangGraphOwlSystem instance
    """
    config = LangGraphOwlConfig(
        model_name=model_name,
        temperature=temperature,
        max_rounds=max_rounds,
        tools=tools or [],
        streaming=streaming,
        verbose=verbose
    )
    
    return LangGraphOwlSystem(config)