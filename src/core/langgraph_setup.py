"""
Module for setting up and managing the Langgraph graph for orchestrating the AI workflow.
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

from src.api.llm_api import LLMApi


def setup_langgraph(llm_api: LLMApi, config: Dict[str, Any]) -> StateGraph:
    """
    Set up the Langgraph workflow for orchestrating the AI chat application.
    
    Args:
        llm_api: The LLM API client instance.
        config: The application configuration.
        
    Returns:
        A configured StateGraph instance.
    """
    # Define the state schema
    class State:
        """State schema for the Langgraph workflow."""
        messages: List[Dict[str, str]]
        model: str
        temperature: float
        top_k: int
        document_tags: List[str]
        focus_mode: str
        selected_documents: List[str]
        recency_bias: bool
        thoughts: Optional[Dict[str, Any]]
        response: str
    
    # Define the nodes in the graph
    def retrieve_thoughts(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve the AI's thoughts and reasoning steps.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            Updated state with thoughts.
        """
        thoughts = llm_api.get_thoughts(
            messages=state["messages"],
            model=state["model"],
            document_tags=state["document_tags"],
            focus_mode=state["focus_mode"]
        )
        
        return {"thoughts": thoughts}
    
    def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the AI's response.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            Updated state with response.
        """
        response_stream = llm_api.generate_response(
            messages=state["messages"],
            model=state["model"],
            temperature=state["temperature"],
            top_k=state["top_k"],
            document_tags=state["document_tags"],
            focus_mode=state["focus_mode"],
            selected_documents=state["selected_documents"],
            recency_bias=state["recency_bias"]
        )
        
        # In a real implementation, this would be streamed to the UI
        # For now, we'll collect the entire response
        response = "".join(list(response_stream))
        
        return {"response": response}
    
    def update_messages(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the message history with the AI's response.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            Updated state with new message history.
        """
        messages = state["messages"].copy()
        messages.append({"role": "assistant", "content": state["response"]})
        
        return {"messages": messages}
    
    # Create the graph
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("retrieve_thoughts", retrieve_thoughts)
    builder.add_node("generate_response", generate_response)
    builder.add_node("update_messages", update_messages)
    
    # Define the edges
    builder.add_edge("retrieve_thoughts", "generate_response")
    builder.add_edge("generate_response", "update_messages")
    builder.add_edge("update_messages", END)
    
    # Set the entry point
    builder.set_entry_point("retrieve_thoughts")
    
    # Compile the graph
    graph = builder.compile()
    
    return graph


def create_initial_state(
    user_message: str,
    model: str,
    temperature: float,
    top_k: int,
    document_tags: List[str],
    focus_mode: str,
    selected_documents: List[str],
    recency_bias: bool,
    system_message: str
) -> Dict[str, Any]:
    """
    Create the initial state for the Langgraph workflow.
    
    Args:
        user_message: The user's message.
        model: The LLM model to use.
        temperature: Controls randomness in generation.
        top_k: Controls diversity of output.
        document_tags: List of document tags to restrict RAG search.
        focus_mode: The operational mode of the chatbot.
        selected_documents: List of document titles for DocCompare mode.
        recency_bias: Whether to prioritize recently updated documents.
        system_message: The system message defining the chatbot's persona.
        
    Returns:
        The initial state dictionary.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    return {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "document_tags": document_tags,
        "focus_mode": focus_mode,
        "selected_documents": selected_documents,
        "recency_bias": recency_bias,
        "thoughts": None,
        "response": ""
    }
