"""
Enhanced integration between the UI components and the backend LLM and RAG API.
This module provides handler functions for the GenAI Chat Application.
"""

import gradio as gr
from typing import Dict, Any, List, Tuple, Optional, Generator
import time

from src.api.llm_api import LLMApi
from src.core.langgraph_setup import create_initial_state
from src.utils.helpers import parse_citations_from_thoughts, format_citation


# Chat submission handler with streaming response
def on_chat_submit(
    user_message: str,
    chat_history: List[List[str]],
    model: str,
    temperature: float,
    top_k: int,
    document_tags: List[str],
    focus_mode: str,
    selected_documents: List[str],
    recency_bias: bool,
    llm_api: LLMApi,
    config: Dict[str, Any]
) -> Tuple[str, List[List[str]], gr.update, str, gr.update]:
    """Handle chat submission and generate streaming response."""
    if not user_message.strip():
        return "", chat_history, gr.update(visible=False), "", gr.update(visible=False)
    
    # Add user message to chat history
    chat_history = chat_history + [[user_message, None]]
    
    # Get AI thoughts
    system_message = config["ui_messages"]["system_message"]
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Retrieve AI thoughts
        thoughts = llm_api.get_thoughts(
            messages=messages,
            model=model,
            document_tags=document_tags,
            focus_mode=focus_mode
        )
        
        # Format thoughts for display
        thoughts_markdown = format_thoughts_for_display(thoughts)
        
        # Format citations
        citations = parse_citations_from_thoughts(
            thoughts, 
            config["api"]["sharepoint_base_url"]
        )
        
        # Start streaming response
        response_stream = llm_api.generate_response(
            messages=messages,
            model=model,
            temperature=temperature,
            top_k=top_k,
            document_tags=document_tags,
            focus_mode=focus_mode,
            selected_documents=selected_documents,
            recency_bias=recency_bias
        )
        
        # In a real implementation, this would stream to the UI
        # For now, we'll collect the entire response
        response = "".join(list(response_stream))
        
        # Add citations to response if available
        if citations:
            response += "\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                response += f"{i}. [{citation['title']}]({citation['url']})\n"
        
        # Update chat history with AI response
        chat_history[-1][1] = response
        
        return "", chat_history, gr.update(visible=True), thoughts_markdown, gr.update(visible=True)
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        chat_history[-1][1] = error_message
        return "", chat_history, gr.update(visible=False), "", gr.update(visible=False)


# Regenerate response handler
def on_regenerate(
    chat_history: List[List[str]],
    model: str,
    temperature: float,
    top_k: int,
    document_tags: List[str],
    focus_mode: str,
    selected_documents: List[str],
    recency_bias: bool,
    llm_api: LLMApi,
    config: Dict[str, Any]
) -> Tuple[List[List[str]], gr.update, str, gr.update]:
    """Handle regeneration of the last AI response."""
    if not chat_history:
        return chat_history, gr.update(visible=False), "", gr.update(visible=False)
    
    # Get the last user message
    last_user_message = chat_history[-1][0]
    
    # Remove the last AI response
    chat_history[-1][1] = None
    
    # Get AI thoughts
    system_message = config["ui_messages"]["system_message"]
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": last_user_message}
    ]
    
    try:
        # Retrieve AI thoughts
        thoughts = llm_api.get_thoughts(
            messages=messages,
            model=model,
            document_tags=document_tags,
            focus_mode=focus_mode
        )
        
        # Format thoughts for display
        thoughts_markdown = format_thoughts_for_display(thoughts)
        
        # Format citations
        citations = parse_citations_from_thoughts(
            thoughts, 
            config["api"]["sharepoint_base_url"]
        )
        
        # Start streaming response
        response_stream = llm_api.generate_response(
            messages=messages,
            model=model,
            temperature=temperature,
            top_k=top_k,
            document_tags=document_tags,
            focus_mode=focus_mode,
            selected_documents=selected_documents,
            recency_bias=recency_bias
        )
        
        # In a real implementation, this would stream to the UI
        # For now, we'll collect the entire response
        response = "".join(list(response_stream))
        
        # Add citations to response if available
        if citations:
            response += "\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                response += f"{i}. [{citation['title']}]({citation['url']})\n"
        
        # Update chat history with AI response
        chat_history[-1][1] = response
        
        return chat_history, gr.update(visible=True), thoughts_markdown, gr.update(visible=True)
        
    except Exception as e:
        error_message = f"Error regenerating response: {str(e)}"
        chat_history[-1][1] = error_message
        return chat_history, gr.update(visible=False), "", gr.update(visible=False)


# Feedback submission handler
def on_feedback_submit(
    feedback_rating: str,
    feedback_text: str,
    chat_history: List[List[str]]
) -> Tuple[gr.update, gr.update]:
    """Handle feedback submission."""
    from src.utils.helpers import log_user_feedback
    
    # Get the last exchange for context
    last_exchange = chat_history[-1] if chat_history else ["", ""]
    
    feedback = {
        "rating": feedback_rating,
        "comments": feedback_text,
        "user_message": last_exchange[0],
        "ai_response": last_exchange[1],
        "timestamp": time.time()
    }
    
    log_user_feedback(feedback)
    
    return gr.update(value=None), gr.update(value="")


def format_thoughts_for_display(thoughts: Dict[str, Any]) -> str:
    """
    Format the AI's thoughts for display in the UI.
    
    Args:
        thoughts: The AI's thoughts and reasoning steps.
        
    Returns:
        Formatted markdown string.
    """
    markdown = "### AI Reasoning Process\n\n"
    
    if "reasoning" in thoughts:
        markdown += f"{thoughts['reasoning']}\n\n"
    
    if "retrieved_chunks" in thoughts and thoughts["retrieved_chunks"]:
        markdown += "### Retrieved Document Chunks\n\n"
        
        for i, chunk in enumerate(thoughts["retrieved_chunks"], 1):
            markdown += f"**Document {i}:** {chunk['document_title']}\n\n"
            markdown += f"```\n{chunk['text']}\n```\n\n"
    
    return markdown


def simulate_streaming_response(
    response: str,
    chat_history: List[List[str]],
    chunk_size: int = 10,
    delay: float = 0.05
) -> Generator[List[List[str]], None, None]:
    """
    Simulate a streaming response for demonstration purposes.
    
    Args:
        response: The complete response to stream.
        chat_history: The current chat history.
        chunk_size: The number of characters to yield at once.
        delay: The delay between chunks in seconds.
        
    Yields:
        Updated chat history with partial response.
    """
    partial_response = ""
    
    for i in range(0, len(response), chunk_size):
        partial_response += response[i:i+chunk_size]
        chat_history[-1][1] = partial_response
        time.sleep(delay)
        yield chat_history
