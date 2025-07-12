#!/usr/bin/env python3
"""
Streaming web application for the LangGraph OWL multi-agent system.
This provides a Gradio interface with real-time streaming capabilities.
"""

import os
import gradio as gr
import asyncio
import threading
import queue
import time
import json
import logging
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from core import create_owl_system, LangGraphOwlSystem
from tools import (
    create_default_tools, create_search_tools, create_code_tools,
    create_analysis_tools, create_browser_tools, create_comprehensive_toolkit
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
CURRENT_OWL_SYSTEM: Optional[LangGraphOwlSystem] = None
CURRENT_TASK_QUEUE: queue.Queue = queue.Queue()
CURRENT_RESULT_QUEUE: queue.Queue = queue.Queue()
CURRENT_THREAD: Optional[threading.Thread] = None
STOP_REQUESTED = threading.Event()


def validate_environment() -> tuple[bool, str]:
    """Validate that required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        return False, "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    return True, "Environment validated successfully."


def create_owl_system_with_tools(
    model_name: str = "gpt-4o-mini",
    tool_selection: str = "comprehensive",
    max_rounds: int = 10
) -> LangGraphOwlSystem:
    """Create OWL system with selected tools"""
    
    # Select tools based on user choice
    if tool_selection == "search":
        tools = create_search_tools()
    elif tool_selection == "code":
        tools = create_code_tools()
    elif tool_selection == "analysis":
        tools = create_analysis_tools()
    elif tool_selection == "browser":
        tools = create_browser_tools()
    elif tool_selection == "comprehensive":
        tools = create_comprehensive_toolkit()
    else:
        tools = create_default_tools()
    
    # Create OWL system
    owl_system = create_owl_system(
        model_name=model_name,
        temperature=0.0,
        max_rounds=max_rounds,
        tools=tools,
        streaming=True,
        verbose=True
    )
    
    return owl_system


def process_task_streaming(
    task: str,
    model_name: str = "gpt-4o-mini",
    tool_selection: str = "comprehensive",
    max_rounds: int = 10
) -> Generator[tuple[str, str, str], None, None]:
    """Process task with streaming updates"""
    
    # Validate environment
    valid, message = validate_environment()
    if not valid:
        yield "❌ Error", message, ""
        return
    
    try:
        # Create OWL system
        yield "🔄 Initializing", "Creating OWL system...", ""
        owl_system = create_owl_system_with_tools(model_name, tool_selection, max_rounds)
        
        yield "🔄 Processing", "Starting multi-agent collaboration...", ""
        
        # Process with streaming
        conversation_log = []
        step_count = 0
        
        for chunk in owl_system.stream(task):
            step_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if isinstance(chunk, dict):
                # Process each node's output
                for node_name, node_data in chunk.items():
                    if node_name == "user_agent" and "user_agent_response" in node_data:
                        user_response = node_data["user_agent_response"]
                        conversation_log.append(f"[{timestamp}] 👤 **User Agent**: {user_response}")
                        
                        # Check for task completion
                        if "TASK_DONE" in user_response:
                            conversation_log.append(f"[{timestamp}] ✅ **Task Completed**")
                            yield "✅ Completed", "Task completed successfully!", "\n".join(conversation_log)
                            return
                    
                    elif node_name == "assistant_agent" and "assistant_agent_response" in node_data:
                        assistant_response = node_data["assistant_agent_response"]
                        conversation_log.append(f"[{timestamp}] 🤖 **Assistant Agent**: {assistant_response}")
                        
                        # Check for tool calls
                        if "tool_calls" in node_data:
                            tool_calls = node_data["tool_calls"]
                            for tool_call in tool_calls:
                                tool_name = tool_call.get("name", "unknown")
                                conversation_log.append(f"[{timestamp}] 🔧 **Tool Called**: {tool_name}")
                    
                    elif node_name == "check_completion":
                        if node_data.get("task_completed"):
                            conversation_log.append(f"[{timestamp}] ✅ **Task Completed**")
                            yield "✅ Completed", "Task completed successfully!", "\n".join(conversation_log)
                            return
            
            # Yield current progress
            status = f"🔄 Processing (Step {step_count})"
            yield status, f"Multi-agent collaboration in progress...", "\n".join(conversation_log)
        
        # Final yield
        yield "✅ Completed", "Processing finished!", "\n".join(conversation_log)
        
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        logger.error(error_msg)
        yield "❌ Error", error_msg, ""


def create_gradio_interface() -> gr.Blocks:
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="🦉 LangGraph OWL Multi-Agent System",
        theme=gr.themes.Soft()
    ) as app:
        
        # Header
        gr.Markdown("""
        # 🦉 LangGraph OWL Multi-Agent System
        
        Advanced multi-agent collaboration system built with LangGraph, providing the same functionality as the original OWL system with real-time streaming capabilities.
        
        **Features:**
        - Multi-agent collaboration between User and Assistant agents
        - Real-time streaming of agent interactions
        - Comprehensive toolkit with search, code execution, analysis, and more
        - Support for various language models
        """)
        
        # Environment status
        env_valid, env_message = validate_environment()
        env_status = gr.Markdown(
            f"**Environment Status:** {'✅' if env_valid else '❌'} {env_message}"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📝 Task Input")
                
                task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Enter your task here...",
                    lines=4,
                    value="Write a Python script that creates a simple calculator with basic operations (add, subtract, multiply, divide). Save it to a file called 'calculator.py' and then test it with some example calculations."
                )
                
                # Configuration
                gr.Markdown("## ⚙️ Configuration")
                
                model_dropdown = gr.Dropdown(
                    choices=[
                        "gpt-4o-mini",
                        "gpt-4o",
                        "gpt-3.5-turbo",
                        "gpt-4",
                        "gpt-4-turbo"
                    ],
                    value="gpt-4o-mini",
                    label="Model",
                    info="Select the language model to use"
                )
                
                tool_dropdown = gr.Dropdown(
                    choices=[
                        "comprehensive",
                        "search",
                        "code",
                        "analysis",
                        "browser",
                        "default"
                    ],
                    value="comprehensive",
                    label="Tool Selection",
                    info="Choose which tools to provide to the agents"
                )
                
                max_rounds_slider = gr.Slider(
                    minimum=3,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Max Rounds",
                    info="Maximum number of conversation rounds"
                )
                
                # Control buttons
                process_button = gr.Button(
                    "🚀 Start Processing",
                    variant="primary",
                    size="lg"
                )
                
                # Examples
                gr.Markdown("## 📚 Example Tasks")
                example_tasks = [
                    "Search for information about LangGraph and write a summary report.",
                    "Create a data analysis script that processes a CSV file and generates visualizations.",
                    "Write a web scraper that extracts product information from a website.",
                    "Develop a machine learning model training script with proper validation.",
                    "Create a REST API using FastAPI with CRUD operations."
                ]
                
                gr.Examples(
                    examples=example_tasks,
                    inputs=[task_input],
                    label="Click to use example tasks"
                )
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("## 📊 Real-time Results")
                
                # Status display
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready to process tasks",
                    interactive=False
                )
                
                # Progress display
                progress_display = gr.Textbox(
                    label="Progress",
                    value="",
                    interactive=False
                )
                
                # Conversation log
                conversation_display = gr.Textbox(
                    label="Agent Conversation Log",
                    value="",
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Event handlers
        def process_task_wrapper(task, model, tools, max_rounds):
            """Wrapper for processing tasks with streaming"""
            for status, progress, conversation in process_task_streaming(
                task, model, tools, max_rounds
            ):
                yield status, progress, conversation
        
        # Set up streaming
        process_button.click(
            fn=process_task_wrapper,
            inputs=[task_input, model_dropdown, tool_dropdown, max_rounds_slider],
            outputs=[status_display, progress_display, conversation_display],
            show_progress=True
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        **Note:** This is a LangGraph implementation of the OWL multi-agent system. 
        The system uses two AI agents that collaborate to solve complex tasks:
        - **User Agent**: Provides step-by-step instructions
        - **Assistant Agent**: Executes tasks using available tools
        
        For more information, visit the [LangGraph documentation](https://langchain-ai.github.io/langgraph/).
        """)
    
    return app


def main():
    """Main function to run the web application"""
    
    print("🦉 Starting LangGraph OWL Multi-Agent System Web App")
    print("=" * 60)
    
    # Validate environment
    valid, message = validate_environment()
    if not valid:
        print(f"❌ Environment validation failed: {message}")
        print("Please set your OPENAI_API_KEY environment variable and try again.")
        return
    
    print("✅ Environment validation passed")
    
    # Create and launch the interface
    app = create_gradio_interface()
    
    print("🚀 Launching web application...")
    print("📱 Open your browser and navigate to the provided URL")
    print("💡 Tip: Use the comprehensive toolkit for the best experience")
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()