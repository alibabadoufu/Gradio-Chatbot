#!/usr/bin/env python3
"""
Example script demonstrating the LangGraph OWL multi-agent system.
This script shows how to use the system with various tools and streaming capabilities.
"""

import os
import asyncio
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from core import create_owl_system, LangGraphOwlSystem
from tools import create_default_tools, create_search_tools, create_code_tools

def main():
    """Main function demonstrating the LangGraph OWL system"""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return
    
    print("🦉 LangGraph OWL Multi-Agent System Example")
    print("=" * 50)
    
    # Create tools
    print("Creating tools...")
    tools = create_search_tools() + create_code_tools()
    
    # Create OWL system
    print("Creating OWL system...")
    owl_system = create_owl_system(
        model_name="gpt-4o-mini",  # Use mini for faster/cheaper testing
        temperature=0.0,
        max_rounds=10,
        tools=tools,
        streaming=True,
        verbose=True
    )
    
    # Example task
    task = """
    Search for information about the LangGraph library on the web and then write a simple Python script that demonstrates a basic state machine. Save the script to a file called 'langgraph_demo.py'.
    """
    
    print(f"Task: {task}")
    print("\nStarting multi-agent collaboration...")
    print("-" * 50)
    
    # Run the system with streaming
    print("Running with streaming...")
    for i, chunk in enumerate(owl_system.stream(task)):
        print(f"\nStep {i+1}: {chunk}")
        
        # Extract and display the current state
        if isinstance(chunk, dict):
            for node_name, node_data in chunk.items():
                if node_name == "user_agent" and "user_agent_response" in node_data:
                    print(f"👤 User Agent: {node_data['user_agent_response'][:200]}...")
                elif node_name == "assistant_agent" and "assistant_agent_response" in node_data:
                    print(f"🤖 Assistant Agent: {node_data['assistant_agent_response'][:200]}...")
                elif node_name == "check_completion" and "task_completed" in node_data:
                    print(f"✅ Task completed: {node_data['task_completed']}")
    
    print("\nDemo completed!")


async def async_main():
    """Async version of the main function"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return
    
    print("🦉 LangGraph OWL Multi-Agent System (Async Example)")
    print("=" * 50)
    
    # Create tools
    print("Creating tools...")
    tools = create_search_tools() + create_code_tools()
    
    # Create OWL system
    print("Creating OWL system...")
    owl_system = create_owl_system(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_rounds=10,
        tools=tools,
        streaming=True,
        verbose=True
    )
    
    # Example task
    task = """
    Write a Python function that calculates the factorial of a number using recursion. 
    Then create a test script that tests this function with different inputs.
    Save both the function and test script to separate files.
    """
    
    print(f"Task: {task}")
    print("\nStarting async multi-agent collaboration...")
    print("-" * 50)
    
    # Run the system with async streaming
    print("Running with async streaming...")
    i = 0
    async for chunk in owl_system.astream(task):
        i += 1
        print(f"\nAsync Step {i}: {chunk}")
        
        # Extract and display the current state
        if isinstance(chunk, dict):
            for node_name, node_data in chunk.items():
                if node_name == "user_agent" and "user_agent_response" in node_data:
                    print(f"👤 User Agent: {node_data['user_agent_response'][:200]}...")
                elif node_name == "assistant_agent" and "assistant_agent_response" in node_data:
                    print(f"🤖 Assistant Agent: {node_data['assistant_agent_response'][:200]}...")
    
    print("\nAsync demo completed!")


def simple_run_example():
    """Simple non-streaming example"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        return
    
    print("🦉 LangGraph OWL Multi-Agent System (Simple Example)")
    print("=" * 50)
    
    # Create a simple set of tools
    tools = create_code_tools()
    
    # Create OWL system
    owl_system = create_owl_system(
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_rounds=5,
        tools=tools,
        streaming=False,
        verbose=True
    )
    
    # Simple task
    task = "Write a Python script that prints 'Hello, LangGraph OWL!' and saves it to a file."
    
    print(f"Task: {task}")
    print("\nRunning simple example...")
    print("-" * 50)
    
    # Run the system
    final_answer, chat_history, token_usage = owl_system.run(task)
    
    print(f"\nFinal Answer: {final_answer}")
    print(f"Chat History: {len(chat_history)} rounds")
    print(f"Token Usage: {token_usage}")
    
    # Display chat history
    print("\nChat History:")
    for i, round_data in enumerate(chat_history):
        print(f"\nRound {i+1}:")
        print(f"👤 User: {round_data.get('user', '')[:100]}...")
        print(f"🤖 Assistant: {round_data.get('assistant', '')[:100]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "async":
            print("Running async example...")
            asyncio.run(async_main())
        elif sys.argv[1] == "simple":
            print("Running simple example...")
            simple_run_example()
        else:
            print("Unknown argument. Use 'async' or 'simple'")
    else:
        print("Running streaming example...")
        main()