"""
🦉 LangGraph OWL Multi-Agent System

A LangGraph implementation of the OWL (Optimized Workforce Learning) multi-agent system
that provides powerful multi-agent collaboration capabilities with real-time streaming support.

This package includes:
- Core multi-agent system using LangGraph
- Comprehensive toolkit with search, code execution, analysis, and more
- Real-time streaming capabilities
- Web interface powered by Gradio
- Async support for high-performance applications
"""

__version__ = "1.0.0"
__author__ = "LangGraph OWL Team"
__license__ = "Apache 2.0"

# Import main components for easy access
from .core import (
    LangGraphOwlSystem,
    LangGraphOwlConfig,
    AgentState,
    AgentRole,
    create_owl_system
)

from .tools import (
    # Individual tools
    SearchTool,
    WikipediaSearchTool,
    CodeExecutionTool,
    FileWriteTool,
    WebScrapingTool,
    MathCalculationTool,
    ImageAnalysisTool,
    DataAnalysisTool,
    BrowserAutomationTool,
    
    # Tool collection functions
    create_default_tools,
    create_search_tools,
    create_code_tools,
    create_analysis_tools,
    create_browser_tools,
    create_comprehensive_toolkit,
    
    # Pre-built tool collections
    DEFAULT_TOOLS,
    SEARCH_TOOLS,
    CODE_TOOLS,
    ANALYSIS_TOOLS,
    BROWSER_TOOLS,
    COMPREHENSIVE_TOOLS
)

# Export all important components
__all__ = [
    # Core system
    "LangGraphOwlSystem",
    "LangGraphOwlConfig", 
    "AgentState",
    "AgentRole",
    "create_owl_system",
    
    # Individual tools
    "SearchTool",
    "WikipediaSearchTool",
    "CodeExecutionTool",
    "FileWriteTool",
    "WebScrapingTool",
    "MathCalculationTool",
    "ImageAnalysisTool",
    "DataAnalysisTool",
    "BrowserAutomationTool",
    
    # Tool collection functions
    "create_default_tools",
    "create_search_tools",
    "create_code_tools",
    "create_analysis_tools",
    "create_browser_tools",
    "create_comprehensive_toolkit",
    
    # Pre-built tool collections
    "DEFAULT_TOOLS",
    "SEARCH_TOOLS",
    "CODE_TOOLS",
    "ANALYSIS_TOOLS",
    "BROWSER_TOOLS",
    "COMPREHENSIVE_TOOLS"
]


def get_version() -> str:
    """Get the current version of the package."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "langgraph-owl",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "LangGraph implementation of the OWL multi-agent system",
        "features": [
            "Multi-agent collaboration",
            "Real-time streaming",
            "Comprehensive toolkit",
            "Web interface",
            "Async support",
            "LangGraph integration"
        ]
    }


# Quick start example
def quick_start_example():
    """
    Quick start example for the LangGraph OWL system.
    
    This function provides a simple example of how to use the system.
    """
    example_code = '''
# Quick Start Example for LangGraph OWL

from langgraph_owl import create_owl_system, create_comprehensive_toolkit

# Create OWL system
owl_system = create_owl_system(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_rounds=10,
    tools=create_comprehensive_toolkit(),
    streaming=True
)

# Run a task
task = "Write a Python script that prints 'Hello, LangGraph OWL!' and saves it to a file."
answer, chat_history, token_usage = owl_system.run(task)

print(f"Answer: {answer}")
print(f"Rounds: {len(chat_history)}")
print(f"Token Usage: {token_usage}")

# Stream a task
print("\\nStreaming example:")
for chunk in owl_system.stream(task):
    print(f"Update: {chunk}")
'''
    
    print(example_code)
    return example_code


if __name__ == "__main__":
    print("🦉 LangGraph OWL Multi-Agent System")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    
    # Handle potential None value for __doc__
    doc_string = __doc__ or "LangGraph OWL Multi-Agent System"
    print("\n" + doc_string)
    
    print("\nQuick Start Example:")
    print("-" * 20)
    quick_start_example()