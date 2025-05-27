#!/usr/bin/env python3
"""
GenAI Chat Application - Main application entry point

This is the main entry point for the GenAI Chat Application, a Gradio-based chat interface
that leverages an in-house Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG)
framework to provide users with an advanced document interaction experience.
"""

import os
import yaml
import gradio as gr
import argparse
import logging

from src.ui.layout import create_layout
from src.ui.components import create_components
from src.ui.integration import integrate_llm_with_ui
from src.core.langgraph_setup import setup_langgraph
from src.api.llm_api import LLMApi
from src.utils.helpers import load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_application_config(config_path=None):
    """
    Load the application configuration from config.yaml
    
    Args:
        config_path: Optional path to the configuration file. If not provided,
                    the default path will be used.
    
    Returns:
        The configuration as a dictionary.
    """
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    
    logger.info(f"Loading configuration from {config_path}")
    return load_config(config_path)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="GenAI Chat Application")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the Gradio server on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the Gradio server on"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load configuration
    config = load_application_config(args.config)
    
    # Initialize API client
    logger.info(f"Initializing LLM API client with URL: {config['api']['llm_url']}")
    llm_api = LLMApi(config["api"]["llm_url"])
    
    # Setup Langgraph
    logger.info("Setting up Langgraph workflow")
    graph = setup_langgraph(llm_api, config)
    
    # Create UI components
    logger.info("Creating UI components")
    components = create_components(config)
    
    # Integrate LLM with UI
    logger.info("Integrating LLM with UI")
    integrate_llm_with_ui(components, config, llm_api)
    
    # Create and launch the Gradio interface
    logger.info("Creating Gradio interface")
    interface = create_layout(components, graph, config)
    
    logger.info(f"Launching Gradio server on {args.host}:{args.port}")
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=True
    )


if __name__ == "__main__":
    main()
