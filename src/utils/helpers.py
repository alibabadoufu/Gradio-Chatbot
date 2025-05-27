"""
Utility functions for the GenAI Chat Application.
"""

import os
import yaml
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the application configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        The configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def format_citation(document_title: str, sharepoint_base_url: str, chunk_text: str) -> Dict[str, str]:
    """
    Format a citation for a document chunk.
    
    Args:
        document_title: The title of the document.
        sharepoint_base_url: The base URL for SharePoint.
        chunk_text: The text of the document chunk.
        
    Returns:
        A dictionary containing the formatted citation.
    """
    # In a real implementation, this would construct the actual SharePoint URL
    # For now, we'll create a mock URL
    document_url = f"{sharepoint_base_url}{'documents/' + document_title.replace(' ', '%20')}"
    
    return {
        "title": document_title,
        "url": document_url,
        "chunk_text": chunk_text
    }


def parse_citations_from_thoughts(thoughts: Dict[str, Any], sharepoint_base_url: str) -> List[Dict[str, str]]:
    """
    Parse citations from the AI's thoughts.
    
    Args:
        thoughts: The AI's thoughts and reasoning steps.
        sharepoint_base_url: The base URL for SharePoint.
        
    Returns:
        A list of formatted citations.
    """
    citations = []
    
    if "retrieved_chunks" in thoughts:
        for chunk in thoughts["retrieved_chunks"]:
            citation = format_citation(
                document_title=chunk["document_title"],
                sharepoint_base_url=sharepoint_base_url,
                chunk_text=chunk["text"]
            )
            citations.append(citation)
    
    return citations


def get_document_tags_from_sharepoint() -> List[str]:
    """
    Get the list of available document tags from SharePoint.
    
    In a real implementation, this would query the SharePoint API.
    For now, we'll return a mock list.
    
    Returns:
        A list of document tags.
    """
    # This is a mock implementation
    return [
        "Financial Reports",
        "Project Proposals",
        "Technical Specifications",
        "Marketing Materials"
    ]


def log_user_feedback(feedback: Dict[str, Any]) -> None:
    """
    Log user feedback for later analysis.
    
    Args:
        feedback: The user's feedback, including rating and optional comments.
    """
    # In a real implementation, this would store the feedback in a database
    # For now, we'll just print it to the console
    print(f"User Feedback: {feedback}")
