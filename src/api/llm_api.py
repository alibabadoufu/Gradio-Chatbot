"""
Module for handling all interactions with the in-house LLM API.
"""

import json
import requests
from typing import Dict, List, Optional, Generator, Any


class LLMApi:
    """Client for interacting with the in-house LLM API."""
    
    def __init__(self, api_url: str):
        """
        Initialize the LLM API client.
        
        Args:
            api_url: The URL of the in-house LLM API.
        """
        self.api_url = api_url
        self.session = requests.Session()
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        temperature: float = 0.7, 
        top_k: int = 50,
        document_tags: Optional[List[str]] = None,
        focus_mode: str = "DocChat",
        selected_documents: Optional[List[str]] = None,
        recency_bias: bool = False
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: The LLM model to use.
            temperature: Controls randomness in generation (0.0 to 1.0).
            top_k: Controls diversity of output.
            document_tags: Optional list of document tags to restrict RAG search.
            focus_mode: The operational mode of the chatbot (DocChat, DeepResearch, DocCompare).
            selected_documents: Optional list of document titles for DocCompare mode.
            recency_bias: Whether to prioritize recently updated documents.
            
        Returns:
            A generator yielding response tokens.
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "stream": True,
            "rag_config": {
                "document_tags": document_tags or [],
                "focus_mode": focus_mode,
                "selected_documents": selected_documents or [],
                "recency_bias": recency_bias
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        with self.session.post(
            self.api_url, 
            json=payload, 
            headers=headers, 
            stream=True
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    if line.startswith(b'data: '):
                        data = line[6:].decode('utf-8')
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
    
    def get_thoughts(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        document_tags: Optional[List[str]] = None,
        focus_mode: str = "DocChat"
    ) -> Dict[str, Any]:
        """
        Get the AI's intermediate thoughts or reasoning steps.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: The LLM model to use.
            document_tags: Optional list of document tags to restrict RAG search.
            focus_mode: The operational mode of the chatbot.
            
        Returns:
            A dictionary containing the AI's thoughts and retrieved document chunks.
        """
        payload = {
            "messages": messages,
            "model": model,
            "stream": False,
            "return_thoughts": True,
            "rag_config": {
                "document_tags": document_tags or [],
                "focus_mode": focus_mode
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = self.session.post(
            self.api_url, 
            json=payload, 
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_document_titles(self, document_tags: List[str]) -> List[str]:
        """
        Get the titles of documents corresponding to the selected tags.
        
        Args:
            document_tags: List of document tags to filter by.
            
        Returns:
            A list of document titles.
        """
        # In a real implementation, this would query the SharePoint API
        # For now, we'll return a mock list based on the tags
        mock_documents = {
            "Financial Reports": [
                "Q1 2025 Financial Report",
                "Annual Report 2024",
                "Budget Forecast 2025-2026"
            ],
            "Project Proposals": [
                "AI Integration Initiative",
                "Cloud Migration Strategy",
                "Mobile App Development Proposal"
            ],
            "Technical Specifications": [
                "System Architecture v2.0",
                "API Documentation",
                "Database Schema"
            ],
            "Marketing Materials": [
                "Brand Guidelines 2025",
                "Product Launch Campaign",
                "Market Analysis Report"
            ]
        }
        
        result = []
        for tag in document_tags:
            if tag in mock_documents:
                result.extend(mock_documents[tag])
        
        return result
