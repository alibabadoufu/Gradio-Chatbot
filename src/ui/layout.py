"""
Module to define the overall UI layout and structure for the GenAI Chat Application.
"""

import gradio as gr
from typing import Dict, Any

from src.ui.integration import on_chat_submit, on_regenerate, on_feedback_submit


def create_layout(components: Dict[str, Any], graph: Any, config: Dict[str, Any], llm_api: Any) -> gr.Blocks:
    """
    Create the overall UI layout for the GenAI Chat Application.
    
    Args:
        components: The UI components.
        graph: The Langgraph workflow.
        config: The application configuration.
        llm_api: The LLM API client.
        
    Returns:
        A Gradio Blocks interface.
    """
    with gr.Blocks(
        title="GenAI Chat Application",
        theme=gr.themes.Soft(),
        css="""
        #system-message {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin-bottom: 20px;
        }
        
        .citation-box {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            margin-top: 5px;
            font-size: 0.9em;
        }
        
        #thoughts-content {
            background-color: #fffbf0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        }
        """
    ) as interface:
        # Check if user has acknowledged terms
        acknowledged = gr.State(False)
        
        # Terms and Conditions Modal
        with gr.Group(visible=True) as terms_container:
            components["terms_modal"]
        
        # Main Application Container (initially hidden)
        with gr.Group(visible=False) as main_container:
            # Header
            with gr.Row():
                components["system_message"]
            
            # Middle Section (Two-Column Layout)
            with gr.Row():
                # Left Panel (Controls)
                with gr.Column(scale=1):
                    with gr.Box():
                        gr.Markdown("### Model Settings")
                        components["model_selector"]
                        
                        with gr.Group():
                            gr.Markdown("#### Generation Parameters")
                            components["temperature_slider"]
                            components["top_k_slider"]
                        
                        with gr.Group():
                            gr.Markdown("#### Document Selection")
                            components["document_tags"]
                        
                        with gr.Group():
                            gr.Markdown("#### Focus Modes")
                            components["focus_mode"]
                            components["document_selector"]
                        
                        with gr.Group():
                            components["recency_bias"]
                        
                        with gr.Group():
                            components["sharepoint_access"]
                
                # Right Panel (Chat Display)
                with gr.Column(scale=2):
                    components["chat_history"]
                    
                    with components["thoughts_display"]:
                        components["thoughts_content"]
            
            # Footer
            with gr.Row():
                # Chat Input
                with gr.Group():
                    components["chat_input"]
                    
                    with gr.Row():
                        components["submit_button"]
                        components["regenerate_button"]
                
                # User Feedback
                with components["feedback_container"]:
                    components["feedback_buttons"]
                    components["feedback_text"]
                    components["feedback_submit"]
        
        # Setup component interactions within the Blocks context
        
        # Focus mode change handler
        def on_focus_mode_change(focus_mode):
            return gr.update(visible=focus_mode == "DocCompare")
        
        components["focus_mode"].change(
            on_focus_mode_change,
            inputs=[components["focus_mode"]],
            outputs=[components["document_selector"]]
        )
        
        # Document tags change handler
        def on_document_tags_change(document_tags):
            # In a real implementation, this would fetch document titles from SharePoint
            # For now, we'll use mock data
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
            
            document_titles = []
            for tag in document_tags:
                if tag in mock_documents:
                    document_titles.extend(mock_documents[tag])
            
            return gr.update(choices=document_titles)
        
        components["document_tags"].change(
            on_document_tags_change,
            inputs=[components["document_tags"]],
            outputs=[components["document_selector"]]
        )
        
        # Connect the chat submission handler to the UI
        components["submit_button"].click(
            lambda *args: on_chat_submit(*args, llm_api=llm_api, config=config),
            inputs=[
                components["chat_input"],
                components["chat_history"],
                components["model_selector"],
                components["temperature_slider"],
                components["top_k_slider"],
                components["document_tags"],
                components["focus_mode"],
                components["document_selector"],
                components["recency_bias"]
            ],
            outputs=[
                components["chat_input"],
                components["chat_history"],
                components["thoughts_display"],
                components["thoughts_content"],
                components["feedback_container"]
            ]
        )
        
        # Connect the regenerate handler to the UI
        components["regenerate_button"].click(
            lambda *args: on_regenerate(*args, llm_api=llm_api, config=config),
            inputs=[
                components["chat_history"],
                components["model_selector"],
                components["temperature_slider"],
                components["top_k_slider"],
                components["document_tags"],
                components["focus_mode"],
                components["document_selector"],
                components["recency_bias"]
            ],
            outputs=[
                components["chat_history"],
                components["thoughts_display"],
                components["thoughts_content"],
                components["feedback_container"]
            ]
        )
        
        # Connect the feedback handler to the UI
        components["feedback_submit"].click(
            on_feedback_submit,
            inputs=[
                components["feedback_buttons"],
                components["feedback_text"],
                components["chat_history"]
            ],
            outputs=[
                components["feedback_buttons"],
                components["feedback_text"]
            ]
        )
        
        # Terms acceptance handler
        def on_terms_accept():
            return gr.update(visible=False), gr.update(visible=True), True
        
        accept_button = components["terms_modal"].children[0].children[1]
        accept_button.click(
            on_terms_accept,
            inputs=[],
            outputs=[terms_container, main_container, acknowledged]
        )
    
    return interface
