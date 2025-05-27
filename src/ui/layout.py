"""
Module to define the overall UI layout and structure for the GenAI Chat Application.
"""

import gradio as gr
from typing import Dict, Any

from src.ui.components import setup_component_interactions


def create_layout(components: Dict[str, Any], graph: Any, config: Dict[str, Any]) -> gr.Blocks:
    """
    Create the overall UI layout for the GenAI Chat Application.
    
    Args:
        components: The UI components.
        graph: The Langgraph workflow.
        config: The application configuration.
        
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
        
        # Setup component interactions
        setup_component_interactions(components, config, graph)
        
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
