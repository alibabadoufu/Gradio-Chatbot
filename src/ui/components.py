"""
Reusable Gradio UI components for the GenAI Chat Application.
"""

import gradio as gr
from typing import Dict, Any, List, Callable, Tuple


def create_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create all the UI components for the GenAI Chat Application.
    
    Args:
        config: The application configuration.
        
    Returns:
        A dictionary containing all the UI components.
    """
    components = {}
    
    # Terms and Conditions Modal
    components["terms_modal"] = create_terms_modal(config["ui_messages"]["acknowledgement"])
    
    # Header Components
    components["system_message"] = gr.Markdown(
        config["ui_messages"]["system_message"],
        elem_id="system-message"
    )
    
    # Left Panel Components
    components["model_selector"] = gr.Dropdown(
        choices=config["models"],
        value=config["defaults"]["llm_model"],
        label="Model Selection",
        info="Select the AI model to use for chat"
    )
    
    components["temperature_slider"] = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=config["defaults"]["temperature"],
        step=0.1,
        label="Temperature",
        info="Controls randomness: lower values are more deterministic, higher values more creative"
    )
    
    components["top_k_slider"] = gr.Slider(
        minimum=1,
        maximum=100,
        value=config["defaults"]["top_k"],
        step=1,
        label="Top-K",
        info="Controls diversity: higher values consider more token options at each step"
    )
    
    components["document_tags"] = gr.Dropdown(
        choices=config["document_tags"],
        multiselect=True,
        label="Document Tags",
        info="Select document categories to search within"
    )
    
    components["focus_mode"] = gr.Radio(
        choices=["DocChat", "Deep Research", "DocCompare"],
        value=config["defaults"]["focus_mode"],
        label="Focus Mode",
        info="Select the operational mode of the AI"
    )
    
    components["document_selector"] = gr.Dropdown(
        multiselect=True,
        label="Select Documents to Compare",
        info="Choose specific documents to compare (only for DocCompare mode)",
        visible=False
    )
    
    components["recency_bias"] = gr.Checkbox(
        value=config["defaults"]["recency_bias"],
        label="Recency Bias",
        info="Prioritize recently updated documents"
    )
    
    # Create SharePoint links
    sharepoint_links = []
    for tag in config["document_tags"]:
        url = f"{config['api']['sharepoint_base_url']}{tag.replace(' ', '%20')}"
        sharepoint_links.append(f"[{tag}]({url})")
    
    components["sharepoint_access"] = gr.Markdown(
        "### SharePoint Access\n" + "\n".join(sharepoint_links),
        label="SharePoint Access"
    )
    
    # Right Panel Components
    components["chat_history"] = gr.Chatbot(
        [],
        elem_id="chat-history",
        height=500,
        avatar_images=("👤", "🤖")
    )
    
    components["thoughts_display"] = gr.Accordion(
        label="AI Thoughts",
        open=False,
        visible=False
    )
    
    components["thoughts_content"] = gr.Markdown(
        "",
        elem_id="thoughts-content"
    )
    
    # Footer Components
    components["chat_input"] = gr.Textbox(
        placeholder="Type your message here...",
        lines=3,
        label="Your Message"
    )
    
    components["submit_button"] = gr.Button("Submit", variant="primary")
    components["regenerate_button"] = gr.Button("Regenerate", variant="secondary")
    
    components["feedback_container"] = gr.Group(visible=False)
    components["feedback_buttons"] = gr.Radio(
        choices=["👍 Helpful", "👎 Not Helpful"],
        label="Was this response helpful?"
    )
    components["feedback_text"] = gr.Textbox(
        placeholder="Optional: Tell us why...",
        label="Additional Feedback"
    )
    components["feedback_submit"] = gr.Button("Submit Feedback")
    
    return components


def create_terms_modal(terms_text: str) -> gr.Blocks:
    """
    Create a modal for displaying terms and conditions.
    
    Args:
        terms_text: The terms and conditions text.
        
    Returns:
        A Gradio Blocks component for the modal.
    """
    with gr.Blocks() as modal:
        with gr.Box():
            gr.Markdown(terms_text)
            accept_button = gr.Button("I Accept", variant="primary")
    
    return modal


def setup_component_interactions(
    components: Dict[str, Any],
    config: Dict[str, Any],
    graph: Any
) -> None:
    """
    Set up the interactions between UI components.
    
    Args:
        components: The UI components.
        config: The application configuration.
        graph: The Langgraph workflow.
    """
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
    
    # Chat submission handler
    def on_chat_submit(
        user_message,
        chat_history,
        model,
        temperature,
        top_k,
        document_tags,
        focus_mode,
        selected_documents,
        recency_bias
    ):
        if not user_message.strip():
            return "", chat_history, gr.update(visible=False), ""
        
        # Add user message to chat history
        chat_history = chat_history + [[user_message, None]]
        
        # In a real implementation, this would run the Langgraph workflow
        # For now, we'll simulate a response
        from src.core.langgraph_setup import create_initial_state
        
        initial_state = create_initial_state(
            user_message=user_message,
            model=model,
            temperature=temperature,
            top_k=top_k,
            document_tags=document_tags,
            focus_mode=focus_mode,
            selected_documents=selected_documents,
            recency_bias=recency_bias,
            system_message=config["ui_messages"]["system_message"]
        )
        
        # Simulate thoughts
        thoughts = {
            "reasoning": "The user is asking about...",
            "retrieved_chunks": [
                {
                    "document_title": "Sample Document",
                    "text": "This is a sample document chunk that was retrieved."
                }
            ]
        }
        
        # Format thoughts for display
        thoughts_markdown = f"""
        ### AI Reasoning Process
        
        {thoughts['reasoning']}
        
        ### Retrieved Document Chunks
        
        **Document:** {thoughts['retrieved_chunks'][0]['document_title']}
        
        {thoughts['retrieved_chunks'][0]['text']}
        """
        
        # Simulate response
        response = "This is a simulated response from the AI. In a real implementation, this would be generated by the LLM."
        
        # Update chat history with AI response
        chat_history[-1][1] = response
        
        return "", chat_history, gr.update(visible=True), thoughts_markdown, gr.update(visible=True)
    
    components["submit_button"].click(
        on_chat_submit,
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
    
    # Feedback submission handler
    def on_feedback_submit(feedback_rating, feedback_text):
        from src.utils.helpers import log_user_feedback
        
        feedback = {
            "rating": feedback_rating,
            "comments": feedback_text
        }
        
        log_user_feedback(feedback)
        
        return gr.update(value=None), gr.update(value="")
    
    components["feedback_submit"].click(
        on_feedback_submit,
        inputs=[
            components["feedback_buttons"],
            components["feedback_text"]
        ],
        outputs=[
            components["feedback_buttons"],
            components["feedback_text"]
        ]
    )
