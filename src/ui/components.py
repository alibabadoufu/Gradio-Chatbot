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
