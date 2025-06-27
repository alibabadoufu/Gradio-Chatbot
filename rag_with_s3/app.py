import os
import gradio as gr
import shutil
from backend.document_processor import process_document, s3_client, S3_BUCKET_NAME
from backend.rag_agent import RAGAgent

# Initialize the RAG agent
rag_agent = RAGAgent()

# Global variable to store uploaded documents (names of vectorstores)
uploaded_documents = []

def upload_document(file):
    """Process uploaded document and create vectorstore"""
    if file is None:
        return "No file uploaded", get_document_list()
    
    try:
        # Gradio provides a temporary file path, we need to copy it to a known location
        # or directly process it. For S3, we'll process it from the temporary path
        # and then upload the original to S3.
        
        # Process the document (this will also upload the raw file and vectorstore to S3)
        vectorstore_name = process_document(file.name)
        
        # Add to the list of uploaded documents
        # In a real application, this list would be persisted (e.g., in a database)
        # For now, we'll refresh it by listing from S3.
        
        return f"Document \'{vectorstore_name}\' processed successfully!", get_document_list()
    except Exception as e:
        return f"Error processing document: {str(e)}", get_document_list()

def get_document_list():
    """Return list of uploaded documents from S3"""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="vectorstores/")
        if 'Contents' not in response:
            return "No documents uploaded yet."
        
        doc_names = set()
        for obj in response['Contents']:
            key = obj['Key']
            if key.startswith("vectorstores/") and "/index.faiss" in key:
                # Extract document name from the S3 key (e.g., vectorstores/doc_name/index.faiss)
                doc_name = key.split("/")[1]
                doc_names.add(doc_name)
        
        global uploaded_documents
        uploaded_documents = sorted(list(doc_names))

        if not uploaded_documents:
            return "No documents uploaded yet."
        
        return "\n".join([f"• {doc}" for doc in uploaded_documents])
    except Exception as e:
        return f"Error listing documents from S3: {str(e)}"

def chat_with_document(message, history, selected_document):
    """Chat with selected document using RAG agent"""
    if not message:
        return history, ""
    
    if not selected_document or selected_document == "Select a document":
        history.append([message, "Please select a document first."])
        return history, ""
    
    try:
        # Use the RAG agent to get response
        result = rag_agent.invoke_agent(
            question=message,
            selected_vectorstore=selected_document,
            available_vectorstores=uploaded_documents
        )
        
        # Extract the response from the result
        if result and 'chat_history' in result:
            response = result['chat_history'][-1].content if result['chat_history'] else "No response generated."
        else:
            response = "No response generated."
        
        history.append([message, response])
        return history, ""
    except Exception as e:
        history.append([message, f"Error: {str(e)}"])
        return history, ""

def update_document_dropdown():
    """Update the dropdown with available documents"""
    current_docs = []
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="vectorstores/")
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.startswith("vectorstores/") and "/index.faiss" in key:
                    doc_name = key.split("/")[1]
                    current_docs.append(doc_name)
        current_docs = sorted(list(set(current_docs)))
    except Exception as e:
        print(f"Error updating dropdown: {e}")
        current_docs = []

    if not current_docs:
        return gr.Dropdown(choices=["Select a document"], value="Select a document", interactive=True)
    return gr.Dropdown(choices=current_docs, value=current_docs[0], interactive=True)

# Create the Gradio interface
with gr.Blocks(title="Document Chat RAG System") as demo:
    gr.Markdown("# Document Chat RAG System")
    gr.Markdown("Upload documents and chat with them using an AI agent powered by LangGraph and S3.")
    
    with gr.Tabs():
        # Document Upload Tab
        with gr.TabItem("📄 Upload Documents"):
            gr.Markdown("## Upload and Process Documents")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload PDF Document",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("Process Document", variant="primary")
                
                with gr.Column():
                    upload_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3
                    )
        
        # Document List Tab
        with gr.TabItem("📋 Document List"):
            gr.Markdown("## Uploaded Documents")
            document_list = gr.Textbox(
                label="Documents",
                value=get_document_list(),
                interactive=False,
                lines=10
            )
            refresh_btn = gr.Button("Refresh List")
        
        # Chat Tab
        with gr.TabItem("💬 Chat"):
            gr.Markdown("## Chat with Documents")
            
            with gr.Row():
                document_dropdown = gr.Dropdown(
                    label="Select Document",
                    choices=["Select a document"] if not uploaded_documents else uploaded_documents,
                    value="Select a document",
                    interactive=True
                )
                refresh_dropdown_btn = gr.Button("Refresh Documents")
            
            chatbot = gr.Chatbot(
                label="Chat History",
                height=400
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask a question about the selected document...",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Event handlers
    upload_btn.click(
        fn=upload_document,
        inputs=[file_input],
        outputs=[upload_status, document_list]
    )
    
    refresh_btn.click(
        fn=get_document_list,
        outputs=[document_list]
    )
    
    refresh_dropdown_btn.click(
        fn=update_document_dropdown,
        outputs=[document_dropdown]
    )
    
    send_btn.click(
        fn=chat_with_document,
        inputs=[msg_input, chatbot, document_dropdown],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=chat_with_document,
        inputs=[msg_input, chatbot, document_dropdown],
        outputs=[chatbot, msg_input]
    )

if __name__ == "__main__":
    # These local directories are no longer strictly needed for S3 storage
    # but can be kept for local temporary file handling if desired.
    # os.makedirs("gradio_chatbot_rag/vectorstores", exist_ok=True)
    # os.makedirs("gradio_chatbot_rag/documents", exist_ok=True)
    
    # Initial call to populate the dropdown and list on startup
    initial_docs = []
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="vectorstores/")
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.startswith("vectorstores/") and "/index.faiss" in key:
                    doc_name = key.split("/")[1]
                    initial_docs.append(doc_name)
        global uploaded_documents
        uploaded_documents = sorted(list(set(initial_docs)))
    except Exception as e:
        print(f"Error during initial document list population: {e}")

    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


