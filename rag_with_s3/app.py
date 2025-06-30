import os
import gradio as gr
import shutil
from backend.document_processor import process_document, s3_client, S3_BUCKET_NAME
from backend.rag_agent import RAGAgent

# Initialize the RAG agent
rag_agent = RAGAgent()

# Global variable to store uploaded documents (names of vectorstores)
uploaded_documents = []

# Pagination settings
ITEMS_PER_PAGE = 10

def upload_document(file):
    """Process uploaded document and create vectorstore"""
    if file is None:
        return "No file uploaded", *get_document_list()
    
    try:
        # Gradio provides a temporary file path, we need to copy it to a known location
        # or directly process it. For S3, we\'ll process it from the temporary path
        # and then upload the original to S3.
        
        # Process the document (this will also upload the raw file and vectorstore to S3)
        vectorstore_name = process_document(file.name)
        
        # Add to the list of uploaded documents
        # In a real application, this list would be persisted (e.g., in a database)
        # For now, we\'ll refresh it by listing from S3.
        
        return f"Document \'{vectorstore_name}\' processed successfully!", *get_document_list()
    except Exception as e:
        return f"Error processing document: {str(e)}", *get_document_list()

def get_all_document_names_from_s3():
    """Helper function to get all document names from S3."""
    doc_names = set()
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="vectorstores/")
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.startswith("vectorstores/") and "/index.faiss" in key:
                    # Extract document name from the S3 key (e.g., vectorstores/doc_name/index.faiss)
                    doc_name = key.split("/")[1]
                    doc_names.add(doc_name)
    except Exception as e:
        print(f"Error getting all document names from S3: {e}")
    return sorted(list(doc_names))

def get_document_list(search_query="", page=1):
    """Return list of uploaded documents from S3 with search and pagination"""
    all_docs = get_all_document_names_from_s3()
    
    # Filter by search query
    if search_query:
        filtered_docs = [doc for doc in all_docs if search_query.lower() in doc.lower()]
    else:
        filtered_docs = all_docs

    # Apply pagination
    total_pages = (len(filtered_docs) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_docs = filtered_docs[start_index:end_index]

    global uploaded_documents
    uploaded_documents = all_docs # Keep all docs for dropdown update

    if not paginated_docs:
        return "No documents found.", "Page 0/0", 0
    
    return "\n".join([f"• {doc}" for doc in paginated_docs]), f"Page {page}/{total_pages}", total_pages

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
    current_docs = get_all_document_names_from_s3()
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
            with gr.Row():
                search_input = gr.Textbox(label="Search Documents", placeholder="Enter document name...")
                search_btn = gr.Button("Search")
            document_list = gr.Textbox(
                label="Documents",
                value="", # Initial value will be set by get_document_list_initial
                interactive=False,
                lines=10
            )
            with gr.Row():
                prev_page_btn = gr.Button("Previous Page")
                page_number_display = gr.Textbox(value="Page 1/1", interactive=False, scale=0)
                next_page_btn = gr.Button("Next Page")
            refresh_btn = gr.Button("Refresh List")

            current_page = gr.State(1)
            total_pages = gr.State(1)

            def get_document_list_initial():
                docs, page_str, pages = get_document_list()
                return docs, page_str, pages
            
            demo.load(get_document_list_initial, outputs=[document_list, page_number_display, total_pages])

            def update_document_list_with_pagination(search_query, current_page_num):
                docs, page_str, pages = get_document_list(search_query, current_page_num)
                return docs, page_str, pages

            def go_to_prev_page(search_query, current_page_num, total_pages_num):
                new_page = max(1, current_page_num - 1)
                docs, page_str, pages = get_document_list(search_query, new_page)
                return new_page, docs, page_str, pages

            def go_to_next_page(search_query, current_page_num, total_pages_num):
                new_page = min(total_pages_num, current_page_num + 1)
                docs, page_str, pages = get_document_list(search_query, new_page)
                return new_page, docs, page_str, pages


        # Chat Tab
        with gr.TabItem("💬 Chat"):
            gr.Markdown("## Chat with Documents")
            
            with gr.Row():
                document_dropdown = gr.Dropdown(
                    label="Select Document",
                    choices=get_all_document_names_from_s3(), # Initial choices
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
        outputs=[upload_status, document_list, page_number_display, total_pages]
    )
    
    refresh_btn.click(
        fn=lambda: get_document_list(search_input.value, 1), # Reset to page 1 on refresh
        inputs=[],
        outputs=[document_list, page_number_display, total_pages]
    )

    search_btn.click(
        fn=lambda query: update_document_list_with_pagination(query, 1), # Reset to page 1 on search
        inputs=[search_input],
        outputs=[document_list, page_number_display, total_pages]
    )

    prev_page_btn.click(
        fn=go_to_prev_page,
        inputs=[search_input, current_page, total_pages],
        outputs=[current_page, document_list, page_number_display, total_pages]
    )

    next_page_btn.click(
        fn=go_to_next_page,
        inputs=[search_input, current_page, total_pages],
        outputs=[current_page, document_list, page_number_display, total_pages]
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
    # Initial call to populate the dropdown and list on startup
    # This is now handled by demo.load and update_document_dropdown initial value
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
