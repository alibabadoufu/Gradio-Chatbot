import os
import gradio as gr
import shutil
from datetime import datetime, timedelta
from backend.document_processor import process_document, s3_client, S3_BUCKET_NAME
from backend.rag_agent import RAGAgent
from backend.logger import log_conversation, init_db, get_daily_trends, get_conversation_dataframe, get_key_metrics

# Initialize the RAG agent
rag_agent = RAGAgent()

# Global variable to store uploaded documents (names of vectorstores)
uploaded_documents = []

# Pagination settings
ITEMS_PER_PAGE = 10
CONVERSATIONS_PER_PAGE = 20

def upload_document(file):
    """Process uploaded document and create vectorstore"""
    if file is None:
        return "No file uploaded", *get_document_list()
    
    try:
        # Process the document (this will also upload the raw file and vectorstore to S3)
        vectorstore_name = process_document(file.name)
        
        return f"Document '{vectorstore_name}' processed successfully!", *get_document_list()
    except Exception as e:
        return f"Error processing document: {str(e)}", *get_document_list()

def get_all_document_names_from_s3():
    """Helper function to get all document names from S3."""
    doc_names = set()
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="vectorstores/")
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
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
        if result and "chat_history" in result:
            response = result["chat_history"][-1].content if result["chat_history"] else "No response generated."
        else:
            response = "No response generated."
        
        # Log the conversation
        log_conversation(user_message=message, ai_response=response, selected_document=selected_document)

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

def create_daily_trends_data(start_date, end_date):
    """Create data for Gradio LinePlot showing daily conversation trends"""
    try:
        daily_trends = get_daily_trends(start_date, end_date)
        
        if daily_trends.empty:
            return None
        
        # Convert to format expected by Gradio LinePlot
        data = []
        for _, row in daily_trends.iterrows():
            data.append({
                "date": str(row['date']),
                "conversations": row['conversations']
            })
        
        return data
    except Exception as e:
        print(f"Error creating trends data: {str(e)}")
        return None

def get_conversation_list_paginated(search_query="", page=1, start_date=None, end_date=None):
    """Return paginated conversation data with search functionality"""
    try:
        # Get all conversation data
        # Handle optional parameters properly for the function signature
        start_param = start_date if start_date and start_date.strip() else None
        end_param = end_date if end_date and end_date.strip() else None
        conv_df = get_conversation_dataframe(start_param, end_param)
        
        if conv_df.empty:
            return [], "Page 0/0", 0, "No conversations found."
        
        # Apply search filter
        if search_query:
            mask = (
                conv_df['user_message'].str.contains(search_query, case=False, na=False) |
                conv_df['ai_response'].str.contains(search_query, case=False, na=False) |
                conv_df['selected_document'].str.contains(search_query, case=False, na=False)
            )
            filtered_df = conv_df[mask]
        else:
            filtered_df = conv_df
        
        # Sort by timestamp (newest first)
        filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        
        # Apply pagination
        total_conversations = len(filtered_df)
        total_pages = (total_conversations + CONVERSATIONS_PER_PAGE - 1) // CONVERSATIONS_PER_PAGE
        start_index = (page - 1) * CONVERSATIONS_PER_PAGE
        end_index = start_index + CONVERSATIONS_PER_PAGE
        paginated_df = filtered_df.iloc[start_index:end_index]
        
        # Convert to list format for display
        conversation_list = []
        for _, row in paginated_df.iterrows():
            conversation_list.append([
                row['timestamp'][:19],  # Trim milliseconds
                row['conversation_id'][:8] + "...",  # Shorten ID
                row['user_message'][:100] + "..." if len(row['user_message']) > 100 else row['user_message'],
                row['ai_response'][:100] + "..." if len(row['ai_response']) > 100 else row['ai_response'],
                row['selected_document'] or "N/A"
            ])
        
        page_info = f"Page {page}/{total_pages}" if total_pages > 0 else "Page 0/0"
        summary = f"Showing {len(conversation_list)} of {total_conversations} conversations"
        
        return conversation_list, page_info, total_pages, summary
    except Exception as e:
        return [], "Page 0/0", 0, f"Error loading conversations: {str(e)}"

def update_monitoring_dashboard(start_date, end_date):
    """Update the monitoring dashboard with filtered data"""
    try:
        # Get key metrics
        metrics = get_key_metrics(start_date, end_date)
        
        # Create daily trends data for LinePlot
        trends_data = create_daily_trends_data(start_date, end_date)
        
        # Format metrics display
        metrics_text = f"""
## Key Metrics ({start_date} to {end_date})

- **Total Conversations**: {metrics['total_conversations']}
- **Total Messages**: {metrics['total_messages']}
- **Unique Documents Chatted**: {metrics['unique_documents_chatted']}
        """
        
        return metrics_text, trends_data
    except Exception as e:
        error_text = f"Error updating dashboard: {str(e)}"
        return error_text, None

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

        # Monitoring Tab
        with gr.TabItem("📊 Monitoring"):
            gr.Markdown("## Usage Dashboard")
            
            with gr.Row():
                with gr.Column(scale=1):
                    start_date_input = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        placeholder="2024-01-01"
                    )
                with gr.Column(scale=1):
                    end_date_input = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.now().strftime("%Y-%m-%d"),
                        placeholder="2024-12-31"
                    )
                with gr.Column(scale=1):
                    update_dashboard_btn = gr.Button("Update Dashboard", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    metrics_display = gr.Markdown("## Key Metrics\nLoading...")
                with gr.Column(scale=2):
                    trends_plot = gr.LinePlot(
                        x="date",
                        y="conversations",
                        title="Daily Conversation Trends",
                        x_title="Date",
                        y_title="Number of Conversations",
                        width=600,
                        height=400
                    )

        # Conversation History Tab
        with gr.TabItem("💬 Conversation History"):
            gr.Markdown("## Conversation History & Search")
            
            with gr.Row():
                with gr.Column(scale=2):
                    conv_search_input = gr.Textbox(
                        label="Search Conversations",
                        placeholder="Search in messages, responses, or document names..."
                    )
                with gr.Column(scale=1):
                    conv_search_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=1):
                    conv_refresh_btn = gr.Button("Refresh")
            
            with gr.Row():
                with gr.Column(scale=1):
                    conv_start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        placeholder="2024-01-01"
                    )
                with gr.Column(scale=1):
                    conv_end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.now().strftime("%Y-%m-%d"),
                        placeholder="2024-12-31"
                    )
                with gr.Column(scale=1):
                    conv_filter_btn = gr.Button("Filter by Date", variant="secondary")
            
            conversation_summary = gr.Markdown("Loading conversation history...")
            
            conversation_data = gr.Dataframe(
                label="Conversation Details",
                headers=["Timestamp", "Conversation ID", "User Message", "AI Response", "Selected Document"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                height=400
            )
            
            with gr.Row():
                conv_prev_btn = gr.Button("Previous Page")
                conv_page_display = gr.Textbox(value="Page 1/1", interactive=False, scale=0)
                conv_next_btn = gr.Button("Next Page")

            # State variables for conversation pagination
            conv_current_page = gr.State(1)
            conv_total_pages = gr.State(1)
    
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

    # Monitoring dashboard event handlers
    update_dashboard_btn.click(
        fn=update_monitoring_dashboard,
        inputs=[start_date_input, end_date_input],
        outputs=[metrics_display, trends_plot]
    )

    # Load initial dashboard data
    demo.load(
        fn=lambda: update_monitoring_dashboard(
            (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        ),
        outputs=[metrics_display, trends_plot]
    )

    # Conversation history event handlers
    def update_conversation_display(search_query, page, start_date, end_date):
        conv_list, page_info, total_pages, summary = get_conversation_list_paginated(
            search_query, page, start_date, end_date
        )
        return conv_list, page_info, total_pages, summary

    def conv_search_handler(search_query, start_date, end_date):
        conv_list, page_info, total_pages, summary = get_conversation_list_paginated(
            search_query, 1, start_date, end_date
        )
        return conv_list, page_info, total_pages, summary, 1

    def conv_prev_page_handler(search_query, current_page, total_pages, start_date, end_date):
        new_page = max(1, current_page - 1)
        conv_list, page_info, total_pages_updated, summary = get_conversation_list_paginated(
            search_query, new_page, start_date, end_date
        )
        return conv_list, page_info, total_pages_updated, summary, new_page

    def conv_next_page_handler(search_query, current_page, total_pages, start_date, end_date):
        new_page = min(total_pages, current_page + 1)
        conv_list, page_info, total_pages_updated, summary = get_conversation_list_paginated(
            search_query, new_page, start_date, end_date
        )
        return conv_list, page_info, total_pages_updated, summary, new_page

    conv_search_btn.click(
        fn=conv_search_handler,
        inputs=[conv_search_input, conv_start_date, conv_end_date],
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

    conv_filter_btn.click(
        fn=conv_search_handler,
        inputs=[conv_search_input, conv_start_date, conv_end_date],
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

    conv_refresh_btn.click(
        fn=lambda: conv_search_handler("", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")),
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

    conv_prev_btn.click(
        fn=conv_prev_page_handler,
        inputs=[conv_search_input, conv_current_page, conv_total_pages, conv_start_date, conv_end_date],
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

    conv_next_btn.click(
        fn=conv_next_page_handler,
        inputs=[conv_search_input, conv_current_page, conv_total_pages, conv_start_date, conv_end_date],
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

    # Load initial conversation data
    demo.load(
        fn=lambda: conv_search_handler("", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")),
        outputs=[conversation_data, conv_page_display, conv_total_pages, conversation_summary, conv_current_page]
    )

if __name__ == "__main__":
    init_db() # Initialize the database
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

