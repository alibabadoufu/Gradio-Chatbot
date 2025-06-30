# Document Chat RAG System with Usage Monitoring

A Gradio-based chatbot application that allows users to upload documents and chat with them using an AI agent powered by LangGraph and OpenAI, with S3 integration for document and vectorstore storage, plus comprehensive usage monitoring and analytics.

## Features

- **Document Upload**: Upload PDF documents for processing
- **Intelligent Processing**: Automatic chunking and embedding using OpenAI embeddings
- **S3 Storage**: Raw documents and individual FAISS vectorstores are stored in S3
- **Agentic RAG**: LangGraph-powered agent that intelligently selects relevant documents and response types
- **Interactive Chat**: Multi-tab Gradio interface for seamless user experience
- **Document List Search**: Search for documents by name in the document list
- **Document List Pagination**: Browse documents with pagination, showing 10 items per page
- **📊 Usage Monitoring**: Comprehensive usage analytics and monitoring dashboard
- **📈 Trend Analysis**: Daily conversation trends with interactive charts
- **📋 Conversation Logs**: Detailed conversation history with filtering capabilities
- **🔍 Date Range Filtering**: Filter all monitoring data by custom date ranges

## Architecture

The system follows the agentic RAG architecture shown in the reference diagram:

1. **Query Processing**: User input is analyzed to determine the best vectorstore
2. **Vector Database Selection**: Agent selects the most relevant document vectorstore from S3
3. **Context Retrieval**: Relevant chunks are retrieved from the selected vectorstore (downloaded from S3 if not local)
4. **Response Type Selection**: Agent determines whether to generate text, charts, or code
5. **Response Generation**: Final response is generated based on the selected type
6. **Usage Logging**: All conversations are automatically logged to SQLite database for monitoring

## New Monitoring Features

### 📊 Monitoring Dashboard
The new "Monitoring" tab provides comprehensive usage analytics:

#### Key Metrics
- **Total Conversations**: Number of unique conversation sessions
- **Total Messages**: Total number of user messages processed
- **Unique Documents Chatted**: Number of different documents users have interacted with

#### Daily Trends
- Interactive line chart showing conversation volume over time
- Customizable date range selection
- Visual trend analysis for usage patterns

#### Conversation Details
- Complete conversation history in tabular format
- Columns: Timestamp, Conversation ID, User Message, AI Response, Selected Document
- Searchable and filterable data table
- Export capabilities for further analysis

#### Date Range Filtering
- Filter all monitoring data by custom start and end dates
- Default view shows last 30 days
- Real-time dashboard updates

### Database Schema
The monitoring system uses SQLite with the following schema:
```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    selected_document TEXT,
    conversation_id TEXT NOT NULL
);
```

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key and AWS S3 credentials:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:7860`

3. Use the four tabs:
   - **📄 Upload Documents**: Upload and process PDF files (stored in S3)
   - **📋 Document List**: View all uploaded documents with search and pagination
   - **💬 Chat**: Select a document and start chatting (conversations are automatically logged)
   - **📊 Monitoring**: View usage analytics, trends, and conversation history

## Project Structure

```
gradio_chatbot_rag/
├── app.py                     # Main Gradio application with monitoring
├── backend/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing and S3 integration
│   ├── rag_agent.py          # LangGraph agentic RAG pipeline
│   └── logger.py             # Usage logging and monitoring functions
├── requirements.txt           # Python dependencies (includes pandas, matplotlib)
├── .env.example              # Environment variables template
├── README.md                 # This file
├── test_sample.py            # Test script for core functionality
├── test_logging.py           # Test script for logging and monitoring
└── usage_logs.db             # SQLite database (created automatically)
```

## Components

### Document Processor (`backend/document_processor.py`)
- Loads PDF documents using LangChain
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Creates embeddings using OpenAI embeddings
- Uploads raw documents to S3
- Saves and loads FAISS vectorstores to/from S3

### RAG Agent (`backend/rag_agent.py`)
- Implements LangGraph workflow for agentic RAG
- Selects appropriate vectorstore based on query
- Retrieves relevant context from selected documents (loaded from S3)
- Determines response type (text/chart/code)
- Generates appropriate responses

### Usage Logger (`backend/logger.py`)
- **NEW**: Comprehensive logging system for usage monitoring
- SQLite database for storing conversation history
- Functions for retrieving usage data with date filtering
- Analytics functions for trends, metrics, and detailed reports
- Handles empty data gracefully

### Gradio Interface (`app.py`)
- Multi-tab interface for document management and chat
- Real-time document processing feedback
- Interactive chat with document selection
- **NEW**: Comprehensive monitoring dashboard with:
  - Date range selection controls
  - Interactive trend charts using matplotlib
  - Detailed conversation data tables
  - Key metrics display
- Responsive design for desktop and mobile

## Monitoring API

The monitoring system provides several key functions:

```python
# Log a conversation
log_conversation(user_message, ai_response, selected_document, conversation_id)

# Get usage data with optional date filtering
get_usage_data(start_date="2024-01-01", end_date="2024-12-31")

# Get daily conversation trends
get_daily_trends(start_date="2024-01-01", end_date="2024-12-31")

# Get detailed conversation dataframe
get_conversation_dataframe(start_date="2024-01-01", end_date="2024-12-31")

# Get key metrics
get_key_metrics(start_date="2024-01-01", end_date="2024-12-31")
```

## Requirements

- Python 3.8+
- OpenAI API key
- AWS S3 bucket and credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Required packages (see requirements.txt):
  - gradio, langchain, langchain-community, langchain-openai
  - faiss-cpu, pypdf, numpy, python-dotenv, langgraph, boto3
  - **NEW**: pandas, matplotlib (for monitoring and visualization)

## Testing

The application includes comprehensive test suites:

1. **Core functionality tests**:
   ```bash
   python test_sample.py
   ```

2. **Logging and monitoring tests**:
   ```bash
   python test_logging.py
   ```

## Notes

- Currently supports PDF documents only
- Chart and code generation are placeholder features (can be extended)
- The system requires an active internet connection for OpenAI API calls and S3 operations
- **NEW**: Usage data is stored locally in SQLite database (`usage_logs.db`)
- **NEW**: All conversations are automatically logged for monitoring purposes
- **NEW**: The monitoring dashboard provides real-time insights into usage patterns

## Privacy and Data

- Conversation logs are stored locally in SQLite database
- No conversation data is sent to external services beyond OpenAI for response generation
- Users can delete the `usage_logs.db` file to clear all monitoring data
- Date range filtering allows users to view specific time periods

## Extending the System

The modular architecture allows for easy extensions:

1. **Add new document types**: Extend `document_processor.py` with new loaders
2. **Implement chart generation**: Add plotting logic to the RAG agent
3. **Add code generation**: Implement code generation capabilities
4. **Custom embeddings**: Replace OpenAI embeddings with local alternatives
5. **Database integration**: Replace FAISS with persistent vector databases
6. **Advanced analytics**: Extend `logger.py` with more sophisticated analytics
7. **Export capabilities**: Add data export features to the monitoring dashboard
8. **User management**: Add user authentication and per-user analytics

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **API errors**: Check your OpenAI API key in the `.env` file
3. **S3 errors**: Ensure your AWS credentials and S3_BUCKET_NAME are correctly configured
4. **Database errors**: Check file permissions for SQLite database creation
5. **Monitoring issues**: Run `python test_logging.py` to verify logging functionality
6. **Memory issues**: Adjust chunk sizes in `document_processor.py` for large documents
7. **Plot display issues**: Ensure matplotlib backend is compatible with your system

