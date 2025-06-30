# Document Chat RAG System

A Gradio-based chatbot application that allows users to upload documents and chat with them using an AI agent powered by LangGraph and OpenAI, with S3 integration for document and vectorstore storage.

## Features

- **Document Upload**: Upload PDF documents for processing
- **Intelligent Processing**: Automatic chunking and embedding using OpenAI embeddings
- **S3 Storage**: Raw documents and individual FAISS vectorstores are stored in S3
- **Agentic RAG**: LangGraph-powered agent that intelligently selects relevant documents and response types
- **Interactive Chat**: Multi-tab Gradio interface for seamless user experience
- **Document List Search**: Search for documents by name in the document list
- **Document List Pagination**: Browse documents with pagination, showing 10 items per page

## Architecture

The system follows the agentic RAG architecture shown in the reference diagram:

1. **Query Processing**: User input is analyzed to determine the best vectorstore
2. **Vector Database Selection**: Agent selects the most relevant document vectorstore from S3
3. **Context Retrieval**: Relevant chunks are retrieved from the selected vectorstore (downloaded from S3 if not local)
4. **Response Type Selection**: Agent determines whether to generate text, charts, or code
5. **Response Generation**: Final response is generated based on the selected type

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

3. Use the three tabs:
   - **Upload Documents**: Upload and process PDF files (these will be stored in your S3 bucket)
   - **Document List**: View all uploaded documents (listed from your S3 bucket), with search and pagination controls
   - **Chat**: Select a document and start chatting

## Project Structure

```
gradio_chatbot_rag/
├── app.py                 # Main Gradio application
├── backend/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing and S3 integration for vectorstore creation
│   └── rag_agent.py          # LangGraph agentic RAG pipeline
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── README.md           # This file
└── test_sample.py      # Test script
```

## Components

### Document Processor (`backend/document_processor.py`)
- Loads PDF documents using LangChain
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Creates embeddings using OpenAI embeddings
- **Uploads raw documents to S3**
- **Saves and loads FAISS vectorstores to/from S3**

### RAG Agent (`backend/rag_agent.py`)
- Implements LangGraph workflow for agentic RAG
- Selects appropriate vectorstore based on query
- Retrieves relevant context from selected documents (which are loaded from S3)
- Determines response type (text/chart/code)
- Generates appropriate responses

### Gradio Interface (`app.py`)
- Multi-tab interface for document management and chat
- Real-time document processing feedback
- Interactive chat with document selection (documents listed from S3)
- Responsive design for desktop and mobile
- **Includes search and pagination for the document list**

## Requirements

- Python 3.8+
- OpenAI API key
- AWS S3 bucket and credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Required packages (see requirements.txt)

## Notes

- Currently supports PDF documents only
- Chart and code generation are placeholder features (can be extended)
- The system requires an active internet connection for OpenAI API calls and S3 operations

## Extending the System

The modular architecture allows for easy extensions:

1. **Add new document types**: Extend `document_processor.py` with new loaders
2. **Implement chart generation**: Add plotting logic to the RAG agent
3. **Add code generation**: Implement code generation capabilities
4. **Custom embeddings**: Replace OpenAI embeddings with local alternatives
5. **Database integration**: Replace FAISS with persistent vector databases (though S3 now handles persistence)

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **API errors**: Check your OpenAI API key in the `.env` file
3. **S3 errors**: Ensure your AWS credentials and S3_BUCKET_NAME are correctly configured in `.env` and that your AWS user has appropriate permissions for S3 read/write operations.
4. **Memory issues**: Adjust chunk sizes in `document_processor.py` for large documents

