# Document Chat RAG System

A Gradio-based chatbot application that allows users to upload documents and chat with them using an AI agent powered by LangGraph and OpenAI.

## Features

- **Document Upload**: Upload PDF documents for processing
- **Intelligent Processing**: Automatic chunking and embedding using OpenAI embeddings
- **Vector Storage**: Individual FAISS vectorstores for each document
- **Agentic RAG**: LangGraph-powered agent that intelligently selects relevant documents and response types
- **Interactive Chat**: Multi-tab Gradio interface for seamless user experience

## Architecture

The system follows the agentic RAG architecture shown in the reference diagram:

1. **Query Processing**: User input is analyzed to determine the best vectorstore
2. **Vector Database Selection**: Agent selects the most relevant document vectorstore
3. **Context Retrieval**: Relevant chunks are retrieved from the selected vectorstore
4. **Response Type Selection**: Agent determines whether to generate text, charts, or code
5. **Response Generation**: Final response is generated based on the selected type

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:7860`

3. Use the three tabs:
   - **Upload Documents**: Upload and process PDF files
   - **Document List**: View all uploaded documents
   - **Chat**: Select a document and start chatting

## Project Structure

```
gradio_chatbot_rag/
├── app.py                 # Main Gradio application
├── backend/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing and vectorstore creation
│   └── rag_agent.py          # LangGraph agentic RAG pipeline
├── vectorstores/         # FAISS vectorstores (created automatically)
├── documents/           # Uploaded documents storage
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Components

### Document Processor (`backend/document_processor.py`)
- Loads PDF documents using LangChain
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Creates embeddings using OpenAI embeddings
- Stores vectors in individual FAISS databases

### RAG Agent (`backend/rag_agent.py`)
- Implements LangGraph workflow for agentic RAG
- Selects appropriate vectorstore based on query
- Retrieves relevant context from selected documents
- Determines response type (text/chart/code)
- Generates appropriate responses

### Gradio Interface (`app.py`)
- Multi-tab interface for document management and chat
- Real-time document processing feedback
- Interactive chat with document selection
- Responsive design for desktop and mobile

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)

## Notes

- Currently supports PDF documents only
- Chart and code generation are placeholder features (can be extended)
- Vectorstores are stored locally using FAISS
- The system requires an active internet connection for OpenAI API calls

## Extending the System

The modular architecture allows for easy extensions:

1. **Add new document types**: Extend `document_processor.py` with new loaders
2. **Implement chart generation**: Add plotting logic to the RAG agent
3. **Add code generation**: Implement code generation capabilities
4. **Custom embeddings**: Replace OpenAI embeddings with local alternatives
5. **Database integration**: Replace FAISS with persistent vector databases

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **API errors**: Check your OpenAI API key in the `.env` file
3. **File upload issues**: Ensure the `documents/` and `vectorstores/` directories exist
4. **Memory issues**: Adjust chunk sizes in `document_processor.py` for large documents

