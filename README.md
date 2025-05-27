# GenAI Chat Application

## Overview
This repository contains a sophisticated Gradio-based chat application that leverages an in-house Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) framework to provide users with an advanced document interaction experience.

The application features a modular design, a user-friendly interface with extensive customization options, and a robust backend powered by Langgraph for orchestrating complex AI workflows.

## Features
- Intuitive and powerful chat interface for document-related tasks
- Robust RAG system for retrieving information from SharePoint documents
- Granular user control through model selection, parameter tuning, and document scoping
- Modular architecture for scalability and maintainability
- Transparent AI operation with visible "thoughts" and source citations

## Project Structure
```
genai-chat-app/
│
├── app.py                  # Main Gradio application entry point
│
├── config/
│   └── config.yaml         # Central configuration file
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── llm_api.py      # Module for handling all interactions with the in-house LLM API
│
│   ├── core/
│   │   ├── __init__.py
│   │   └── langgraph_setup.py # Module for setting up and managing the Langgraph graph
│
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── components.py   # Reusable Gradio UI components (e.g., parameter sliders, dropdowns)
│   │   └── layout.py       # Module to define the overall UI layout and structure
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py      # Utility functions (e.g., loading config, data parsing)
│
├── requirements.txt        # Python package dependencies
│
└── README.md               # Project documentation
```

## Installation
1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the application by editing the `config/config.yaml` file

## Usage
Run the application with:
```
python app.py
```

The application will start a Gradio web server that can be accessed through your browser.

## Configuration
All configurable aspects of the application are managed through the `config/config.yaml` file, including:
- API endpoints
- Default model parameters
- UI messages
- Available models and document tags

## Development
The application follows a modular architecture to separate concerns and improve maintainability. Key components include:
- Gradio for the user interface
- Langgraph for orchestrating the AI workflow
- In-house LLM API integration
- RAG system for document retrieval and generation
