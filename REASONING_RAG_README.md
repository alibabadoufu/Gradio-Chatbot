# Reasoning-Augmented Retrieval Agent with LangGraph

A sophisticated RAG (Retrieval-Augmented Generation) agent that uses reasoning to improve retrieval quality and generate more accurate responses. This implementation uses LangGraph to create a stateful, multi-step workflow that decomposes queries, performs intelligent retrieval, evaluates context sufficiency, and generates comprehensive answers.

## 🏗️ Architecture

The agent implements a four-node graph structure:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Reasoning  │───►│  Retrieval  │───►│ Evaluation  │───►│  Response   │
│    Node     │    │    Node     │    │    Node     │    │    Node     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                           │                        ▲
                                           │                        │
                                           └────────────────────────┘
                                           (if context insufficient)
```

### Node Functions:

1. **Reasoning Node**: Decomposes complex queries, rewrites them for better retrieval, and creates an execution plan
2. **Retrieval Node**: Executes RAG queries using the processed query and sub-questions
3. **Evaluation Node**: Uses LLM to self-evaluate retrieved context sufficiency
4. **Response Node**: Generates final user-facing answers based on validated context

## 📋 Features

- **Query Decomposition**: Breaks complex queries into manageable sub-questions
- **Intelligent Rewriting**: Optimizes queries for better retrieval performance
- **Self-Evaluation**: Automatically assesses context quality and completeness
- **Retry Logic**: Re-attempts retrieval if initial context is insufficient
- **State Management**: Maintains context across multiple retrieval attempts
- **Comprehensive Logging**: Detailed execution tracking and debugging information

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### Basic Usage

```python
from reasoning_augmented_retrieval import ReasoningAugmentedRAG

# Initialize the agent
agent = ReasoningAugmentedRAG(model_name="gpt-3.5-turbo", max_retries=2)

# Setup knowledge base
documents = [
    "Your document content here...",
    "More documents...",
]
agent.setup_vectorstore(documents)

# Run a query
result = agent.run("What is machine learning?")
print(result["final_answer"])
```

## 📚 Dependencies

The implementation requires the following packages:

```
langchain==0.1.0
langchain-core==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
langchain-text-splitters==0.0.1
langgraph==0.0.15
openai==1.6.1
chromadb==0.4.18
tiktoken==0.5.2
python-dotenv==1.0.0
```

## 🔧 Configuration

### Agent Parameters

- `model_name`: LLM model to use (default: "gpt-3.5-turbo")
- `max_retries`: Maximum retrieval attempts (default: 2)

### Vector Store Configuration

- `chunk_size`: Document chunk size (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `k`: Number of documents to retrieve (default: 3)

## 📝 Examples

### Interactive Demo

Run the interactive demo to test the agent:

```bash
python rag_agent_example.py
```

Choose option 1 for interactive mode where you can ask your own questions.

### Batch Processing

```python
# Example batch processing
test_queries = [
    "What is Python and how is it used?",
    "Explain the difference between machine learning and deep learning",
    "How does RAG work and why is it useful?",
]

for query in test_queries:
    result = agent.run(query)
    print(f"Query: {query}")
    print(f"Answer: {result['final_answer']}")
    print("-" * 40)
```

### Custom Document Loading

```python
# Load documents from files
from langchain_community.document_loaders import TextLoader

loader = TextLoader("your_document.txt")
documents = loader.load()

# Extract text content
doc_texts = [doc.page_content for doc in documents]
agent.setup_vectorstore(doc_texts)
```

## 🔍 Understanding the Output

The agent returns a comprehensive result dictionary:

```python
{
    "query": "Original user query",
    "final_answer": "Generated response",
    "plan": "Execution plan created by reasoning node",
    "rewritten_query": "Optimized query for retrieval",
    "decomposed_queries": ["Sub-question 1", "Sub-question 2"],
    "retrieved_context": ["Context piece 1", "Context piece 2"],
    "evaluation_result": {
        "sufficient": True/False,
        "reason": "Evaluation explanation",
        "confidence": 85
    },
    "retry_count": 0
}
```

## 🎯 Key Components

### State Management

The agent uses a TypedDict state structure:

```python
class AgentState(TypedDict):
    original_query: str
    decomposed_queries: List[str]
    rewritten_query: str
    plan: str
    retrieved_context: List[str]
    evaluation_result: Dict[str, Any]
    is_sufficient: bool
    retry_count: int
    final_answer: str
    messages: List[Any]
```

### Conditional Logic

The evaluation node uses conditional edges to determine next steps:

- **Context Sufficient**: Proceed to response generation
- **Context Insufficient + Retries Available**: Loop back to retrieval
- **Max Retries Reached**: Proceed to response with available context

## 🔬 Advanced Usage

### Custom Evaluation Criteria

Modify the evaluation prompt to customize assessment criteria:

```python
# In evaluation_node method
evaluation_prompt = ChatPromptTemplate.from_template("""
    Custom evaluation criteria here...
    Consider domain-specific requirements...
""")
```

### Custom Retrieval Strategy

Extend the retrieval node for specialized retrieval:

```python
def custom_retrieval_node(self, state: AgentState) -> AgentState:
    # Custom retrieval logic
    # Multi-source retrieval
    # Filtering and ranking
    return state
```

### Multiple Model Support

Use different models for different nodes:

```python
reasoning_llm = ChatOpenAI(model="gpt-4", temperature=0.1)
evaluation_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
response_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
```

## 📊 Performance Considerations

### Optimization Tips

1. **Chunk Size**: Adjust based on document type and query complexity
2. **Retrieval K**: Balance between context richness and processing time
3. **Max Retries**: Set based on quality requirements vs. response time
4. **Model Selection**: Use appropriate models for each node's complexity

### Monitoring

The agent includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Track execution time, retrieval quality, and evaluation decisions.

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **API Key Issues**: Verify OpenAI API key is correctly set
3. **Memory Issues**: Reduce chunk size or retrieval count for large documents
4. **Slow Performance**: Consider using faster models or reducing max_retries

### Error Handling

The agent includes robust error handling:

```python
try:
    result = agent.run(query)
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

## 📈 Future Enhancements

Potential improvements and extensions:

- **Multi-Modal Support**: Add image and audio processing
- **Streaming Responses**: Implement real-time response streaming
- **Caching**: Add intelligent caching for repeated queries
- **Metrics**: Implement comprehensive performance metrics
- **A/B Testing**: Framework for testing different reasoning strategies

## 🤝 Contributing

To contribute to this implementation:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add comprehensive tests
5. Submit a pull request

## 📄 License

This implementation is provided as-is for educational and research purposes. Please review the licenses of all dependencies before commercial use.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the examples and documentation
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key configuration

---

**Note**: This implementation demonstrates the core concepts of reasoning-augmented retrieval. For production use, consider additional features like persistent storage, authentication, rate limiting, and comprehensive error handling.