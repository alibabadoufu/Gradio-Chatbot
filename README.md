# 🦉 LangGraph OWL Multi-Agent System

A **LangGraph implementation** of the OWL (Optimized Workforce Learning) multi-agent system that provides powerful multi-agent collaboration capabilities with real-time streaming support.

## 🚀 Features

### Core Multi-Agent Architecture
- **User Agent**: Provides step-by-step task instructions
- **Assistant Agent**: Executes tasks using available tools
- **Dynamic Collaboration**: Agents collaborate until task completion
- **Real-time Streaming**: Watch agents work together in real-time

### Advanced Capabilities
- **Comprehensive Toolkit**: Search, code execution, data analysis, web scraping, and more
- **Streaming Interface**: Real-time updates via Gradio web interface
- **Flexible Configuration**: Multiple models, tool selections, and parameters
- **Async Support**: Full async/await support for high-performance applications
- **Memory & Checkpointing**: Persistent conversation state with LangGraph

### Tools & Integrations
- **Search Tools**: DuckDuckGo, Wikipedia, web scraping
- **Code Execution**: Python code execution with safety controls
- **File Operations**: Read, write, and manage files
- **Data Analysis**: Pandas, image analysis, data visualization
- **Math & Computation**: SymPy, NumPy, advanced calculations
- **Browser Automation**: Basic web automation capabilities

## 📦 Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- OpenAI API key (required)
- Git (for cloning the repository)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd langgraph-owl
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env file with your API keys
   ```

4. **Run the example**
   ```bash
   cd langgraph_owl
   python example.py
   ```

5. **Launch the web interface**
   ```bash
   python webapp.py
   ```

## � Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for additional search capabilities)
GOOGLE_API_KEY=your_google_api_key
SEARCH_ENGINE_ID=your_search_engine_id
TAVILY_API_KEY=your_tavily_api_key
```

### Model Configuration

The system supports various OpenAI models:
- `gpt-4o` (recommended for best performance)
- `gpt-4o-mini` (recommended for cost-effective usage)
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## 🎯 Usage Examples

### Basic Python Usage

```python
from langgraph_owl.core import create_owl_system
from langgraph_owl.tools import create_comprehensive_toolkit

# Create OWL system
owl_system = create_owl_system(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_rounds=10,
    tools=create_comprehensive_toolkit(),
    streaming=True
)

# Run a task
task = "Create a Python script that analyzes data from a CSV file"
answer, chat_history, token_usage = owl_system.run(task)
print(f"Answer: {answer}")
```

### Streaming Usage

```python
# Stream the execution
for chunk in owl_system.stream(task):
    print(f"Update: {chunk}")
```

### Async Usage

```python
import asyncio

async def main():
    answer, chat_history, token_usage = await owl_system.arun(task)
    print(f"Answer: {answer}")

asyncio.run(main())
```

### Web Interface Usage

Launch the web interface:
```bash
python langgraph_owl/webapp.py
```

Then open your browser to `http://localhost:7860`

## 🛠️ Architecture

### Multi-Agent Flow

```
User Input → User Agent → Assistant Agent → Tool Execution → Result
     ↑                                                            ↓
     ←──────────── Feedback Loop ←─────────────────────────────────
```

### LangGraph State Management

The system uses LangGraph's state management for:
- **Message History**: Conversation between agents
- **Task State**: Current progress and completion status
- **Tool Results**: Outputs from tool executions
- **Checkpointing**: Persistent state across sessions

### Agent Roles

**User Agent:**
- Analyzes the overall task
- Breaks down complex tasks into steps
- Provides specific instructions to Assistant Agent
- Monitors progress and provides feedback

**Assistant Agent:**
- Executes specific instructions
- Uses tools to gather information or perform actions
- Provides detailed solutions and explanations
- Reports back to User Agent with results

## 🧰 Available Tools

### Search & Information
- `SearchTool`: Web search via DuckDuckGo
- `WikipediaSearchTool`: Wikipedia search
- `WebScrapingTool`: Extract content from web pages

### Code & Development
- `CodeExecutionTool`: Execute Python code safely
- `MathCalculationTool`: Advanced mathematical calculations
- `FileWriteTool`: Write content to files

### Analysis & Data
- `DataAnalysisTool`: Analyze CSV, Excel, JSON files
- `ImageAnalysisTool`: Basic image analysis
- `BrowserAutomationTool`: Basic web automation

### Tool Collections
- `create_search_tools()`: Search and web tools
- `create_code_tools()`: Programming tools
- `create_analysis_tools()`: Data analysis tools
- `create_comprehensive_toolkit()`: All tools combined

## 🎨 Web Interface Features

### Real-time Streaming
- Live updates as agents collaborate
- Step-by-step progress tracking
- Real-time conversation log

### Configuration Options
- Model selection
- Tool selection
- Max rounds configuration
- Task input with examples

### User Experience
- Clean, modern interface
- Progress indicators
- Copy-to-clipboard functionality
- Example tasks for quick testing

## 📊 Comparison with Original OWL

| Feature | Original OWL | LangGraph OWL |
|---------|--------------|---------------|
| Multi-Agent Architecture | ✅ | ✅ |
| Real-time Streaming | ✅ | ✅ |
| Tool Integration | ✅ | ✅ |
| Web Interface | ✅ | ✅ |
| State Management | Custom | LangGraph |
| Async Support | ✅ | ✅ |
| Checkpointing | ❌ | ✅ |
| Graph Visualization | ❌ | ✅ (via LangGraph) |

## 🔍 Performance & Scaling

### Memory Management
- Efficient state management with LangGraph
- Automatic memory cleanup
- Configurable message history limits

### Streaming Performance
- Real-time updates without blocking
- Efficient chunk processing
- Minimal memory overhead

### Token Usage
- Automatic token counting
- Usage reporting
- Cost optimization tips

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. API Key Issues**
```bash
# Check your .env file
cat .env
# Ensure OPENAI_API_KEY is set correctly
```

**3. Tool Execution Errors**
```bash
# Ensure Python is in PATH for code execution
which python
```

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the same terms as the original OWL system.

## 🙏 Acknowledgments

- Original OWL system by CAMEL-AI
- LangGraph by LangChain
- Gradio for the web interface
- OpenAI for the language models

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with:
   - Error messages
   - Steps to reproduce
   - Environment details

---

**Happy Multi-Agent Collaboration! 🦉✨**

