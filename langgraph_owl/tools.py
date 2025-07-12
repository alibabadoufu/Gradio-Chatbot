"""
Tools module for the LangGraph OWL system.
This module provides various tools and toolkits that can be used by the multi-agent system.
"""

from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
import subprocess
import tempfile
import os
import requests
from bs4 import BeautifulSoup
import json
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Input for search tools"""
    query: str = Field(description="Search query")


class CodeExecutionInput(BaseModel):
    """Input for code execution tool"""
    code: str = Field(description="Python code to execute")
    language: str = Field(default="python", description="Programming language")


class FileWriteInput(BaseModel):
    """Input for file write tool"""
    filename: str = Field(description="Name of the file to write")
    content: str = Field(description="Content to write to the file")


class WebScrapingInput(BaseModel):
    """Input for web scraping tool"""
    url: str = Field(description="URL to scrape")


class SearchTool(BaseTool):
    """Search tool using DuckDuckGo"""
    name = "search"
    description = "Search the web using DuckDuckGo"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """Execute the search"""
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {str(e)}"


class WikipediaSearchTool(BaseTool):
    """Wikipedia search tool"""
    name = "wikipedia_search"
    description = "Search Wikipedia for information"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """Execute Wikipedia search"""
        try:
            import wikipedia
            # Set language to English
            wikipedia.set_lang("en")
            
            # Search for the query
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia results found for: {query}"
            
            # Get the first result
            page = wikipedia.page(search_results[0])
            
            # Return title and summary
            return f"**{page.title}**\n\n{page.summary}"
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation
            try:
                page = wikipedia.page(e.options[0])
                return f"**{page.title}**\n\n{page.summary}"
            except:
                return f"Multiple results found for '{query}'. Options: {', '.join(e.options[:5])}"
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return f"Wikipedia search failed: {str(e)}"


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code"""
    name = "code_execution"
    description = "Execute Python code and return the results"
    args_schema: Type[BaseModel] = CodeExecutionInput
    
    def _run(self, code: str, language: str = "python") -> str:
        """Execute code and return results"""
        if language.lower() != "python":
            return f"Language '{language}' is not supported. Only Python is supported."
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute the code
                result = subprocess.run(
                    ["python", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = []
                if result.stdout:
                    output.append(f"Output:\n{result.stdout}")
                if result.stderr:
                    output.append(f"Error:\n{result.stderr}")
                if result.returncode != 0:
                    output.append(f"Exit code: {result.returncode}")
                
                return "\n\n".join(output) if output else "Code executed successfully with no output."
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return "Code execution timed out (30 seconds limit)"
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return f"Code execution failed: {str(e)}"


class FileWriteTool(BaseTool):
    """Tool for writing files"""
    name = "file_write"
    description = "Write content to a file"
    args_schema: Type[BaseModel] = FileWriteInput
    
    def _run(self, filename: str, content: str) -> str:
        """Write content to file"""
        try:
            # Ensure the directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"Successfully wrote content to {filename}"
            
        except Exception as e:
            logger.error(f"File write error: {e}")
            return f"Failed to write file: {str(e)}"


class WebScrapingTool(BaseTool):
    """Tool for web scraping"""
    name = "web_scraping"
    description = "Scrape content from a webpage"
    args_schema: Type[BaseModel] = WebScrapingInput
    
    def _run(self, url: str) -> str:
        """Scrape webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            if len(text) > 5000:
                text = text[:5000] + "... (truncated)"
            
            return f"Content from {url}:\n\n{text}"
            
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return f"Failed to scrape {url}: {str(e)}"


class MathCalculationTool(BaseTool):
    """Tool for mathematical calculations"""
    name = "math_calculation"
    description = "Perform mathematical calculations using Python"
    args_schema: Type[BaseModel] = CodeExecutionInput
    
    def _run(self, code: str, language: str = "python") -> str:
        """Execute mathematical code"""
        # Add common math imports
        math_imports = """
import math
import numpy as np
import sympy as sp
from sympy import symbols, solve, integrate, diff, limit, series, simplify
"""
        
        full_code = math_imports + "\n" + code
        
        # Use the code execution tool
        code_tool = CodeExecutionTool()
        return code_tool._run(full_code, language)


class ImageAnalysisTool(BaseTool):
    """Tool for basic image analysis"""
    name = "image_analysis"
    description = "Analyze images and provide basic information"
    args_schema: Type[BaseModel] = FileWriteInput  # Reuse for file path
    
    def _run(self, filename: str, content: str = "") -> str:
        """Analyze image file"""
        try:
            if not os.path.exists(filename):
                return f"Image file not found: {filename}"
            
            # Open and analyze the image
            with Image.open(filename) as img:
                info = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height
                }
                
                # Convert to base64 for potential viewing
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                return f"""Image Analysis Results:
Format: {info['format']}
Mode: {info['mode']}
Size: {info['width']} x {info['height']} pixels
File: {filename}

Image has been analyzed successfully."""
                
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Failed to analyze image: {str(e)}"


class DataAnalysisTool(BaseTool):
    """Tool for data analysis using pandas"""
    name = "data_analysis"
    description = "Analyze data files (CSV, Excel, etc.)"
    args_schema: Type[BaseModel] = FileWriteInput  # Reuse for file path
    
    def _run(self, filename: str, content: str = "") -> str:
        """Analyze data file"""
        try:
            if not os.path.exists(filename):
                return f"Data file not found: {filename}"
            
            # Determine file type and read accordingly
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(filename)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filename)
            elif file_ext == '.json':
                df = pd.read_json(filename)
            else:
                return f"Unsupported file format: {file_ext}"
            
            # Basic analysis
            analysis = f"""Data Analysis Results for {filename}:

Shape: {df.shape}
Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

First 5 rows:
{df.head().to_string()}

Basic Statistics:
{df.describe().to_string()}

Missing Values:
{df.isnull().sum().to_string()}
"""
            
            return analysis
            
        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return f"Failed to analyze data: {str(e)}"


def create_default_tools() -> List[BaseTool]:
    """Create a default set of tools for the OWL system"""
    tools = [
        SearchTool(),
        WikipediaSearchTool(),
        CodeExecutionTool(),
        FileWriteTool(),
        WebScrapingTool(),
        MathCalculationTool(),
        ImageAnalysisTool(),
        DataAnalysisTool(),
    ]
    
    # Add Tavily search if available
    try:
        tavily_search = TavilySearchResults(max_results=5)
        tools.append(tavily_search)
    except Exception as e:
        logger.warning(f"Tavily search not available: {e}")
    
    return tools


def create_search_tools() -> List[BaseTool]:
    """Create search-specific tools"""
    return [
        SearchTool(),
        WikipediaSearchTool(),
        WebScrapingTool(),
    ]


def create_code_tools() -> List[BaseTool]:
    """Create code execution tools"""
    return [
        CodeExecutionTool(),
        MathCalculationTool(),
        FileWriteTool(),
    ]


def create_analysis_tools() -> List[BaseTool]:
    """Create data and image analysis tools"""
    return [
        ImageAnalysisTool(),
        DataAnalysisTool(),
    ]


class BrowserAutomationTool(BaseTool):
    """Basic browser automation tool using requests"""
    name = "browser_automation"
    description = "Perform basic browser automation tasks"
    args_schema: Type[BaseModel] = WebScrapingInput
    
    def _run(self, url: str) -> str:
        """Perform browser automation"""
        try:
            # This is a simplified version - full implementation would use Playwright
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the page
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract useful information
            title = soup.title.string if soup.title else "No title"
            
            # Get all links
            links = []
            for link in soup.find_all('a', href=True):
                links.append({
                    'text': link.get_text(strip=True),
                    'href': link['href']
                })
            
            # Get forms
            forms = []
            for form in soup.find_all('form'):
                forms.append({
                    'action': form.get('action', ''),
                    'method': form.get('method', 'GET')
                })
            
            result = f"""Browser Automation Results for {url}:

Title: {title}

Found {len(links)} links and {len(forms)} forms.

First 10 links:
"""
            
            for i, link in enumerate(links[:10]):
                result += f"{i+1}. {link['text'][:50]}... -> {link['href']}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Browser automation error: {e}")
            return f"Browser automation failed: {str(e)}"


def create_browser_tools() -> List[BaseTool]:
    """Create browser automation tools"""
    return [
        BrowserAutomationTool(),
        WebScrapingTool(),
    ]


def create_comprehensive_toolkit() -> List[BaseTool]:
    """Create a comprehensive toolkit with all available tools"""
    tools = []
    
    # Add all tool categories
    tools.extend(create_search_tools())
    tools.extend(create_code_tools())
    tools.extend(create_analysis_tools())
    tools.extend(create_browser_tools())
    
    # Remove duplicates
    unique_tools = []
    seen_names = set()
    
    for tool in tools:
        if tool.name not in seen_names:
            unique_tools.append(tool)
            seen_names.add(tool.name)
    
    return unique_tools


# Export commonly used tool collections
DEFAULT_TOOLS = create_default_tools()
SEARCH_TOOLS = create_search_tools()
CODE_TOOLS = create_code_tools()
ANALYSIS_TOOLS = create_analysis_tools()
BROWSER_TOOLS = create_browser_tools()
COMPREHENSIVE_TOOLS = create_comprehensive_toolkit()