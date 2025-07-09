"""
Reasoning-Augmented Retrieval Agent using LangGraph

This script implements a RAG agent that uses reasoning to improve retrieval quality.
The agent decomposes queries, performs retrieval, evaluates the context, and generates responses.

Graph Flow:
1. reasoning_node: Decomposes and rewrites the query for better retrieval
2. retrieval_node: Executes RAG query using the processed query
3. evaluation_node: Evaluates if retrieved context is sufficient
4. response_node: Generates final answer OR loops back to retrieval_node
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state structure for the agent
class AgentState(TypedDict):
    """State structure for the Reasoning-Augmented Retrieval agent"""
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

class ReasoningAugmentedRAG:
    """Reasoning-Augmented Retrieval Agent using LangGraph"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_retries: int = 2):
        """
        Initialize the RAG agent with reasoning capabilities
        
        Args:
            model_name: The LLM model to use
            max_retries: Maximum number of retrieval retries
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.max_retries = max_retries
        self.vectorstore = None
        self.retriever = None
        
        # Initialize the state graph
        self.app = self._build_graph()
    
    def setup_vectorstore(self, documents: List[str] = None):
        """
        Set up the vector store for retrieval
        
        Args:
            documents: List of document texts to index
        """
        if documents is None:
            # Default sample documents for demonstration
            documents = [
                "Python is a high-level programming language known for its simplicity and readability.",
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
                "Retrieval-Augmented Generation (RAG) combines retrieval with language generation.",
                "Vector databases are optimized for storing and querying high-dimensional vectors.",
            ]
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # Split documents into chunks
        chunks = []
        for doc in documents:
            chunks.extend(text_splitter.split_text(doc))
        
        # Create vector store
        self.vectorstore = Chroma.from_texts(chunks, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        logger.info(f"Vector store initialized with {len(chunks)} chunks")
    
    def reasoning_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Reasoning - Decomposes query, rewrites it, and creates a plan
        
        This node takes the original query and uses the LLM to:
        1. Decompose complex queries into sub-questions
        2. Rewrite the query for better retrieval
        3. Create a retrieval plan
        """
        logger.info("Executing reasoning_node")
        
        reasoning_prompt = ChatPromptTemplate.from_template("""
        You are a query analysis expert. Given a user query, perform the following tasks:
        
        1. DECOMPOSE: Break down the query into 2-3 sub-questions if it's complex
        2. REWRITE: Rewrite the query to be more specific and retrieval-friendly
        3. PLAN: Create a brief plan for how to approach answering this query
        
        Original Query: {query}
        
        Provide your response in the following format:
        DECOMPOSITION:
        - [sub-question 1]
        - [sub-question 2]
        - [sub-question 3]
        
        REWRITTEN_QUERY:
        [rewritten query optimized for retrieval]
        
        PLAN:
        [brief plan for answering the query]
        """)
        
        chain = reasoning_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["original_query"]})
        
        # Parse the result
        lines = result.strip().split('\n')
        decomposed_queries = []
        rewritten_query = ""
        plan = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("DECOMPOSITION:"):
                current_section = "decomposition"
            elif line.startswith("REWRITTEN_QUERY:"):
                current_section = "rewritten"
            elif line.startswith("PLAN:"):
                current_section = "plan"
            elif line.startswith("- ") and current_section == "decomposition":
                decomposed_queries.append(line[2:])
            elif current_section == "rewritten" and line:
                rewritten_query = line
            elif current_section == "plan" and line:
                plan = line
        
        # Update state
        state["decomposed_queries"] = decomposed_queries
        state["rewritten_query"] = rewritten_query or state["original_query"]
        state["plan"] = plan
        
        logger.info(f"Reasoning complete. Rewritten query: {state['rewritten_query']}")
        return state
    
    def retrieval_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Retrieval - Executes RAG query using the processed query
        
        This node performs retrieval using the rewritten query from the reasoning node
        """
        logger.info("Executing retrieval_node")
        
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call setup_vectorstore() first.")
        
        # Retrieve documents using the rewritten query
        retrieved_docs = self.retriever.invoke(state["rewritten_query"])
        
        # Extract content from retrieved documents
        retrieved_context = [doc.page_content for doc in retrieved_docs]
        
        # Also try to retrieve for decomposed queries if they exist
        for sub_query in state["decomposed_queries"]:
            sub_docs = self.retriever.invoke(sub_query)
            retrieved_context.extend([doc.page_content for doc in sub_docs])
        
        # Remove duplicates while preserving order
        unique_context = []
        seen = set()
        for context in retrieved_context:
            if context not in seen:
                unique_context.append(context)
                seen.add(context)
        
        state["retrieved_context"] = unique_context
        
        logger.info(f"Retrieved {len(unique_context)} unique context pieces")
        return state
    
    def evaluation_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Evaluation - Evaluates if retrieved context is sufficient
        
        This conditional node uses the LLM to assess whether the retrieved context
        is sufficient to answer the original query
        """
        logger.info("Executing evaluation_node")
        
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are an evaluation expert. Assess whether the retrieved context is sufficient to answer the user's query.
        
        Original Query: {query}
        Rewritten Query: {rewritten_query}
        Plan: {plan}
        
        Retrieved Context:
        {context}
        
        Evaluate the context and provide:
        1. SUFFICIENT: Yes/No - Is the context sufficient to answer the query?
        2. REASON: Explain why it is or isn't sufficient
        3. MISSING: What information is missing (if any)?
        4. CONFIDENCE: Score from 0-100 on confidence level
        
        Format your response as:
        SUFFICIENT: [Yes/No]
        REASON: [explanation]
        MISSING: [missing information or "None"]
        CONFIDENCE: [0-100]
        """)
        
        context_text = "\n".join(state["retrieved_context"])
        
        chain = evaluation_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": state["original_query"],
            "rewritten_query": state["rewritten_query"],
            "plan": state["plan"],
            "context": context_text
        })
        
        # Parse evaluation result
        lines = result.strip().split('\n')
        evaluation_result = {}
        
        for line in lines:
            if line.startswith("SUFFICIENT:"):
                evaluation_result["sufficient"] = line.split(": ", 1)[1].strip().lower() == "yes"
            elif line.startswith("REASON:"):
                evaluation_result["reason"] = line.split(": ", 1)[1].strip()
            elif line.startswith("MISSING:"):
                evaluation_result["missing"] = line.split(": ", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    evaluation_result["confidence"] = int(line.split(": ", 1)[1].strip())
                except ValueError:
                    evaluation_result["confidence"] = 0
        
        state["evaluation_result"] = evaluation_result
        state["is_sufficient"] = evaluation_result.get("sufficient", False)
        
        logger.info(f"Evaluation complete. Sufficient: {state['is_sufficient']}")
        return state
    
    def response_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Response - Generates the final user-facing answer
        
        This node creates the final response based on the validated context
        """
        logger.info("Executing response_node")
        
        response_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Generate a comprehensive answer to the user's query based on the provided context.
        
        Original Query: {query}
        Plan: {plan}
        
        Context:
        {context}
        
        Evaluation Result: {evaluation}
        
        Instructions:
        1. Provide a direct answer to the user's query
        2. Use the context to support your answer
        3. Be clear and concise
        4. If the context is insufficient, acknowledge limitations
        5. Cite relevant information from the context
        
        Answer:
        """)
        
        context_text = "\n".join(state["retrieved_context"])
        evaluation_summary = f"Sufficient: {state['evaluation_result'].get('sufficient', False)}, Confidence: {state['evaluation_result'].get('confidence', 0)}"
        
        chain = response_prompt | self.llm | StrOutputParser()
        final_answer = chain.invoke({
            "query": state["original_query"],
            "plan": state["plan"],
            "context": context_text,
            "evaluation": evaluation_summary
        })
        
        state["final_answer"] = final_answer
        
        logger.info("Response generation complete")
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """
        Conditional function to determine the next step after evaluation
        
        Returns:
            "response" if context is sufficient or max retries reached
            "retrieval" if context is insufficient and retries remain
        """
        if state["is_sufficient"]:
            return "response"
        
        if state["retry_count"] >= self.max_retries:
            logger.info(f"Max retries ({self.max_retries}) reached. Proceeding to response.")
            return "response"
        
        logger.info(f"Context insufficient. Retrying retrieval. Attempt {state['retry_count'] + 1}")
        state["retry_count"] += 1
        
        # Modify the query for re-retrieval
        state["rewritten_query"] = f"More specific: {state['rewritten_query']}"
        
        return "retrieval"
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with all nodes and edges
        
        Returns:
            Compiled StateGraph application
        """
        logger.info("Building LangGraph StateGraph")
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("reasoning", self.reasoning_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("evaluation", self.evaluation_node)
        workflow.add_node("response", self.response_node)
        
        # Set entry point
        workflow.set_entry_point("reasoning")
        
        # Add edges
        workflow.add_edge("reasoning", "retrieval")
        workflow.add_edge("retrieval", "evaluation")
        
        # Add conditional edge from evaluation
        workflow.add_conditional_edges(
            "evaluation",
            self.should_continue,
            {
                "response": "response",
                "retrieval": "retrieval"
            }
        )
        
        # Add finish edge
        workflow.add_edge("response", END)
        
        # Compile the graph
        app = workflow.compile()
        
        logger.info("StateGraph built successfully")
        return app
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the reasoning-augmented retrieval agent
        
        Args:
            query: User's query
            
        Returns:
            Dictionary containing the final result and intermediate states
        """
        logger.info(f"Running RAG agent with query: {query}")
        
        # Initialize state
        initial_state = {
            "original_query": query,
            "decomposed_queries": [],
            "rewritten_query": "",
            "plan": "",
            "retrieved_context": [],
            "evaluation_result": {},
            "is_sufficient": False,
            "retry_count": 0,
            "final_answer": "",
            "messages": []
        }
        
        # Execute the graph
        final_state = self.app.invoke(initial_state)
        
        logger.info("RAG agent execution complete")
        
        return {
            "query": query,
            "final_answer": final_state["final_answer"],
            "plan": final_state["plan"],
            "rewritten_query": final_state["rewritten_query"],
            "decomposed_queries": final_state["decomposed_queries"],
            "retrieved_context": final_state["retrieved_context"],
            "evaluation_result": final_state["evaluation_result"],
            "retry_count": final_state["retry_count"]
        }

# Example usage and demonstration
def main():
    """
    Demonstration of the Reasoning-Augmented Retrieval agent
    """
    print("🤖 Initializing Reasoning-Augmented Retrieval Agent...")
    
    # Initialize the agent
    agent = ReasoningAugmentedRAG(model_name="gpt-3.5-turbo", max_retries=2)
    
    # Setup vector store with sample documents
    sample_docs = [
        "Python is a versatile programming language widely used for web development, data science, artificial intelligence, and automation. It was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
        "LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs). It extends LangChain with graph-based workflows and state management.",
        "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. It retrieves relevant documents and uses them to generate more accurate and informative responses.",
        "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for similarity search and recommendation systems.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves tasks like text analysis, sentiment analysis, and language translation.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and speech recognition."
    ]
    
    agent.setup_vectorstore(sample_docs)
    
    # Test queries
    test_queries = [
        "What is Python and how is it used?",
        "Explain the relationship between machine learning and artificial intelligence",
        "How does RAG work and what are its benefits?",
        "What is the difference between machine learning and deep learning?"
    ]
    
    print("\n🔍 Testing Reasoning-Augmented Retrieval Agent:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = agent.run(query)
            
            print(f"🧠 Plan: {result['plan']}")
            print(f"🔄 Rewritten Query: {result['rewritten_query']}")
            print(f"📊 Evaluation: {result['evaluation_result']}")
            print(f"🔁 Retry Count: {result['retry_count']}")
            print(f"\n💡 Final Answer:\n{result['final_answer']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()