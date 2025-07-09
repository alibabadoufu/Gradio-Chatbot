"""
Example Usage: Reasoning-Augmented Retrieval Agent

This script demonstrates how to use the ReasoningAugmentedRAG agent.
It includes setup instructions, example usage, and error handling.

Setup Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Set up OpenAI API key: export OPENAI_API_KEY="your-api-key"
3. Run this script: python rag_agent_example.py

The agent will:
- Take a user query
- Decompose and rewrite it for better retrieval
- Retrieve relevant context from documents
- Evaluate if the context is sufficient
- Generate a comprehensive answer
"""

import os
import sys
from dotenv import load_dotenv
from reasoning_augmented_retrieval import ReasoningAugmentedRAG

# Load environment variables
load_dotenv()

def setup_environment():
    """Check if required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        print("Or add it to a .env file in the same directory")
        return False
    return True

def load_sample_documents():
    """Load sample documents for the knowledge base"""
    return [
        # Technology and Programming
        "Python is a versatile, high-level programming language created by Guido van Rossum in 1991. It's widely used for web development, data science, artificial intelligence, automation, and scientific computing. Python's syntax is designed to be readable and simple, making it an excellent choice for beginners and experienced developers alike.",
        
        # AI and Machine Learning
        "Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical algorithms to identify patterns in data and make predictions or decisions. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
        
        # Deep Learning
        "Deep learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition. Deep learning requires large amounts of data and computational power.",
        
        # LangGraph and LangChain
        "LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs). It extends LangChain by providing graph-based workflows and state management capabilities. LangGraph allows developers to create complex AI agents that can maintain state across multiple interactions and coordinate different AI components.",
        
        # RAG Systems
        "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. It works by first retrieving relevant documents from a knowledge base, then using this context to generate more accurate and informative responses. RAG systems help reduce hallucinations in LLMs by grounding responses in factual information.",
        
        # Vector Databases
        "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for similarity search, recommendation systems, and RAG applications. Popular vector databases include Pinecone, Weaviate, and Chroma. They use techniques like approximate nearest neighbor search to quickly find similar vectors.",
        
        # Natural Language Processing
        "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves tasks like text analysis, sentiment analysis, language translation, named entity recognition, and text summarization. Modern NLP heavily relies on transformer-based models like BERT and GPT.",
        
        # Data Science
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, domain expertise, and programming skills to analyze and interpret complex data sets to inform business decisions.",
        
        # Web Development
        "Web development involves creating websites and web applications. It includes frontend development (user interface using HTML, CSS, JavaScript), backend development (server-side logic using languages like Python, Node.js, or Java), and database management. Modern web development often uses frameworks like React, Angular, Django, or Flask.",
        
        # Software Engineering
        "Software engineering is the systematic approach to designing, developing, and maintaining software systems. It involves requirements analysis, system design, implementation, testing, deployment, and maintenance. Good software engineering practices include version control, code reviews, testing, documentation, and following design patterns and principles."
    ]

def interactive_demo():
    """Run an interactive demonstration of the RAG agent"""
    print("🤖 Reasoning-Augmented Retrieval Agent - Interactive Demo")
    print("=" * 60)
    
    if not setup_environment():
        return
    
    try:
        # Initialize the agent
        print("🔧 Initializing RAG agent...")
        agent = ReasoningAugmentedRAG(model_name="gpt-3.5-turbo", max_retries=2)
        
        # Setup vector store
        print("📚 Setting up knowledge base...")
        documents = load_sample_documents()
        agent.setup_vectorstore(documents)
        
        print("✅ Agent initialized successfully!")
        print("\nYou can now ask questions about:")
        print("• Python programming and web development")
        print("• Machine learning and AI concepts")
        print("• LangGraph and RAG systems")
        print("• Data science and software engineering")
        print("• Type 'quit' to exit")
        
        print("\n" + "=" * 60)
        
        while True:
            # Get user input
            user_query = input("\n🤔 Enter your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the RAG agent!")
                break
            
            if not user_query:
                print("❌ Please enter a valid question.")
                continue
            
            print(f"\n🔍 Processing query: {user_query}")
            print("-" * 40)
            
            try:
                # Run the agent
                result = agent.run(user_query)
                
                # Display results
                print(f"🧠 Analysis Plan: {result['plan']}")
                print(f"🔄 Rewritten Query: {result['rewritten_query']}")
                
                if result['decomposed_queries']:
                    print(f"📝 Sub-questions: {', '.join(result['decomposed_queries'])}")
                
                evaluation = result['evaluation_result']
                print(f"📊 Context Evaluation: {evaluation.get('reason', 'N/A')}")
                print(f"🎯 Confidence: {evaluation.get('confidence', 0)}/100")
                
                if result['retry_count'] > 0:
                    print(f"🔁 Retrieval attempts: {result['retry_count'] + 1}")
                
                print(f"\n💡 Answer:")
                print(f"{result['final_answer']}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("Please try rephrasing your question or check your API key.")
            
            print("\n" + "=" * 60)
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Please check your OpenAI API key and internet connection.")

def batch_demo():
    """Run a batch demonstration with predefined queries"""
    print("🚀 Reasoning-Augmented Retrieval Agent - Batch Demo")
    print("=" * 60)
    
    if not setup_environment():
        return
    
    try:
        # Initialize the agent
        print("🔧 Initializing RAG agent...")
        agent = ReasoningAugmentedRAG(model_name="gpt-3.5-turbo", max_retries=2)
        
        # Setup vector store
        documents = load_sample_documents()
        agent.setup_vectorstore(documents)
        
        # Test queries
        test_queries = [
            "What is Python and what are its main applications?",
            "Explain the difference between machine learning and deep learning",
            "How does RAG work and why is it useful?",
            "What is LangGraph and how does it relate to LangChain?",
            "What are vector databases and why are they important for AI?",
            "Compare supervised and unsupervised learning approaches"
        ]
        
        print("🔍 Running batch queries...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)
            
            try:
                result = agent.run(query)
                
                print(f"🧠 Plan: {result['plan']}")
                print(f"🔄 Rewritten: {result['rewritten_query']}")
                print(f"📊 Evaluation: {result['evaluation_result'].get('reason', 'N/A')}")
                print(f"🎯 Confidence: {result['evaluation_result'].get('confidence', 0)}/100")
                print(f"🔁 Retries: {result['retry_count']}")
                print(f"\n💡 Answer:\n{result['final_answer']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("\n" + "=" * 60)
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")

def main():
    """Main function to choose demo mode"""
    print("🤖 Reasoning-Augmented Retrieval Agent Demo")
    print("Choose a demo mode:")
    print("1. Interactive Demo (ask your own questions)")
    print("2. Batch Demo (predefined test queries)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_demo()
            break
        elif choice == '2':
            batch_demo()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()