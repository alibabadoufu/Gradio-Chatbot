
import operator
from typing import Annotated, List, Tuple, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from .document_processor import load_vectorstore


class AgentState(TypedDict):
    chat_history: Annotated[List[BaseMessage], operator.add]
    question: str
    selected_vectorstore: str
    context: str
    response_type: str


class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in selecting the most relevant vectorstore based on the user's query. "
                       "Given the user's question and a list of available vectorstores, choose the one that is most likely to contain the answer. "
                       "Respond only with the name of the selected vectorstore."),
            ("user", "Question: {question}\nAvailable vectorstores: {vectorstores}")
        ])

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question based on the provided context."),
            ("user", "Question: {question}\nContext: {context}")
        ])

        self.response_type_prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on the user's question and the retrieved context, determine if the best response is a 'text' answer, 'chart' generation, or 'code' generation. "
                       "Respond only with 'text', 'chart', or 'code'."),
            ("user", "Question: {question}\nContext: {context}")
        ])

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("select_vectorstore", self.select_vectorstore)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("determine_response_type", self.determine_response_type)
        workflow.add_node("generate_text_response", self.generate_text_response)
        workflow.add_node("generate_chart", self.generate_chart)
        workflow.add_node("generate_code", self.generate_code)

        workflow.set_entry_point("select_vectorstore")

        workflow.add_edge("select_vectorstore", "retrieve_context")
        workflow.add_edge("retrieve_context", "determine_response_type")

        workflow.add_conditional_edges(
            "determine_response_type",
            self.route_response_type,
            {
                "text": "generate_text_response",
                "chart": "generate_chart",
                "code": "generate_code",
                "end": END, # If no specific response type is determined, end the graph
            },
        )

        workflow.add_edge("generate_text_response", END)
        workflow.add_edge("generate_chart", END)
        workflow.add_edge("generate_code", END)

        return workflow.compile()

    def select_vectorstore(self, state: AgentState):
        question = state["question"]
        available_vectorstores = state["available_vectorstores"]
        
        # This is a placeholder. In a real scenario, you'd have a list of available vectorstores.
        # For now, we'll assume the user provides the selected vectorstore directly.
        # Or, the agent could query a meta-database of vectorstores.
        # For this implementation, we'll use the 'selected_vectorstore' from the state directly.
        # If it's not provided, we'll default to the first one for demonstration.
        
        if state.get("selected_vectorstore"):
            selected_vs = state["selected_vectorstore"]
        elif available_vectorstores:
            selected_vs = available_vectorstores[0] # Default to first if not specified
        else:
            selected_vs = None # Handle case with no vectorstores

        print(f"Selected vectorstore: {selected_vs}")
        return {"selected_vectorstore": selected_vs}

    def retrieve_context(self, state: AgentState):
        question = state["question"]
        selected_vectorstore_name = state["selected_vectorstore"]
        
        if not selected_vectorstore_name:
            return {"context": "No vectorstore selected or available."}

        vectorstore = load_vectorstore(selected_vectorstore_name)
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        print(f"Retrieved context: {context[:200]}...")
        return {"context": context}

    def determine_response_type(self, state: AgentState):
        question = state["question"]
        context = state["context"]
        
        if "No vectorstore selected" in context:
            return {"response_type": "end"} # Route to end if no context

        response_type_chain = self.response_type_prompt | self.llm | StrOutputParser()
        response_type = response_type_chain.invoke({"question": question, "context": context})
        print(f"Determined response type: {response_type}")
        return {"response_type": response_type.strip().lower()}

    def generate_text_response(self, state: AgentState):
        question = state["question"]
        context = state["context"]
        
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({"question": question, "context": context})
        print(f"Generated text response: {response[:200]}...")
        return {"chat_history": [HumanMessage(content=response)]}

    def generate_chart(self, state: AgentState):
        # Placeholder for chart generation logic
        # In a real application, this would involve parsing context, generating data, and creating a chart.
        print("Generating chart (placeholder)...")
        return {"chat_history": [HumanMessage(content="I'm sorry, I cannot generate charts yet. This is a placeholder for future functionality.")]}

    def generate_code(self, state: AgentState):
        # Placeholder for code generation logic
        # In a real application, this would involve parsing context and generating executable code.
        print("Generating code (placeholder)...")
        return {"chat_history": [HumanMessage(content="I'm sorry, I cannot generate code yet. This is a placeholder for future functionality.")]}

    def route_response_type(self, state: AgentState):
        response_type = state["response_type"]
        if "text" in response_type:
            return "text"
        elif "chart" in response_type:
            return "chart"
        elif "code" in response_type:
            return "code"
        else:
            return "end" # Failsafe

    def invoke_agent(self, question: str, selected_vectorstore: str = None, available_vectorstores: List[str] = None):
        inputs = {"question": question, "chat_history": [], "selected_vectorstore": selected_vectorstore, "available_vectorstores": available_vectorstores}
        for s in self.graph.stream(inputs):
            print(s)
            print("------")
        return s


if __name__ == "__main__":
    # Example usage (requires a vectorstore to be processed first)
    # from document_processor import process_document
    # process_document("path/to/your/document.pdf")

    agent = RAGAgent()
    # Replace 'your_vectorstore_name' with an actual vectorstore name you've processed
    # result = agent.invoke_agent("What is the main topic of the document?", selected_vectorstore="your_vectorstore_name")
    # print(result)

    # Example with no specific vectorstore selected, assuming one exists
    # result = agent.invoke_agent("What is the capital of France?", available_vectorstores=["document1", "document2"])
    # print(result)

    # Example with a specific vectorstore selected
    # result = agent.invoke_agent("Summarize the key findings.", selected_vectorstore="example_doc")
    # print(result)

    # Example with a question that might lead to chart/code (placeholder responses)
    # result = agent.invoke_agent("Show me the sales data for Q3 2023.", selected_vectorstore="sales_report")
    # print(result)

    # result = agent.invoke_agent("Write a python function to calculate fibonacci sequence.", selected_vectorstore="programming_docs")
    # print(result)

    print("RAG Agent initialized. Run with appropriate vectorstores.")


