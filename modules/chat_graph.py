import os
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters

from modules.document_processor import get_vectorstore
from config import GROQ_MODEL, VECTORSTORE_TOP_K

logger = logging.getLogger(__name__)

# Define state schema
class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    context: List[str]
    current_question: str
    current_answer: str

def retrieve_context(state: ChatState) -> ChatState:
    """
    Retrieve relevant document chunks based on the current question
    """
    try:
        # Get the vectorstore
        vectorstore = get_vectorstore()
        if vectorstore is None:
            state["context"] = ["Failed to load document database. Please check the logs."]
            return state
        
        # Recreate the index from the vectorstore
        index = VectorStoreIndex.from_vector_store(vectorstore)
        
        # Get retriever
        retriever = index.as_retriever(similarity_top_k=VECTORSTORE_TOP_K)
        
        # Retrieve nodes
        retrieval_results = retriever.retrieve(state["current_question"])
        
        # Process retrieved documents based on type
        context_parts = []
        image_references = []
        
        for result in retrieval_results:
            node = result.node
            # Handle image documents differently
            if node.metadata.get("is_image", False):
                image_type = node.metadata.get("type", "image")
                image_ref = f"The document contains a {image_type} that may be relevant to your question."
                image_references.append(image_ref)
            else:
                # Add text content with source information
                content = node.text
                doc_type = node.metadata.get("type", "Text")
                context_parts.append(f"[{doc_type}] {content}")
        
        # Combine all context
        all_context = context_parts + image_references
        state["context"] = all_context if all_context else ["No relevant information found in the document."]
        logger.info(f"Retrieved {len(all_context)} context items")
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}")
        state["context"] = [f"Error retrieving context: {str(e)}"]
    
    return state

def generate_response(state: ChatState) -> ChatState:
    """
    Generate a response based on the question and retrieved context
    """
    try:
        # Get API key from environment
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            state["current_answer"] = "Error: GROQ_API_KEY environment variable is not set."
            return state
            
        # Initialize Groq model
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GROQ_MODEL
        )
        
        # Create system message with context
        context_text = "\n\n".join(state["context"])
        system_msg = SystemMessage(content=f"""You are a helpful assistant that answers questions based on the provided context. 
Use the information from the document context to provide accurate and comprehensive answers.
Start your answer directly without small talk.

Context:
{context_text}
""")
        
        # Create user message
        user_msg = HumanMessage(content=state["current_question"])
        
        # Generate response
        messages = [system_msg, user_msg]
        response = groq_chat.invoke(messages)
        
        # Update state with response
        state["current_answer"] = response.content
        logger.info("Generated response successfully")
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        state["current_answer"] = f"Error generating response: {str(e)}"
    
    return state

def build_graph() -> StateGraph:
    """
    Build and return the state graph for the chat workflow
    """
    # Create a new StateGraph with the ChatState type
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve_context")
    
    # Compile the graph
    return workflow.compile()