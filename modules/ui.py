import os
import streamlit as st
import logging
from typing import Dict, Any

from modules.document_processor import get_vectorstore, UNSTRUCTURED_AVAILABLE
from modules.chat_graph import build_graph
from config import TEMP_UPLOAD_DIR

logger = logging.getLogger(__name__)

def setup_ui_config():
    """Configure the Streamlit UI settings"""
    st.set_page_config(
        page_title="PDF Document Q&A Chatbot",
        page_icon="📄",
        layout="wide"
    )

def show_sidebar():
    """Display sidebar content and controls"""
    st.sidebar.title("Settings & Information")
    
    # Display unstructured status
    if UNSTRUCTURED_AVAILABLE:
        st.sidebar.success("✅ Unstructured library is available")
    else:
        st.sidebar.warning("⚠️ Unstructured library not available - Install for better PDF parsing")
        with st.sidebar.expander("Installation Instructions"):
            st.code("""
# Install unstructured and dependencies
pip install "unstructured[pdf,docx]>=0.10.30" "unstructured-inference>=0.7.30"
pip install pdf2image pdfminer.six
pip install pillow-heif

# For Linux systems
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice
            """)
    
    # Check for environment variables
    if not os.environ.get("GROQ_API_KEY"):
        st.sidebar.warning("GROQ_API_KEY environment variable is not set.")
    if not os.environ.get("OPENAI_API_KEY"):
        st.sidebar.warning("OPENAI_API_KEY environment variable is not set.")

    # Display API key input fields
    with st.sidebar.expander("API Keys (Optional)"):
        groq_key = st.text_input("GROQ API Key", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")
        
        if st.button("Save Keys"):
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            st.success("API keys saved for this session")
    
    return st.sidebar

def handle_file_upload(sidebar):
    """Handle PDF file upload in the sidebar"""
    pdf_file = sidebar.file_uploader("Upload a PDF document", type="pdf")
    if pdf_file:
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        temp_path = os.path.join(TEMP_UPLOAD_DIR, pdf_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        os.environ["PDF_PATH"] = temp_path
        sidebar.success(f"Uploaded: {pdf_file.name}")
        
        # Process the document
        with st.spinner("Processing document..."):
            # Preload the vector store
            vectorstore = get_vectorstore()
            if vectorstore:
                sidebar.success("✅ Document processed and ready")
                return True
            else:
                sidebar.error("❌ Failed to process document")
                return False
    return None

def display_chat_interface():
    """Display the chat interface and handle user interactions"""
    st.title('PDF Document Q&A Chatbot')
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input('Ask a question about the document')
    
    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Store the user prompt in state
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Show a spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Initialize the graph
                graph = build_graph()
                
                # Run the graph
                result = graph.invoke({
                    "messages": st.session_state.messages,
                    "current_question": prompt,
                    "context": [],
                    "current_answer": ""
                })
                
                # Get the response
                response = result["current_answer"]
                
                # Display assistant message
                with st.chat_message('assistant'):
                    st.markdown(response)
                
                # Store assistant message in state
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                st.error(f"Error processing your request: {str(e)}")

def render_ui():
    """Main UI rendering function"""
    setup_ui_config()
    sidebar = show_sidebar()
    handle_file_upload(sidebar)
    display_chat_interface()