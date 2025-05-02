# **Enterprise Knowledge Assistant with Multi-Agent Agentic RAG CI/CD Pipeline**

## **Project Overview**
This project implements an **Enterprise Knowledge Assistant** using a **Multi-Agent Agentic RAG (Retrieval-Augmented Generation)** approach. The system enables document-based question answering from PDF files by utilizing a powerful combination of **LangChain**, **LangGraph**, **Groq’s LLaMA model**, **Chroma DB**, and **OpenAI's GPT-4**, integrating them into a seamless **CI/CD pipeline** for automation and efficiency.

---

## **Technologies Used**
- **LangChain & LangGraph**: For document processing, retrieval, and workflow orchestration.
- **Groq LLaMA Model**: Used for generating responses to queries based on the document context.
- **Chroma DB**: A vector store used for efficient document indexing and retrieval.
- **Streamlit**: Provides the user interface for interacting with the knowledge assistant.
- **OpenAI GPT-4**: For handling conversational AI responses and embedding generation.
- **Unstructured Library**: For extracting rich content (images, tables, and text) from PDFs.

---

## **Key Features**
1. **PDF Document Processing**: 
   - Upload PDF files.
   - Automatically extract relevant content, including images, tables, and text.
   - Summarize extracted content for efficient retrieval.

2. **Multi-Agent Workflow**: 
   - Retrieval and response generation are handled by different agents in a DAG (Directed Acyclic Graph)-based workflow using **LangGraph**.
   - **State Graph** coordinates the flow between retrieving context and generating responses.

3. **Dynamic Context Retrieval**:
   - Uses a **vector store (Chroma DB)** to retrieve relevant document chunks based on the user's query.
   - Context includes text and image references.

4. **Answer Generation**:
   - Utilizes **Groq’s LLaMA model** to generate high-quality responses based on the document context.
   - The system ensures that responses are tailored to the content within the documents.

5. **CI/CD Pipeline**:
   - Automated document processing and retrieval through the pipeline.
   - Supports dynamic updates and deployment using environment variables, ensuring scalability and continuous integration.

---

## **How to Run the Application**

### 1. **Install Dependencies**
First, install all necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. **Set Up Environment Variables**
Ensure the following environment variables are set:
- `GROQ_API_KEY`: Your API key for Groq’s LLaMA model.
- `PDF_PATH`: Path to a single PDF file (optional).
- `PDF_DIR`: Directory containing PDFs (optional).
- `EXTRACTED_IMG_DIR`: Directory to store extracted images (optional).
- `PERSIST_DIR`: Directory to persist the Chroma DB vector store (optional).

### 3. **Run the Application**
Run the Streamlit app:
```bash
streamlit run app.py
```

This will start the application on your local server, where you can interact with the knowledge assistant via a user-friendly web interface.

---

## **How It Works**

1. **Document Upload**: 
   - Users upload a PDF document using the sidebar. 
   - The system processes the document by extracting text, tables, and images.

2. **Document Summarization**:
   - Extracted content is summarized for easier retrieval. Summaries are generated using the **Groq LLaMA model** to optimize for retrieval.

3. **Knowledge Retrieval**:
   - When a user submits a question, the system retrieves relevant content (text and images) from the processed documents using **Chroma DB**.
   - The retrieved context is used to answer the user's question.

4. **Answer Generation**:
   - The system generates a detailed answer using **Groq LLaMA** based on the context retrieved from the vector store.

---

## **Pipeline Workflow**

- **Agent 1**: Retrieves relevant content from the vector store (Chroma DB).
- **Agent 2**: Generates an answer using the Groq LLaMA model based on the retrieved context.

The workflow is orchestrated using **LangGraph**, where:
- The first agent fetches relevant context from the document.
- The second agent processes the context and generates a comprehensive answer.

---

## **CI/CD Pipeline Integration**
- The project is designed for **continuous integration** and **continuous deployment**. As documents are processed, the system updates the vector store, allowing for easy scalability and integration into larger enterprise knowledge systems.
  
---

## **Project Structure**
```
|-- app.py                          # Streamlit app for the UI
|-- requirements.txt                # Python dependencies
|-- .env                            # Environment variables
|-- data/                           # PDF data directory
|-- extracted_images/               # Directory for extracted images
|-- chroma_db/                      # Chroma DB directory for vector store
|-- utils/                          # Utility functions for image and PDF processing
|-- workflows/                      # LangGraph workflows for agent coordination
```

---

## **Troubleshooting**

- **Missing Dependencies**: Ensure that all dependencies are installed, especially the **unstructured library** for rich PDF parsing.
- **Environment Variables**: Make sure that the necessary API keys (like `GROQ_API_KEY`) are correctly set in your `.env` file.
- **File Upload Limitations**: Streamlit may have a file upload size limitation. For larger documents, consider increasing the file upload size in Streamlit settings.

---

## **Conclusion**
This **Enterprise Knowledge Assistant** project leverages advanced AI and document processing techniques to create an intelligent assistant capable of interacting with enterprise-level documents. By combining multi-agent workflows, retrieval-augmented generation, and a scalable CI/CD pipeline, the system offers a robust solution for enterprise knowledge management.

