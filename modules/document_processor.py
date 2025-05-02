import os
import logging
from typing import Optional, List

import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llama_index.core import Document, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

from modules.image_processor import generate_img_summaries
from config import GROQ_MODEL, PERSIST_DIR, PDF_DIR, EXTRACTED_IMG_DIR

logger = logging.getLogger(__name__)

# Check for unstructured library
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import (
        Table, Header, Footer, NarrativeText, Title, Text, ListItem, Image
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured library not available. Falling back to basic PDF extraction.")

def process_pdf_basic(pdf_path: str) -> List[Document]:
    """
    Basic PDF processing fallback when unstructured is not available
    """
    try:
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return [Document(text=page.page_content) for page in pages]
    except Exception as e:
        logger.error(f"Error in basic PDF processing for {pdf_path}: {str(e)}")
        return []

def process_pdf_advanced(pdf_path: str, extracted_img_dir: str, summarize_chain) -> List[Document]:
    """
    Advanced PDF processing using unstructured library
    """
    final_docs = []
    try:
        # Extract elements from PDF
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=extracted_img_dir
        )
        
        # Separate tables and text
        tables, texts = [], []
        for element in elements:
            if isinstance(element, Table):
                tables.append(str(element))
            elif isinstance(element, (Text, NarrativeText, ListItem, Title)):
                texts.append(str(element))
        
        # Process tables
        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
            for summary in table_summaries:
                final_docs.append(Document(text=summary, metadata={"type": "Table"}))
        
        # Process texts
        if texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
            for summary in text_summaries:
                final_docs.append(Document(text=summary, metadata={"type": "Text"}))
                
    except Exception as e:
        logger.error(f"Error in advanced PDF processing for {pdf_path}: {str(e)}")
    
    return final_docs

def get_vectorstore(
    pdf_dir: Optional[str] = None, 
    extracted_img_dir: Optional[str] = None, 
    persist_dir: Optional[str] = None,
    process_images: bool = False
) -> Optional[ChromaVectorStore]:
    """
    Create and return a vector store from PDF documents
    """
    # Use default values from config if not provided
    pdf_dir = pdf_dir or PDF_DIR
    extracted_img_dir = extracted_img_dir or EXTRACTED_IMG_DIR
    persist_dir = persist_dir or PERSIST_DIR
    
    # Create directories if they don't exist
    os.makedirs(extracted_img_dir, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Handle single PDF file case
    single_pdf_path = os.environ.get("PDF_PATH")
    if single_pdf_path and os.path.isfile(single_pdf_path):
        pdf_files = [single_pdf_path]
    else:
        # Check if pdf_dir exists and contains PDFs
        if not os.path.isdir(pdf_dir):
            logger.error(f"PDF directory {pdf_dir} not found")
            return None
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning("No PDF files found to process")
        return None
    
    # Set up summarization chain
    prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. text: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    # Initialize Groq model
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable is not set.")
        return None
    
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL
    )
    
    summarize_chain = {"element": lambda x: x} | prompt | groq_chat | StrOutputParser()
    
    final_docs = []
    
    # Process each PDF
    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path}")
        
        if not UNSTRUCTURED_AVAILABLE:
            # Use basic PDF processing
            docs = process_pdf_basic(pdf_path)
            final_docs.extend(docs)
        else:
            # Use advanced PDF processing
            docs = process_pdf_advanced(pdf_path, extracted_img_dir, summarize_chain)
            final_docs.extend(docs)
    
    # # Process images if enabled
    if process_images and UNSTRUCTURED_AVAILABLE:
        try:
            if os.path.isdir(extracted_img_dir) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(extracted_img_dir)):
                _, image_summaries = generate_img_summaries(extracted_img_dir)
                for summary in image_summaries:
                    final_docs.append(Document(text=summary, metadata={"type": "Image", "is_image": True}))
                logger.info(f"Processed {len(image_summaries)} images")
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
    
    # Create and return vectorstore
    if not final_docs:
        logger.warning("No documents were processed.")
        return None
    
    try:
        # Use OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create Chroma client and collection
        chroma_client = PersistentClient(path=persist_dir)
        collection_name = "document_collection"
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        
        # Create a storage context
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        
        # Create llama_index Documents and add to index
        index = VectorStoreIndex.from_documents(
            final_docs,
            storage_context=storage_context,
        )
        
        logger.info(f"Successfully created vectorstore with {len(final_docs)} documents")
        
        # Return the vector store
        return index.vector_store
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        return None