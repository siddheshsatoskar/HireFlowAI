"""
HireFlow Simple Pipeline - Local Embeddings Version (Refactored with Classes)
Uses HuggingFace embeddings (runs locally, no API quota needed)
Demonstrates: Reading resumes → Ingestion → Vector Store → Retrieval → Re-ranking

This module now serves as a compatibility layer that wraps the new class-based architecture.
"""

import os
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import NodeWithScore

# Import the new service classes
from src.services import (
    EnvironmentSetup,
    DocumentIngestionManager,
    VectorStoreManager,
    RetrievalManager,
    ReRankingManager,
    EvaluationManager,
    ChatbotManager
)

from config.settings import DATA_DIR, TOP_K_CANDIDATES, GEMINI_API_KEY


# Backward compatibility functions that wrap the new classes
def setup_local_environment():
    """
    Step 1: Setup Local Embedding Model
    
    Returns:
        HuggingFaceEmbedding: Configured embedding model
    """
    env_setup = EnvironmentSetup()
    return env_setup.setup()


def read_single_resume(file_path: str) -> Document:
    """
    Step 2: Reading from 1 resume
    
    Args:
        file_path: Path to the resume file
        
    Returns:
        Document: Parsed document object
    """
    doc_manager = DocumentIngestionManager()
    return doc_manager.read_single_resume(file_path)


def ingest_multiple_resumes(resume_dir: str) -> List[Document]:
    """
    Step 3: Ingesting multiple resumes
    
    Args:
        resume_dir: Directory containing resumes
        
    Returns:
        List[Document]: List of ingested documents
    """
    doc_manager = DocumentIngestionManager()
    return doc_manager.ingest_multiple_resumes(resume_dir)


def create_vector_store(documents: List[Document]) -> VectorStoreIndex:
    """
    Step 4: Creating Vector Store with FAISS
    
    Args:
        documents: List of documents to index
        
    Returns:
        VectorStoreIndex: Created vector store index
    """
    vector_store_manager = VectorStoreManager()
    return vector_store_manager.create_and_save_index(documents)


def retrieve_candidates(index: VectorStoreIndex, job_description: str, top_k: int = 5):
    """
    Step 5: Retrieval - Find top candidates
    
    Args:
        index: Vector store index
        job_description: Job description query
        top_k: Number of candidates to retrieve
        
    Returns:
        List[NodeWithScore]: Retrieved candidate nodes
    """
    retrieval_manager = RetrievalManager(index, similarity_top_k=top_k)
    return retrieval_manager.retrieve(job_description)


def simple_rerank(retrieved_nodes, job_description: str, top_n: int = 3):
    """
    Step 6: Simple Re-ranking based on keyword matching
    
    Args:
        retrieved_nodes: List of retrieved nodes
        job_description: Job description for context
        top_n: Number of top candidates to return
        
    Returns:
        List[NodeWithScore]: Top N candidates
    """
    reranking_manager = ReRankingManager()
    return reranking_manager.simple_rerank(retrieved_nodes, job_description, top_n)


def generate_summary_report(top_candidates, job_description: str):
    """
    Generate a summary report
    
    Args:
        top_candidates: List of top candidates
        job_description: Job description for context
        
    Returns:
        bool: True if successful
    """
    evaluation_manager = EvaluationManager()
    return evaluation_manager.generate_summary_report(top_candidates, job_description)


def generate_candidate_evaluation(index: VectorStoreIndex, job_description: str):
    """
    Generate detailed candidate evaluation using Gemini LLM
    
    Args:
        index: Vector store index
        job_description: Job description for evaluation
        
    Returns:
        str: Evaluation report
    """
    evaluation_manager = EvaluationManager(gemini_api_key=GEMINI_API_KEY)
    return evaluation_manager.generate_detailed_evaluation(index, job_description)


def interactive_chatbot(index: VectorStoreIndex, job_description: str = None):
    """
    Step 8: Interactive Chatbot for Candidate Questions
    
    Args:
        index: Vector store index
        job_description: Optional job description for context
        
    Returns:
        bool: True if successful
    """
    chatbot_manager = ChatbotManager(index, gemini_api_key=GEMINI_API_KEY)
    return chatbot_manager.start_interactive_session(job_description)


def main():
    """Main pipeline execution"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  HIREFLOW - INTELLIGENT CANDIDATE SEARCH".center(58) + "*")
    print("*" + "  (Local Embeddings Version - Class-Based)".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    try:
        # Step 1: Setup environment
        embed_model = setup_local_environment()
        
        # Step 2: Read single resume
        # sample_resume_path = DATA_DIR / "Senior_Accountant_Position.pdf"
        # single_doc = read_single_resume(str(sample_resume_path))
        
        # Step 3: Ingest resumes
        documents = ingest_multiple_resumes(str(DATA_DIR))
        
        if not documents:
            print("No documents found. Using single document for demo.")
            # documents = [single_doc]
        
        # Step 4: Create vector store
        index = create_vector_store(documents)
        
        # Define job description
        job_description = """
        We are looking for a Senior Accountant with 5+ years of experience in 
        financial reporting, tax preparation, and audit support. The ideal candidate 
        should have strong knowledge of GAAP, proficiency in accounting software 
        (QuickBooks, SAP), and excellent analytical skills. CPA certification is required. 
        Experience in the manufacturing or tech industry is a plus.
        """
        
        # Step 5: Retrieve candidates
        retrieved_nodes = retrieve_candidates(index, job_description, top_k=5)
        
        # Step 6: Re-rank candidates
        top_candidates = simple_rerank(retrieved_nodes, job_description, top_n=3)
        
        # Generate summary report
        generate_summary_report(top_candidates, job_description)

        # Step 7: Generate candidate evaluation with Gemini LLM
        evaluation = generate_candidate_evaluation(index, job_description)

        # Step 8: Interactive Chatbot
        interactive_chatbot(index, job_description)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
