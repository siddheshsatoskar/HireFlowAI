"""
Document Ingestion Service

Handles reading and processing of resume documents.
"""

from pathlib import Path
from typing import List
from llama_index.core import Document, SimpleDirectoryReader


class DocumentIngestionManager:
    """
    Manages document ingestion and processing for HireFlow.
    
    This class handles:
    - Reading single resumes
    - Ingesting multiple resumes from directories
    - Extracting and processing document content
    """
    
    def __init__(self):
        """Initialize the DocumentIngestionManager."""
        self.documents = []
    
    def read_single_resume(self, file_path: str) -> Document:
        """
        Read a single resume from a file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Document: Parsed document object
            
        Raises:
            ValueError: If no content could be extracted
        """
        print("=" * 60)
        print("STEP 2: Reading a Single Resume")
        print("=" * 60)
        
        print(f"Reading resume: {file_path}")
        
        # Read PDF using LlamaIndex SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        if documents:
            doc = documents[0]
            print(f"✓ Successfully read resume")
            print(f"  - File: {Path(file_path).name}")
            print(f"  - Content length: {len(doc.text)} characters")
            print(f"  - Preview:")
            print(f"    {doc.text[:300]}...")
            print()
            return doc
        else:
            raise ValueError("No content could be extracted from the resume")
    
    def ingest_multiple_resumes(self, resume_dir: str) -> List[Document]:
        """
        Ingest multiple resumes from a directory.
        
        Args:
            resume_dir: Directory containing resume files
            
        Returns:
            List[Document]: List of parsed document objects
        """
        print("=" * 60)
        print("STEP 3: Ingesting Multiple Resumes")
        print("=" * 60)
        
        resume_path = Path(resume_dir)
        
        if not resume_path.exists():
            print(f"⚠ Directory not found: {resume_dir}")
            return []
        
        # Read all PDF files from directory
        reader = SimpleDirectoryReader(
            input_dir=str(resume_path),
            required_exts=[".pdf"],
            recursive=True
        )
        self.documents = reader.load_data()
        
        print(f"✓ Successfully ingested {len(self.documents)} document(s)")
        for i, doc in enumerate(self.documents, 1):
            file_name = doc.metadata.get('file_name', 'Unknown')
            char_count = len(doc.text)
            print(f"  {i}. {file_name} - {char_count} chars")
        print()
        
        return self.documents
    
    def get_documents(self) -> List[Document]:
        """
        Get the list of ingested documents.
        
        Returns:
            List[Document]: List of documents
        """
        return self.documents
    
    def get_document_count(self) -> int:
        """
        Get the count of ingested documents.
        
        Returns:
            int: Number of documents
        """
        return len(self.documents)


