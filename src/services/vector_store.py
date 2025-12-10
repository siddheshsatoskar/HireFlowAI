"""
Vector Store Service

Handles creation and management of vector stores for semantic search.
"""

from pathlib import Path
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage


class VectorStoreManager:
    """
    Manages vector store creation and persistence for HireFlow.
    
    This class handles:
    - Creating vector store indices from documents
    - Saving and loading indices from disk
    - Managing vector database operations
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the VectorStoreManager.
        
        Args:
            persist_dir: Directory to persist the vector store (optional)
        """
        self.index = None
        if persist_dir:
            self.persist_dir = Path(persist_dir)
        else:
            # Default to vector_store directory in project root
            self.persist_dir = Path(__file__).parent.parent.parent / "vector_store"
        
        self.persist_dir.mkdir(exist_ok=True)
    
    def create_index(self, documents: List[Document], show_progress: bool = True) -> VectorStoreIndex:
        """
        Create a vector store index from documents.
        
        Args:
            documents: List of documents to index
            show_progress: Whether to show progress during indexing
            
        Returns:
            VectorStoreIndex: Created vector store index
        """
        print("=" * 60)
        print("STEP 4: Creating Vector Store")
        print("=" * 60)
        
        print("Creating vector embeddings and building FAISS index...")
        print("(This may take a minute for local embeddings)")
        
        # Create vector store index (uses FAISS by default)
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=show_progress
        )
        
        print("✓ Vector store index created successfully")
        print(f"  - Number of documents indexed: {len(documents)}")
        print(f"  - Vector database: FAISS (in-memory)")
        print()
        
        return self.index
    
    def save_index(self) -> bool:
        """
        Save the index to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.index is None:
            print("⚠ No index to save")
            return False
        
        try:
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
            print(f"✓ Index saved to: {self.persist_dir}")
            print()
            return True
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load an index from disk.
        
        Returns:
            VectorStoreIndex: Loaded index or None if failed
        """
        if not self.persist_dir.exists():
            print(f"⚠ Persist directory not found: {self.persist_dir}")
            return None
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self.index = load_index_from_storage(storage_context)
            print(f"✓ Index loaded from: {self.persist_dir}")
            print()
            return self.index
        except Exception as e:
            print(f"⚠ Could not load index: {e}")
            return None
    
    def create_and_save_index(self, documents: List[Document], show_progress: bool = True) -> VectorStoreIndex:
        """
        Create a vector store index and save it to disk.
        
        Args:
            documents: List of documents to index
            show_progress: Whether to show progress during indexing
            
        Returns:
            VectorStoreIndex: Created vector store index
        """
        self.create_index(documents, show_progress)
        self.save_index()
        return self.index
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index.
        
        Returns:
            VectorStoreIndex: Current index or None
        """
        return self.index


