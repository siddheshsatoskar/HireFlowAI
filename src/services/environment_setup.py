"""
Environment Setup Service

Handles the initialization and configuration of the embedding model and global settings.
"""

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EnvironmentSetup:
    """
    Manages environment setup and configuration for HireFlow.
    
    This class handles:
    - Loading and configuring embedding models
    - Setting global LlamaIndex settings
    - Configuring chunking parameters
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the EnvironmentSetup.
        
        Args:
            model_name: HuggingFace model name for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = None
    
    def setup(self):
        """
        Setup the local embedding model and configure global settings.
        
        Returns:
            HuggingFaceEmbedding: Configured embedding model
        """
        print("=" * 60)
        print("STEP 1: Setting up Local Embedding Model")
        print("=" * 60)
        
        # Use HuggingFace embeddings directly
        self.embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
        print(f"✓ Local embedding model loaded: {self.model_name}")
        print(f"✓ Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        print()
        
        return self.embed_model
    
    def get_embed_model(self):
        """
        Get the embedding model (setup if not already done).
        
        Returns:
            HuggingFaceEmbedding: The embedding model
        """
        if self.embed_model is None:
            return self.setup()
        return self.embed_model

