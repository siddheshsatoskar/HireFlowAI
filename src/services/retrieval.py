"""
Retrieval Service

Handles candidate retrieval from the vector store using semantic search.
"""

from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class RetrievalManager:
    """
    Manages candidate retrieval using semantic search.
    
    This class handles:
    - Creating retrievers from vector store indices
    - Performing semantic search queries
    - Retrieving top-k similar candidates
    """
    
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 5):
        """
        Initialize the RetrievalManager.
        
        Args:
            index: Vector store index to retrieve from
            similarity_top_k: Number of top similar results to retrieve
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.retriever = None
        self.retrieved_nodes = []
    
    def create_retriever(self) -> VectorIndexRetriever:
        """
        Create a retriever from the index.
        
        Returns:
            VectorIndexRetriever: Configured retriever
        """
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k
        )
        return self.retriever
    
    def retrieve(self, query: str, top_k: int = None) -> List[NodeWithScore]:
        """
        Retrieve candidates matching the query.
        
        Args:
            query: Search query (typically job description)
            top_k: Number of results to retrieve (overrides default if provided)
            
        Returns:
            List[NodeWithScore]: Retrieved nodes with similarity scores
        """
        print("=" * 60)
        print("STEP 5: Retrieval - Finding Top Candidates")
        print("=" * 60)
        
        print(f"Job Description Query:")
        print(f"  {query[:200]}...")
        print()
        
        # Update top_k if provided
        if top_k is not None:
            self.similarity_top_k = top_k
        
        # Create retriever if not already created or if top_k changed
        if self.retriever is None or top_k is not None:
            self.create_retriever()
        
        # Retrieve relevant nodes
        self.retrieved_nodes = self.retriever.retrieve(query)
        
        print(f"✓ Retrieved {len(self.retrieved_nodes)} candidate node(s)")
        print()
        
        # Display results
        for i, node in enumerate(self.retrieved_nodes, 1):
            print(f"Result {i}:")
            print(f"  - Similarity Score: {node.score:.4f}")
            print(f"  - Source: {node.node.metadata.get('file_name', 'Unknown')}")
            print(f"  - Content Preview:")
            print(f"    {node.text[:150]}...")
            print()
        
        return self.retrieved_nodes
    
    def get_retrieved_nodes(self) -> List[NodeWithScore]:
        """
        Get the last retrieved nodes.
        
        Returns:
            List[NodeWithScore]: Retrieved nodes
        """
        return self.retrieved_nodes
    
    def get_retrieval_count(self) -> int:
        """
        Get the count of retrieved nodes.
        
        Returns:
            int: Number of retrieved nodes
        """
        return len(self.retrieved_nodes)


