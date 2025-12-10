"""
Re-Ranking Service

Handles re-ranking of retrieved candidates based on various criteria.
"""

from typing import List
from llama_index.core.schema import NodeWithScore


class ReRankingManager:
    """
    Manages candidate re-ranking and scoring.
    
    This class handles:
    - Re-ranking candidates based on similarity scores
    - Selecting top N candidates
    - Providing ranking explanations
    """
    
    def __init__(self):
        """Initialize the ReRankingManager."""
        self.top_candidates = []
    
    def simple_rerank(self, retrieved_nodes: List[NodeWithScore], 
                     job_description: str, top_n: int = 3) -> List[NodeWithScore]:
        """
        Simple re-ranking based on similarity scores.
        
        Args:
            retrieved_nodes: List of retrieved nodes with scores
            job_description: Job description for context
            top_n: Number of top candidates to select
            
        Returns:
            List[NodeWithScore]: Top N candidates after re-ranking
        """
        print("=" * 60)
        print("STEP 6: Re-ranking Candidates")
        print("=" * 60)
        
        print(f"Re-ranking {len(retrieved_nodes)} candidates...")
        print(f"Using similarity scores from semantic search")
        print()
        
        # Already ranked by similarity, just take top N
        self.top_candidates = retrieved_nodes[:top_n]
        
        print(f"✓ Top {len(self.top_candidates)} candidates after re-ranking:")
        print()
        
        for i, node in enumerate(self.top_candidates, 1):
            print(f"Rank #{i}:")
            print(f"  - Score: {node.score:.4f}")
            print(f"  - Source: {node.node.metadata.get('file_name', 'Unknown')}")
            print(f"  - Match Reason: High semantic similarity to job requirements")
            print()
        
        return self.top_candidates
    
    def advanced_rerank(self, retrieved_nodes: List[NodeWithScore],
                       job_description: str, top_n: int = 3,
                       boost_keywords: List[str] = None) -> List[NodeWithScore]:
        """
        Advanced re-ranking with keyword boosting.
        
        Args:
            retrieved_nodes: List of retrieved nodes with scores
            job_description: Job description for context
            top_n: Number of top candidates to select
            boost_keywords: Keywords to boost in ranking
            
        Returns:
            List[NodeWithScore]: Top N candidates after advanced re-ranking
        """
        print("=" * 60)
        print("STEP 6: Advanced Re-ranking Candidates")
        print("=" * 60)
        
        print(f"Re-ranking {len(retrieved_nodes)} candidates...")
        print(f"Using similarity scores with keyword boosting")
        
        if boost_keywords:
            print(f"Boost keywords: {', '.join(boost_keywords)}")
        print()
        
        # Apply keyword boosting to scores
        boosted_nodes = []
        for node in retrieved_nodes:
            boosted_score = node.score
            
            if boost_keywords:
                text_lower = node.text.lower()
                keyword_matches = sum(1 for kw in boost_keywords if kw.lower() in text_lower)
                # Boost score by 5% for each keyword match, max 20%
                boost_factor = min(keyword_matches * 0.05, 0.20)
                boosted_score = boosted_score * (1 + boost_factor)
            
            # Create a copy with boosted score
            boosted_node = NodeWithScore(node=node.node, score=boosted_score)
            boosted_nodes.append(boosted_node)
        
        # Sort by boosted score
        boosted_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Take top N
        self.top_candidates = boosted_nodes[:top_n]
        
        print(f"✓ Top {len(self.top_candidates)} candidates after re-ranking:")
        print()
        
        for i, node in enumerate(self.top_candidates, 1):
            print(f"Rank #{i}:")
            print(f"  - Score: {node.score:.4f}")
            print(f"  - Source: {node.node.metadata.get('file_name', 'Unknown')}")
            print(f"  - Match Reason: High semantic similarity with keyword relevance")
            print()
        
        return self.top_candidates
    
    def get_top_candidates(self) -> List[NodeWithScore]:
        """
        Get the top ranked candidates.
        
        Returns:
            List[NodeWithScore]: Top candidates
        """
        return self.top_candidates
    
    def get_candidate_count(self) -> int:
        """
        Get the count of top candidates.
        
        Returns:
            int: Number of top candidates
        """
        return len(self.top_candidates)


