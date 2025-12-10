"""
Evaluation Service

Handles candidate evaluation and report generation.
"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.memory import ChatMemoryBuffer


class EvaluationManager:
    """
    Manages candidate evaluation and report generation.
    
    This class handles:
    - Generating summary reports
    - Creating detailed AI-powered evaluations
    - Candidate scoring and recommendations
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the EvaluationManager.
        
        Args:
            gemini_api_key: API key for Google Gemini (optional, loaded from config if not provided)
        """
        self.gemini_api_key = gemini_api_key
        self.llm = None
    
    def _initialize_llm(self):
        """Initialize the Gemini LLM if not already initialized."""
        if self.llm is None:
            if self.gemini_api_key is None:
                from config.settings import GEMINI_API_KEY
                self.gemini_api_key = GEMINI_API_KEY
            
            self.llm = GoogleGenAI(
                model="gemini-2.5-flash",
                api_key=self.gemini_api_key
            )
    
    def generate_summary_report(self, top_candidates: List[NodeWithScore], 
                               job_description: str) -> bool:
        """
        Generate a summary report for top candidates.
        
        Args:
            top_candidates: List of top candidates with scores
            job_description: Job description for context
            
        Returns:
            bool: True if successful
        """
        print("=" * 60)
        print("FINAL SUMMARY REPORT")
        print("=" * 60)
        print()
        
        print(f"Job Description: {job_description[:200]}...")
        print()
        print(f"Total Candidates Evaluated: {len(top_candidates)}")
        print()
        print("Top Recommendations:")
        print()
        
        for i, node in enumerate(top_candidates, 1):
            score_pct = node.score * 100
            recommendation = ("HIGHLY RECOMMENDED" if score_pct > 80 
                            else "RECOMMENDED" if score_pct > 60 
                            else "CONSIDER")
            
            print(f"{i}. Candidate from: {node.node.metadata.get('file_name', 'Unknown')}")
            print(f"   Match Score: {score_pct:.1f}%")
            print(f"   Recommendation: {recommendation}")
            print(f"   Key Snippet: {node.text[:200]}...")
            print()
        
        return True
    
    def generate_detailed_evaluation(self, index: VectorStoreIndex, 
                                    job_description: str) -> str:
        """
        Generate detailed AI-powered candidate evaluation.
        
        Args:
            index: Vector store index for querying
            job_description: Job description for evaluation
            
        Returns:
            str: Detailed evaluation report
        """
        print("=" * 60)
        print("STEP 7: GENERATING DETAILED CANDIDATE EVALUATION")
        print("=" * 60)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Create memory buffer
        memory = ChatMemoryBuffer(token_limit=1500)
        
        # Create query engine with Gemini LLM
        query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3,
            response_mode="compact"
        )
        
        evaluation_prompt = f"""
        Based on the job description below, evaluate the top candidate from the retrieved resume data:
        
        Job Description:
        {job_description}
        
        Provide a structured evaluation with:
        1. Overall Match Score (0-100%)
        2. Key Strengths (3-5 points)
        3. Potential Gaps (2-3 points)
        4. Final Recommendation (Highly Recommended / Recommended / Not Recommended)
        
        Be specific and cite actual experience/skills from the resume.
        """
        
        print("Generating AI-powered evaluation report with Gemini...")
        print()
        response = query_engine.query(evaluation_prompt)
        
        print("=" * 60)
        print("CANDIDATE EVALUATION REPORT (Powered by Gemini AI)")
        print("=" * 60)
        print(response)
        print()
        
        return str(response)
    
    def evaluate_multiple_candidates(self, index: VectorStoreIndex,
                                    job_description: str,
                                    top_n: int = 3) -> List[str]:
        """
        Generate evaluations for multiple top candidates.
        
        Args:
            index: Vector store index for querying
            job_description: Job description for evaluation
            top_n: Number of candidates to evaluate
            
        Returns:
            List[str]: List of evaluation reports
        """
        print("=" * 60)
        print(f"GENERATING EVALUATIONS FOR TOP {top_n} CANDIDATES")
        print("=" * 60)
        
        # Initialize LLM
        self._initialize_llm()
        
        evaluations = []
        
        # Create query engine
        query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=top_n,
            response_mode="compact"
        )
        
        for i in range(1, top_n + 1):
            print(f"\nEvaluating Candidate #{i}...")
            
            evaluation_prompt = f"""
            Based on the job description below, evaluate candidate #{i} from the retrieved resume data:
            
            Job Description:
            {job_description}
            
            Provide a brief evaluation with:
            1. Overall Match Score (0-100%)
            2. Top 3 Strengths
            3. Top 2 Concerns
            4. Final Recommendation
            
            Keep it concise and specific.
            """
            
            response = query_engine.query(evaluation_prompt)
            evaluations.append(str(response))
            
            print(f"✓ Evaluation #{i} complete")
        
        print()
        print("=" * 60)
        print("ALL EVALUATIONS COMPLETE")
        print("=" * 60)
        
        return evaluations


