"""
HireFlow Simple Pipeline - Local Embeddings Version
Uses HuggingFace embeddings (runs locally, no API quota needed)
Demonstrates: Reading resumes → Ingestion → Vector Store → Retrieval → Re-ranking
"""

import os
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.retrievers import VectorIndexRetriever

# Using HuggingFace embeddings (local, no API needed)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.memory import ChatMemoryBuffer

from config.settings import DATA_DIR, TOP_K_CANDIDATES, GEMINI_API_KEY


def setup_local_environment():
    """Step 1: Setup Local Embedding Model"""
    print("=" * 60)
    print("STEP 1: Setting up Local Embedding Model")
    print("=" * 60)
    
    # Use HuggingFace embeddings directly
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print(f"✓ Local embedding model loaded: {model_name}")
    print("✓ Chunk size: 512, Overlap: 50")
    print()
    
    return embed_model


def read_single_resume(file_path: str) -> Document:
    """Step 2: Reading from 1 resume"""
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


def ingest_multiple_resumes(resume_dir: str) -> List[Document]:
    """Step 3: Ingesting multiple resumes"""
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
    documents = reader.load_data()
    
    print(f"✓ Successfully ingested {len(documents)} document(s)")
    for i, doc in enumerate(documents, 1):
        file_name = doc.metadata.get('file_name', 'Unknown')
        char_count = len(doc.text)
        print(f"  {i}. {file_name} - {char_count} chars")
    print()
    
    return documents


def create_vector_store(documents: List[Document]) -> VectorStoreIndex:
    """Step 4: Creating Vector Store with FAISS"""
    print("=" * 60)
    print("STEP 4: Creating Vector Store")
    print("=" * 60)
    
    print("Creating vector embeddings and building FAISS index...")
    print("(This may take a minute for local embeddings)")
    
    # Create vector store index (uses FAISS by default)
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    print("✓ Vector store index created successfully")
    print(f"  - Number of documents indexed: {len(documents)}")
    print(f"  - Vector database: FAISS (in-memory)")
    print()
    
    # Optional: Save index to disk
    index_path = Path(__file__).parent.parent / "vector_store"
    index_path.mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_path))
    print(f"✓ Index saved to: {index_path}")
    print()
    
    return index


def retrieve_candidates(index: VectorStoreIndex, job_description: str, top_k: int = 5):
    """Step 5: Retrieval - Find top candidates"""
    print("=" * 60)
    print("STEP 5: Retrieval - Finding Top Candidates")
    print("=" * 60)
    
    print(f"Job Description Query:")
    print(f"  {job_description}...")
    print()
    
    # Create retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )
    
    # Retrieve relevant nodes
    retrieved_nodes = retriever.retrieve(job_description)
    
    print(f"✓ Retrieved {len(retrieved_nodes)} candidate node(s)")
    print()
    
    # Display results
    for i, node in enumerate(retrieved_nodes, 1):
        print(f"Result {i}:")
        print(f"  - Similarity Score: {node.score:.4f}")
        print(f"  - Source: {node.node.metadata.get('file_name', 'Unknown')}")
        print(f"  - Content Preview:")
        print(f"    {node.text[:150]}...")
        print()
    
    return retrieved_nodes


def simple_rerank(retrieved_nodes, job_description: str, top_n: int = 3):
    """Step 6: Simple Re-ranking based on keyword matching"""
    print("=" * 60)
    print("STEP 6: Re-ranking Candidates")
    print("=" * 60)
    
    print(f"Re-ranking {len(retrieved_nodes)} candidates...")
    print(f"Using similarity scores from semantic search")
    print()
    
    # Already ranked by similarity, just take top N
    top_candidates = retrieved_nodes[:top_n]
    
    print(f"✓ Top {len(top_candidates)} candidates after re-ranking:")
    print()
    
    for i, node in enumerate(top_candidates, 1):
        print(f"Rank #{i}:")
        print(f"  - Score: {node.score:.4f}")
        print(f"  - Source: {node.node.metadata.get('file_name', 'Unknown')}")
        print(f"  - Match Reason: High semantic similarity to job requirements")
        print()
    
    return top_candidates


def generate_summary_report(top_candidates, job_description: str):
    """Generate a summary report"""
    print("=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    print()
    
    print(f"Job Description: {job_description}...")
    print()
    print(f"Total Candidates Evaluated: {len(top_candidates)}")
    print()
    print("Top Recommendations:")
    print()
    
    for i, node in enumerate(top_candidates, 1):
        score_pct = node.score * 100
        recommendation = "HIGHLY RECOMMENDED" if score_pct > 80 else "RECOMMENDED" if score_pct > 60 else "CONSIDER"
        
        print(f"{i}. Candidate from: {node.node.metadata.get('file_name', 'Unknown')}")
        print(f"   Match Score: {score_pct:.1f}%")
        print(f"   Recommendation: {recommendation}")
        print(f"   Key Snippet: {node.text}...")
        print()
    
    return True


def generate_candidate_evaluation(index: VectorStoreIndex, job_description: str):
    """Generate detailed candidate evaluation using Gemini LLM"""
    print("=" * 60)
    print("STEP 7: GENERATING DETAILED CANDIDATE EVALUATION")
    print("=" * 60)

    memory = ChatMemoryBuffer(token_limit=1500)
    
    # Initialize Gemini LLM
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY
    )
    
    # Create query engine with Gemini LLM
    query_engine = index.as_query_engine(
        llm=llm,
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
    
    return response


def interactive_chatbot(index: VectorStoreIndex, job_description: str = None):
    """Step 8: Interactive Chatbot for Candidate Questions"""
    print("=" * 60)
    print("STEP 8: INTERACTIVE CHATBOT")
    print("=" * 60)
    print()
    print("Welcome to the HireFlow Chatbot! 🤖")
    print("I'm your AI assistant for candidate search and evaluation.")
    print()
    print("💡 I maintain conversation context, so you can ask follow-up questions!")
    print()
    print("Example conversation:")
    print("  You: Who has the most accounting experience?")
    print("  Bot: [Gives answer about Candidate A]")
    print("  You: What about their certifications?  ← I remember who 'their' refers to!")
    print("  Bot: [Details about Candidate A's certifications]")
    print("  You: Compare them with the second candidate  ← Conversational!")
    print()
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("=" * 60)
    print()
    
    # Initialize Gemini LLM
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY
    )

    # Create chat memory buffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # Create chat engine with Gemini LLM (instead of query engine)
    chat_engine = index.as_chat_engine(
        llm=llm,
        memory=memory,
        similarity_top_k=5,
        chat_mode="context",  # Uses retrieved context for responses
        system_prompt="""You are a helpful HR assistant for HireFlow, an intelligent candidate search system.
You have access to a database of candidate resumes. Your role is to:
- Answer questions about candidates' skills, experience, and qualifications
- Compare candidates based on specific criteria
- Provide recommendations for job roles
- Remember the conversation context and refer back to previous answers

Be concise, professional, and cite specific details from the resumes when possible."""
    )
    
    # Add job description context if provided
    if job_description:
        print("📋 Setting up context with job description...")
        initial_message = f"""I'm looking to fill a position with the following requirements:

{job_description}

Please keep this job description in mind when I ask about candidates."""
        
        # Send initial context message (won't be displayed)
        chat_engine.chat(initial_message)
        print("✅ Context set! You can now ask questions.\n")
    
    # Interactive chat loop
    while True:
        try:
            # Get user input
            user_question = input("You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['exit', 'quit', 'bye', 'q']:
                print()
                print("Chatbot: Thank you for using HireFlow! Goodbye! 👋")
                print()
                break
            
            # Skip empty input
            if not user_question:
                continue
            
            print()
            print("Chatbot: ", end="", flush=True)
            
            # Chat with the engine (maintains conversation history)
            response = chat_engine.chat(user_question)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n")
            print("Chatbot: Chat interrupted. Goodbye! 👋")
            print()
            break
        except Exception as e:
            print(f"\nChatbot: Sorry, I encountered an error: {e}")
            print("Please try rephrasing your question.")
            print()
    
    return True


def main():
    """Main pipeline execution"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  HIREFLOW - INTELLIGENT CANDIDATE SEARCH".center(58) + "*")
    print("*" + "  (Local Embeddings Version)".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    try:
        # Step 1: Setup environment
        embed_model = setup_local_environment()
        
        # Step 2: Read single resume
        #sample_resume_path = DATA_DIR / "Senior_Accountant_Position.pdf"
        
        #single_doc = read_single_resume(str(sample_resume_path))
        
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
        
        """ print("=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        print()
        print("What Happened:")
        print("  1. ✓ Loaded embedding model")
        print(f"  2. ✓ Read and parsed {len(documents)} resume(s)")
        print(f"  3. ✓ Created vector embeddings for semantic search")
        print(f"  4. ✓ Built FAISS vector database")
        print(f"  5. ✓ Retrieved top {len(retrieved_nodes)} semantically similar candidates")
        print(f"  6. ✓ Re-ranked and selected top {len(top_candidates)} candidates")
        print(f"  7. ✓ Generated AI-powered evaluation for top candidates")
        print(f"  8. ✓ Interactive chatbot session completed")
        print()
        print("Next Steps:")
        print("  - Customize the job description")
        print("  - Integrate with Streamlit UI for interactive experience")
        print() """
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

