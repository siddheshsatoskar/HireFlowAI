#!/usr/bin/env python3
"""
HireFlow CLI - Command Line Interface for Intelligent Candidate Search

Features:
- Interactive job description input
- Semantic search using vector embeddings
- Candidate ranking and evaluation
- Conversational AI chatbot using as_chat_engine()
  * Maintains conversation memory (up to 3000 tokens)
  * Understands follow-up questions and pronouns
  * Natural multi-turn dialogues
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.simple_pipeline_local import (
    setup_local_environment,
    ingest_multiple_resumes,
    create_vector_store,
    retrieve_candidates,
    simple_rerank,
    generate_summary_report,
    generate_candidate_evaluation,
    interactive_chatbot
)
from config.settings import DATA_DIR, TOP_K_CANDIDATES


def print_banner():
    """Print application banner"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  HIREFLOW - INTELLIGENT CANDIDATE SEARCH".center(58) + "*")
    print("*" + "  Command Line Interface".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")


def get_job_description_from_user():
    """Get job description interactively from user"""
    print("=" * 60)
    print("📝 JOB DESCRIPTION INPUT")
    print("=" * 60)
    print()
    print("Please enter the job description.")
    print("You can enter multiple lines. When done, press Enter twice (empty line).")
    print("Or type 'default' to use the default job description.")
    print()
    print("Start typing:")
    print("-" * 60)
    
    lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
            
            # Check if user wants default
            if line.strip().lower() == 'default' and not lines:
                return """
                We are looking for a Senior Accountant with 5+ years of experience in 
                financial reporting, tax preparation, and audit support. The ideal candidate 
                should have strong knowledge of GAAP, proficiency in accounting software 
                (QuickBooks, SAP), and excellent analytical skills. CPA certification is required.
                """
            
            if not line.strip():
                empty_line_count += 1
                if empty_line_count >= 2:  # Two consecutive empty lines
                    break
            else:
                empty_line_count = 0
                lines.append(line)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Input cancelled. Using default job description.")
            return """
            We are looking for a Senior Accountant with 5+ years of experience in 
            financial reporting, tax preparation, and audit support. The ideal candidate 
            should have strong knowledge of GAAP, proficiency in accounting software 
            (QuickBooks, SAP), and excellent analytical skills. CPA certification is required.
            """
    
    job_description = "\n".join(lines).strip()
    
    if not job_description:
        print("\n⚠️  No job description entered. Using default.")
        return """
        We are looking for a Senior Accountant with 5+ years of experience in 
        financial reporting, tax preparation, and audit support. The ideal candidate 
        should have strong knowledge of GAAP, proficiency in accounting software 
        (QuickBooks, SAP), and excellent analytical skills. CPA certification is required.
        """
    
    print("-" * 60)
    print("✅ Job description captured!")
    print()
    
    return job_description


def run_full_pipeline(args):
    """
    Run the complete HireFlow pipeline.
    
    Includes:
    1. Environment setup with embeddings
    2. Resume ingestion
    3. Vector store creation
    4. Candidate retrieval and ranking
    5. Summary report generation
    6. Optional AI evaluation
    7. Interactive chatbot (uses as_chat_engine with conversation memory)
    """
    print_banner()
    
    try:
        # Step 1: Setup environment
        print("🔧 Setting up environment...")
        embed_model = setup_local_environment()
        
        # Step 2: Ingest resumes
        resume_dir = args.resume_dir if args.resume_dir else str(DATA_DIR)
        print(f"📁 Ingesting resumes from: {resume_dir}")
        documents = ingest_multiple_resumes(resume_dir)
        
        if not documents:
            print("❌ No documents found. Please check the resume directory.")
            return
        
        # Step 3: Create vector store
        print("🔍 Creating vector store...")
        index = create_vector_store(documents)
        
        # Step 4: Get job description
        if args.job_description:
            job_description = args.job_description
        elif args.job_file:
            with open(args.job_file, 'r') as f:
                job_description = f.read()
        else:
            # Get job description from user input
            job_description = get_job_description_from_user()
        
        print("\n📋 Job Description:")
        print(f"{job_description[:200]}...")
        print()
        
        # Step 5: Retrieve candidates
        top_k = args.top_k if args.top_k else TOP_K_CANDIDATES
        retrieved_nodes = retrieve_candidates(index, job_description, top_k=top_k)
        
        # Step 6: Re-rank candidates
        top_n = args.top_n if args.top_n else 3
        top_candidates = simple_rerank(retrieved_nodes, job_description, top_n=top_n)
        
        # Step 7: Generate summary report
        if not args.skip_report:
            generate_summary_report(top_candidates, job_description)
        
        # Step 8: Generate detailed evaluation
        if args.detailed_evaluation:
            generate_candidate_evaluation(index, job_description)
        
        # Step 9: Interactive chatbot (uses as_chat_engine with conversation memory)
        # Maintains context across questions, understands follow-ups and pronouns
        if args.interactive:
            interactive_chatbot(index, job_description)
        
        print("=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_chatbot_only(args):
    """
    Run only the chatbot with existing or new index.
    
    Uses as_chat_engine() for conversational AI with memory.
    Maintains conversation history and understands follow-up questions.
    """
    print_banner()
    
    try:
        # Setup environment
        print("🔧 Setting up environment...")
        embed_model = setup_local_environment()
        
        # Check if we should load existing index or create new one
        resume_dir = args.resume_dir if args.resume_dir else str(DATA_DIR)
        
        print(f"📁 Loading resumes from: {resume_dir}")
        documents = ingest_multiple_resumes(resume_dir)
        
        if not documents:
            print("❌ No documents found. Please check the resume directory.")
            return
        
        print("🔍 Creating vector store...")
        index = create_vector_store(documents)
        
        # Get job description if provided (optional for chatbot mode)
        job_description = None
        if args.job_description:
            job_description = args.job_description
        elif args.job_file:
            with open(args.job_file, 'r') as f:
                job_description = f.read()
        elif args.ask_job_description:
            job_description = get_job_description_from_user()
        
        # Start chatbot (uses as_chat_engine for conversational AI)
        # Remembers conversation history and handles follow-up questions naturally
        interactive_chatbot(index, job_description)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_search_only(args):
    """Run only the search and ranking without chatbot"""
    print_banner()
    
    try:
        # Setup environment
        print("🔧 Setting up environment...")
        embed_model = setup_local_environment()
        
        # Ingest resumes
        resume_dir = args.resume_dir if args.resume_dir else str(DATA_DIR)
        print(f"📁 Ingesting resumes from: {resume_dir}")
        documents = ingest_multiple_resumes(resume_dir)
        
        if not documents:
            print("❌ No documents found. Please check the resume directory.")
            return
        
        # Create vector store
        print("🔍 Creating vector store...")
        index = create_vector_store(documents)
        
        # Get job description (required for search mode)
        if args.job_description:
            job_description = args.job_description
        elif args.job_file:
            with open(args.job_file, 'r') as f:
                job_description = f.read()
        else:
            # Prompt user for job description
            job_description = get_job_description_from_user()
        
        # Retrieve and rank
        top_k = args.top_k if args.top_k else TOP_K_CANDIDATES
        retrieved_nodes = retrieve_candidates(index, job_description, top_k=top_k)
        
        top_n = args.top_n if args.top_n else 3
        top_candidates = simple_rerank(retrieved_nodes, job_description, top_n=top_n)
        
        # Generate report
        generate_summary_report(top_candidates, job_description)
        
        print("✅ Search complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HireFlow - Intelligent Candidate Search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline - will prompt for job description
  # Includes conversational chatbot with memory at the end
  python3 main.py

  # Run with custom job description (inline)
  python3 main.py --job-description "Looking for Python developer with 3+ years experience"

  # Run with job description from file
  python3 main.py --job-file job_desc.txt

  # Run only conversational chatbot (remembers context, handles follow-ups)
  python3 main.py --mode chatbot --ask-job-description

  # Run search only - will prompt for job description if not provided
  python3 main.py --mode search

  # Custom resume directory and top candidates
  python3 main.py --resume-dir ./my_resumes --top-k 10 --top-n 5

  # Skip interactive chatbot in full pipeline
  python3 main.py --no-interactive
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['full', 'chatbot', 'search'],
        default='full',
        help='Execution mode: full pipeline, chatbot only (conversational with memory), or search only (default: full)'
    )
    
    # Input options
    parser.add_argument(
        '--resume-dir',
        type=str,
        help=f'Directory containing resumes (default: {DATA_DIR})'
    )
    
    parser.add_argument(
        '--job-description',
        '-j',
        type=str,
        help='Job description text'
    )
    
    parser.add_argument(
        '--job-file',
        '-f',
        type=str,
        help='Path to file containing job description'
    )
    
    parser.add_argument(
        '--ask-job-description',
        action='store_true',
        help='Prompt for job description interactively (useful for chatbot mode)'
    )
    
    # Search parameters
    parser.add_argument(
        '--top-k',
        type=int,
        help=f'Number of candidates to retrieve (default: {TOP_K_CANDIDATES})'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        help='Number of top candidates to show after re-ranking (default: 3)'
    )
    
    # Feature flags
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        default=True,
        help='Enable interactive chatbot with conversation memory (default: enabled in full mode)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_false',
        dest='interactive',
        help='Disable interactive chatbot'
    )
    
    parser.add_argument(
        '--detailed-evaluation',
        '-e',
        action='store_true',
        help='Generate detailed AI evaluation report'
    )
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip summary report generation'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output and error traces'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function based on mode
    if args.mode == 'full':
        run_full_pipeline(args)
    elif args.mode == 'chatbot':
        run_chatbot_only(args)
    elif args.mode == 'search':
        run_search_only(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

