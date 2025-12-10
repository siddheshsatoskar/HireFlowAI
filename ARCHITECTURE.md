# HireFlow Architecture Documentation

## Overview

The HireFlow application has been refactored from a procedural approach to a well-organized, class-based architecture. This document describes the new structure and design patterns.

## Architecture Diagram

```
HireFlow/
├── main.py                          # CLI entry point
├── config/
│   └── settings.py                  # Configuration settings
├── src/
│   ├── simple_pipeline_local.py     # Backward compatibility layer
│   └── services/                    # Core service classes
│       ├── __init__.py
│       ├── environment_setup.py     # EnvironmentSetup class
│       ├── document_ingestion.py    # DocumentIngestionManager class
│       ├── vector_store.py          # VectorStoreManager class
│       ├── retrieval.py             # RetrievalManager class
│       ├── reranking.py             # ReRankingManager class
│       ├── evaluation.py            # EvaluationManager class
│       └── chatbot.py               # ChatbotManager class
└── data/                            # Resume data
```

## Service Classes

### 1. EnvironmentSetup (`environment_setup.py`)

**Purpose**: Manages environment configuration and embedding model initialization.

**Key Responsibilities**:
- Loading HuggingFace embedding models
- Configuring global LlamaIndex settings
- Setting chunking parameters

**Key Methods**:
```python
setup()                  # Initialize and configure embedding model
get_embed_model()        # Retrieve configured embedding model
```

**Usage Example**:
```python
env_setup = EnvironmentSetup()
embed_model = env_setup.setup()
```

---

### 2. DocumentIngestionManager (`document_ingestion.py`)

**Purpose**: Handles reading and processing of resume documents.

**Key Responsibilities**:
- Reading individual resume files
- Batch ingestion of multiple resumes
- Document parsing and metadata extraction

**Key Methods**:
```python
read_single_resume(file_path)            # Read one resume
ingest_multiple_resumes(resume_dir)      # Read all resumes from directory
get_documents()                          # Get ingested documents
get_document_count()                     # Get count of documents
```

**Usage Example**:
```python
doc_manager = DocumentIngestionManager()
documents = doc_manager.ingest_multiple_resumes("/path/to/resumes")
```

---

### 3. VectorStoreManager (`vector_store.py`)

**Purpose**: Manages vector store creation, persistence, and loading.

**Key Responsibilities**:
- Creating FAISS vector indices
- Persisting indices to disk
- Loading existing indices
- Managing vector database operations

**Key Methods**:
```python
create_index(documents)                  # Create vector index
save_index()                             # Save index to disk
load_index()                             # Load index from disk
create_and_save_index(documents)         # Create and persist index
get_index()                              # Get current index
```

**Usage Example**:
```python
vector_store_manager = VectorStoreManager()
index = vector_store_manager.create_and_save_index(documents)
```

---

### 4. RetrievalManager (`retrieval.py`)

**Purpose**: Handles semantic search and candidate retrieval.

**Key Responsibilities**:
- Creating retrievers from vector indices
- Performing similarity search
- Retrieving top-k candidates

**Key Methods**:
```python
create_retriever()                       # Create retriever
retrieve(query, top_k)                   # Retrieve candidates
get_retrieved_nodes()                    # Get retrieved results
get_retrieval_count()                    # Get count of results
```

**Usage Example**:
```python
retrieval_manager = RetrievalManager(index, similarity_top_k=5)
retrieved_nodes = retrieval_manager.retrieve(job_description)
```

---

### 5. ReRankingManager (`reranking.py`)

**Purpose**: Re-ranks retrieved candidates based on various criteria.

**Key Responsibilities**:
- Simple re-ranking by similarity scores
- Advanced re-ranking with keyword boosting
- Selecting top N candidates

**Key Methods**:
```python
simple_rerank(nodes, job_desc, top_n)    # Simple re-ranking
advanced_rerank(nodes, job_desc, top_n, keywords)  # Advanced re-ranking
get_top_candidates()                     # Get top candidates
get_candidate_count()                    # Get count of candidates
```

**Usage Example**:
```python
reranking_manager = ReRankingManager()
top_candidates = reranking_manager.simple_rerank(
    retrieved_nodes, job_description, top_n=3
)
```

---

### 6. EvaluationManager (`evaluation.py`)

**Purpose**: Generates candidate evaluations and reports.

**Key Responsibilities**:
- Generating summary reports
- Creating AI-powered detailed evaluations
- Scoring and recommendation generation

**Key Methods**:
```python
generate_summary_report(candidates, job_desc)       # Summary report
generate_detailed_evaluation(index, job_desc)       # AI evaluation
evaluate_multiple_candidates(index, job_desc, n)    # Multiple evaluations
```

**Usage Example**:
```python
evaluation_manager = EvaluationManager(gemini_api_key=API_KEY)
evaluation_manager.generate_summary_report(top_candidates, job_description)
evaluation_manager.generate_detailed_evaluation(index, job_description)
```

---

### 7. ChatbotManager (`chatbot.py`)

**Purpose**: Manages interactive chatbot with conversation memory.

**Key Responsibilities**:
- Creating chat engines with memory
- Managing multi-turn conversations
- Maintaining context across interactions
- Conversational AI for candidate queries

**Key Methods**:
```python
create_chat_engine(job_description)      # Create chat engine
chat(message)                            # Send message and get response
reset_conversation()                     # Reset memory
start_interactive_session(job_desc)      # Start interactive session
```

**Usage Example**:
```python
chatbot_manager = ChatbotManager(index, gemini_api_key=API_KEY)
chatbot_manager.start_interactive_session(job_description)
```

---

## Design Patterns

### Single Responsibility Principle
Each class has a single, well-defined responsibility:
- **EnvironmentSetup**: Configuration only
- **DocumentIngestionManager**: Document processing only
- **VectorStoreManager**: Vector store operations only
- And so on...

### Dependency Injection
Classes accept dependencies through constructors:
```python
# API keys and configurations can be injected
evaluation_manager = EvaluationManager(gemini_api_key=API_KEY)
chatbot_manager = ChatbotManager(index, gemini_api_key=API_KEY)
```

### Encapsulation
Internal state and implementation details are hidden:
- Private methods prefixed with `_` (e.g., `_initialize_llm()`)
- Public interface through well-defined methods
- State management within class boundaries

### Factory Pattern
Classes create and manage their own dependencies:
```python
# VectorStoreManager creates retrievers
retriever = vector_store_manager.create_retriever()

# ChatbotManager creates chat engines
chat_engine = chatbot_manager.create_chat_engine()
```

---

## Pipeline Flow

### Full Pipeline Execution Flow

```
1. EnvironmentSetup.setup()
   ↓ (returns embed_model)

2. DocumentIngestionManager.ingest_multiple_resumes()
   ↓ (returns documents)

3. VectorStoreManager.create_and_save_index(documents)
   ↓ (returns index)

4. RetrievalManager.retrieve(job_description)
   ↓ (returns retrieved_nodes)

5. ReRankingManager.simple_rerank(retrieved_nodes)
   ↓ (returns top_candidates)

6. EvaluationManager.generate_summary_report(top_candidates)
   ↓ (displays report)

7. EvaluationManager.generate_detailed_evaluation(index) [optional]
   ↓ (displays AI evaluation)

8. ChatbotManager.start_interactive_session(job_description) [optional]
   ↓ (interactive Q&A)
```

---

## Backward Compatibility

The `simple_pipeline_local.py` module maintains backward compatibility by providing wrapper functions:

```python
def setup_local_environment():
    env_setup = EnvironmentSetup()
    return env_setup.setup()

def ingest_multiple_resumes(resume_dir: str):
    doc_manager = DocumentIngestionManager()
    return doc_manager.ingest_multiple_resumes(resume_dir)

# ... and so on
```

This allows existing code to continue working without modifications.

---

## Benefits of the New Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Easy to locate and modify specific functionality
- Reduced code duplication

### 2. **Testability**
- Each class can be tested independently
- Easy to mock dependencies
- Clear input/output contracts

### 3. **Reusability**
- Classes can be used in different contexts
- Easy to compose new workflows
- Plug-and-play components

### 4. **Extensibility**
- Add new features without modifying existing code
- Easy to subclass and override behavior
- Clear extension points

### 5. **Readability**
- Self-documenting through class and method names
- Logical grouping of related functionality
- Clear hierarchical structure

---

## Usage Examples

### Example 1: Basic Search Pipeline
```python
from src.services import (
    EnvironmentSetup, 
    DocumentIngestionManager,
    VectorStoreManager,
    RetrievalManager
)

# Setup
env = EnvironmentSetup()
env.setup()

# Ingest
doc_mgr = DocumentIngestionManager()
docs = doc_mgr.ingest_multiple_resumes("./resumes")

# Index
vector_mgr = VectorStoreManager()
index = vector_mgr.create_and_save_index(docs)

# Search
retrieval_mgr = RetrievalManager(index, similarity_top_k=10)
results = retrieval_mgr.retrieve("Looking for Python developer")
```

### Example 2: Custom Evaluation Pipeline
```python
from src.services import (
    VectorStoreManager,
    EvaluationManager
)

# Load existing index
vector_mgr = VectorStoreManager()
index = vector_mgr.load_index()

# Evaluate
eval_mgr = EvaluationManager(gemini_api_key=API_KEY)
evaluations = eval_mgr.evaluate_multiple_candidates(
    index, 
    job_description,
    top_n=5
)
```

### Example 3: Chatbot Only
```python
from src.services import ChatbotManager, VectorStoreManager

# Load index
vector_mgr = VectorStoreManager()
index = vector_mgr.load_index()

# Start chatbot
chatbot = ChatbotManager(index, gemini_api_key=API_KEY)
chatbot.start_interactive_session(job_description)
```

---

## Configuration

Configuration is centralized in `config/settings.py`:

```python
DATA_DIR = Path(__file__).parent.parent / "data" / "Resume_dataset"
TOP_K_CANDIDATES = 5
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

---

## Future Enhancements

Potential areas for extension:

1. **Database Integration**: Replace in-memory FAISS with persistent vector databases (Pinecone, Weaviate, Qdrant)

2. **Advanced Re-ranking**: Implement more sophisticated re-ranking algorithms (cross-encoders, LLM-based re-ranking)

3. **Multi-modal Support**: Add support for images, videos in resumes

4. **Analytics Dashboard**: Track search patterns, popular queries, candidate metrics

5. **API Layer**: Add RESTful API endpoints for programmatic access

6. **Batch Processing**: Add support for processing large batches of resumes asynchronously

7. **Custom Evaluators**: Plugin system for custom evaluation criteria

---

## Contributing

When adding new features:

1. Create a new service class if it represents a distinct responsibility
2. Follow the existing naming conventions
3. Add comprehensive docstrings
4. Update this documentation
5. Add usage examples
6. Ensure backward compatibility when possible

---

## Conclusion

The refactored architecture provides a solid foundation for building and extending the HireFlow application. The class-based design promotes code quality, maintainability, and scalability while maintaining simplicity and ease of use.


