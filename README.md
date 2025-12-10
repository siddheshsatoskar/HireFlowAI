# HireFlow - Intelligent Candidate Search and Evaluation

An AI-powered recruitment system that combines semantic search with Generative AI to intelligently match candidates to job descriptions, providing confidence scoring and explainable insights for data-driven hiring decisions.

## 🎯 Overview

HireFlow addresses the limitations of traditional Applicant Tracking Systems (ATS) by going beyond simple keyword-based filtering. By understanding the meaning and context of both resumes and job postings, HireFlow ranks candidates based on confidence scoring and provides human-readable insights into their strengths, weaknesses, and overall fit.

## 🚀 Features

- **Semantic Search**: Understands the meaning and context beyond keyword matching
- **AI-Powered Candidate Ranking**: Uses Generative AI to evaluate and rank candidates
- **Confidence Scoring**: Provides quantitative match scores for each candidate
- **Explainable Insights**: Generates human-readable explanations for candidate rankings
- **Interactive Chatbot**: Ask natural language questions about candidates (Step 8)
- **Conversation Memory**: Chatbot remembers context for follow-up questions
- **Modular Architecture**: Scalable and maintainable system design
- **Interactive UI**: Streamlit-based interface for recruiters
- **Vector Database Integration**: Fast and efficient similarity search using FAISS/Pinecone

## 🎯 Problem Statement

### Real-World Challenge

Recruiters and HR teams face multiple inefficiencies in candidate selection:

- **Rigid keyword matching**: Overlooks semantically relevant candidates who may use different terminology
- **Time-consuming evaluation**: Reviewing dozens or hundreds of resumes manually delays hiring
- **Lack of qualitative insights**: Many systems provide scores but no clear reasoning behind candidate ranking
- **No adaptability**: Traditional ATS tools struggle to adjust for role-specific priorities or evolving job requirements

### Impact

This leads to:
- Missed high-potential candidates
- Inflated screening times
- Unclear hiring rationale
- Higher recruitment costs

### Solution Benefits

HireFlow enables organizations to:
- ✅ Shorten hiring cycles
- ✅ Improve match quality and retention
- ✅ Reduce recruiter workload
- ✅ Support data-backed hiring decisions

## 🛠️ Technical Stack

- **Python**: 3.12+
- **LLM & Embeddings**: Gemini API
- **Vector Database**: FAISS / Pinecone
- **Data Processing**: Pandas
- **UI Framework**: Streamlit
- **Data Validation**: Pydantic
- **Configuration**: Python-dotenv
- **Document Processing**: PDF parsing for resumes

## 📋 Requirements

### Core Dependencies

```
python>=3.12
google-generativeai
faiss-cpu  # or faiss-gpu for GPU support
pinecone-client
pandas
streamlit
pydantic
python-dotenv
PyPDF2  # or pdfplumber for PDF processing
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd week11day2-HireFlow
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here  # if using Pinecone
```

## 🚀 Usage

### Running the Complete Pipeline

Run the full pipeline with all 8 steps (including chatbot):

```bash
python3 main.py
```

This will:
1. Load embeddings
2. Process resumes
3. Create vector store
4. Retrieve candidates
5. Rank candidates
6. Generate evaluation
7. **Launch interactive chatbot** for Q&A

### Running Different Modes

```bash
# Search only mode
python3 main.py --mode search

# Chatbot only mode
python3 main.py --mode chatbot

# With custom job description
python3 main.py --job-description "Looking for Python developer with 3+ years experience"

# With job description from file
python3 main.py --job-file job_description.txt
```

### Running the Web Application

```bash
streamlit run app.py
```

### Basic Workflow

1. **Upload Resume Dataset**: Load candidate resumes (PDF format)
2. **Enter Job Description**: Paste or type the job requirements
3. **Process & Match**: System semantically matches candidates to the JD
4. **Review Results**: View ranked candidates with confidence scores and explanations
5. **Ask Questions**: Use the interactive chatbot to explore candidate details (Step 8)
6. **Make Decision**: Use insights to shortlist candidates

### Chatbot Example Questions

- "Who are the top 3 candidates for this role?"
- "What experience does the best candidate have?"
- "Find candidates with Python and data analysis skills"
- "Compare the strengths of the top 2 candidates"

See [CHATBOT_USAGE.md](CHATBOT_USAGE.md) for detailed chatbot documentation.

## 📁 Project Structure

```
week11day2-HireFlow/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── app.py                      # Main Streamlit application
├── data/
│   └── Senior_Accountant_Position.pdf
├── main.py                    # CLI entry point
├── src/
│   ├── __init__.py
│   └── services/              # Service classes
│       ├── __init__.py
│       ├── environment_setup.py
│       ├── document_ingestion.py
│       ├── vector_store.py
│       ├── retrieval.py
│       ├── reranking.py
│       ├── evaluation.py
│       └── chatbot.py
├── config/
│   └── settings.py            # Configuration management
├── tests/
│   ├── __init__.py
│   └── test_matcher.py
└── notebooks/
    └── exploration.ipynb      # Data exploration and testing
```

## 🏗️ System Architecture

### RAG Pipeline

1. **Document Processing**: Parse and extract text from PDF resumes
2. **Embedding Generation**: Create vector embeddings using Gemini API
3. **Vector Storage**: Store embeddings in FAISS/Pinecone
4. **Semantic Search**: Query vector database with job description
5. **Candidate Ranking**: Rank candidates based on similarity scores
6. **Evaluation**: Generate explainable insights using LLM

### Key Components

- **Embedding Layer**: Converts text to semantic vectors
- **Retrieval Layer**: Fast similarity search using vector database
- **Generation Layer**: LLM-powered candidate evaluation
- **Presentation Layer**: Interactive Streamlit UI

## 📊 Data

### Resume Dataset
- Format: PDF
- Structure: Parsed into structured text + metadata
- Content: Candidate information, experience, skills, education

### Job Descriptions
- Multiple domains for testing semantic matching
- Structured format with requirements, responsibilities, qualifications

**Note**: All data in this dataset are synthetically generated for research and testing purposes. Any similarity to actual persons, companies, or events is entirely coincidental and unintended.

## 🎓 Educational Context

This project offers learners the opportunity to:

- Build a RAG (Retrieval-Augmented Generation) pipeline for semantic document matching
- Apply embedding models for retrieval tasks
- Use prompt engineering to generate explainable candidate evaluations
- Design modular and scalable AI systems for enterprise use
- Implement evaluation metrics for relevance and interpretability

## 🔍 Evaluation Metrics

- **Relevance Score**: Semantic similarity between candidate and JD
- **Confidence Score**: Overall match confidence
- **Qualitative Analysis**: Strengths, weaknesses, and fit explanation
- **Ranking Accuracy**: Comparison against human evaluations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational purposes. Please ensure compliance with your organization's policies when using with actual candidate data.

## 🔐 Security & Privacy

- Ensure GDPR compliance when handling candidate data
- Store API keys securely in environment variables
- Do not commit sensitive information to version control
- Consider data anonymization for testing purposes

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for smarter, faster, and fairer hiring**

