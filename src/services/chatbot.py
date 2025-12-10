"""
Chatbot Service

Handles interactive chatbot functionality with conversation memory.
"""

from typing import Optional
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.memory import ChatMemoryBuffer


class ChatbotManager:
    """
    Manages interactive chatbot functionality.
    
    This class handles:
    - Creating chat engines with conversation memory
    - Managing multi-turn conversations
    - Maintaining context across interactions
    - Providing conversational AI for candidate queries
    """
    
    def __init__(self, index: VectorStoreIndex, gemini_api_key: Optional[str] = None,
                 token_limit: int = 3000, similarity_top_k: int = 5):
        """
        Initialize the ChatbotManager.
        
        Args:
            index: Vector store index for retrieving information
            gemini_api_key: API key for Google Gemini (optional)
            token_limit: Token limit for conversation memory
            similarity_top_k: Number of similar items to retrieve
        """
        self.index = index
        self.gemini_api_key = gemini_api_key
        self.token_limit = token_limit
        self.similarity_top_k = similarity_top_k
        self.chat_engine = None
        self.llm = None
        self.memory = None
    
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
    
    def create_chat_engine(self, job_description: Optional[str] = None):
        """
        Create a chat engine with conversation memory.
        
        Args:
            job_description: Optional job description for context
        """
        # Initialize LLM
        self._initialize_llm()
        
        # Create chat memory buffer
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=self.token_limit)
        
        # Create chat engine
        self.chat_engine = self.index.as_chat_engine(
            llm=self.llm,
            memory=self.memory,
            similarity_top_k=self.similarity_top_k,
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
            self.chat_engine.chat(initial_message)
            print("✅ Context set! You can now ask questions.\n")
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User message
            
        Returns:
            str: Chatbot response
        """
        if self.chat_engine is None:
            raise RuntimeError("Chat engine not initialized. Call create_chat_engine() first.")
        
        response = self.chat_engine.chat(message)
        return str(response)
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.memory is not None:
            self.memory.reset()
    
    def start_interactive_session(self, job_description: Optional[str] = None):
        """
        Start an interactive chatbot session.
        
        Args:
            job_description: Optional job description for context
        """
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
        
        # Create chat engine
        self.create_chat_engine(job_description)
        
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
                response = self.chat(user_question)
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


