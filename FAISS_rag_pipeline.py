from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from fastembed.embedding import TextEmbedding  
from langchain_core.embeddings import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Dict, List, Optional
import hashlib
import json

load_dotenv()

class FastEmbedLangChainWrapper(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return next(self.model.embed([text]))

class SessionMemoryManager:
    """Manages conversation memory for multiple users using session IDs"""
    
    def __init__(self):
        self.user_memories: Dict[str, ConversationBufferMemory] = {}
    
    def generate_session_id(self, user_profile: dict) -> str:
        """Generate a unique session ID based on user profile"""
        # Create a unique identifier from user profile
        profile_str = f"{user_profile.get('name', '')}{user_profile.get('timestamp', '')}"
        return hashlib.md5(profile_str.encode()).hexdigest()
    
    def get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get existing memory for session or create new one"""
        if session_id not in self.user_memories:
            self.user_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.user_memories[session_id]
    
    def load_chat_history_from_client(self, session_id: str, chat_history: List[Dict]) -> None:
        """Load chat history from client's local storage"""
        if not chat_history:
            return
            
        memory = self.get_or_create_memory(session_id)
        # Clear existing memory first
        memory.clear()
        
        # Add messages from local storage (skip welcome message)
        for message in chat_history:
            if message.get('sender') == 'user':
                memory.chat_memory.add_user_message(message.get('content', ''))
            elif message.get('sender') == 'bot' and 'Hello' not in message.get('content', '')[:50]:  # Skip welcome message
                memory.chat_memory.add_ai_message(message.get('content', ''))
    
    def add_interaction(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Add new interaction to memory"""
        memory = self.get_or_create_memory(session_id)
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_response)
    
    def get_formatted_history(self, session_id: str, max_messages: int = 6) -> str:
        """Get formatted chat history for prompts"""
        if session_id not in self.user_memories:
            return "No previous conversation."
            
        memory = self.user_memories[session_id]
        messages = memory.chat_memory.messages
        
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages[-max_messages:]:  # Last N messages
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted)
    
    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session"""
        if session_id in self.user_memories:
            self.user_memories[session_id].clear()
    
    def cleanup_old_sessions(self, max_sessions: int = 100) -> None:
        """Clean up old sessions to prevent memory bloat"""
        if len(self.user_memories) > max_sessions:
            # Remove oldest sessions (simple FIFO)
            sessions_to_remove = len(self.user_memories) - max_sessions
            for session_id in list(self.user_memories.keys())[:sessions_to_remove]:
                del self.user_memories[session_id]

# Global memory manager instance
memory_manager = SessionMemoryManager()

embedding_model = FastEmbedLangChainWrapper(model_name="BAAI/bge-small-en-v1.5")

# Prompt templates
therapist_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history"], template="""
You are a compassionate clinical psychologist speaking directly with a client.

Client Profile: {user_profile}
Previous Conversation: {chat_history}
Client's Current Question: {question}
Knowledge Base Insights: "{raw_answer}"

Remember this client's information from their profile and previous conversation. You can refer to their name, age, country, and other details naturally in conversation.

If the user provides symptoms, you must:
1. List possible diagnoses in bullet points
2. Include a disclaimer that you are an AI agent, not a professional
3. Ask if they want professional help or want to talk about it
4. If they want to talk, provide gentle and supportive guidance

Respond naturally and directly to the client. Do not use salutations and greetings like hello etc. Avoid saying stuff like of course here is a gentle and supportive reply because then the user will feel that you are not talking to them directly.

Your response:
""")

resource_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history"], template="""
User asked: "{question}"
User Profile: {user_profile}
Previous Conversation: {chat_history}
Resources retrieved: "{raw_answer}"

You are a mental health assistant. Suggest **only country-specific and free (if user is financially struggling or on a limited budget otherwise suggest paid resources too)** support links or helpline numbers.
Keep it short, practical, and clear.
Output only contact options, links, or phone numbers:
""")

classification_prompt = PromptTemplate(input_variables=["question"], template="""
You are an intent classifier for a multi-agent mental health system.
Classify the user input:
- therapist: if the user is asking for a definition, explanation, symptoms, needs emotional support, asks personal questions about themselves, or wants to chat
- resource: if the user is asking for support links, professional help, helpline numbers, or country-specific services

Input: "{question}"
Respond with one word only: therapist or resource
""")

# Load databases
dbTherapy = FAISS.load_local("faiss_therapy_index", embedding_model, allow_dangerous_deserialization=True)
dbResources = FAISS.load_local("faiss_resource_index", embedding_model, allow_dangerous_deserialization=True)

retrieverTherapy = dbTherapy.as_retriever()
retrieverResource = dbResources.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=os.getenv("GEMINI_API_KEY"),
)

def get_therapy_context(question: str) -> str:
    """Get relevant therapy context from vector store"""
    docs = retrieverTherapy.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

def get_resource_context(question: str) -> str:
    """Get relevant resource context from vector store"""
    docs = retrieverResource.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

def classify_agent(user_input: str) -> str:
    prompt = classification_prompt.format(question=user_input)
    response = llm.invoke(prompt)
    result = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()
    return result if result in ["therapist", "resource"] else "therapist"

def therapist_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    """Therapist agent with session-specific memory"""
    raw_answer = get_therapy_context(user_input)
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Country: {user_profile.get('country', 'unknown')}, Financial: {user_profile.get('financial', 'unknown')}, Name: {user_profile.get('name', 'unknown')}, Age: {user_profile.get('age', 'unknown')}"
    
    prompt = therapist_prompt.format(
        question=user_input, 
        raw_answer=raw_answer, 
        user_profile=profile_summary,
        chat_history=chat_history
    )
    
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    # Update memory for this session
    memory_manager.add_interaction(session_id, user_input, response_text)
    
    return response_text

def resource_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    """Resource agent with session-specific memory"""
    raw_answer = get_resource_context(user_input)
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Country: {user_profile.get('country', 'unknown')}, Financial: {user_profile.get('financial', 'unknown')}, Name: {user_profile.get('name', 'unknown')}, Age: {user_profile.get('age', 'unknown')}"
    
    prompt = resource_prompt.format(
        question=user_input, 
        raw_answer=raw_answer, 
        user_profile=profile_summary,
        chat_history=chat_history
    )
    
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    # Update memory for this session
    memory_manager.add_interaction(session_id, user_input, response_text)
    
    return response_text

def multiagent_chain(user_input: str, user_profile: dict, chat_history: List[Dict] = None) -> dict:
    """
    Main entry point for multi-agent chain with session management
    
    Args:
        user_input: Current user message
        user_profile: User profile information
        chat_history: Chat history from client's local storage 
    
    Returns:
        dict with agent and response
    """
    # Generate session ID
    session_id = memory_manager.generate_session_id(user_profile)
    
    # Load chat history from client if provided
    if chat_history:
        memory_manager.load_chat_history_from_client(session_id, chat_history)
    
    # Classify and route to appropriate agent
    agent = classify_agent(user_input)
    
    print(f"[DEBUG] Session ID: {session_id}")
    print(f"[DEBUG] Using agent: {agent}")
    
    if agent == "resource":
        answer = resource_wrapper(user_input, user_profile, session_id)
        agent_name = "Resource Assistant"
    else:
        answer = therapist_wrapper(user_input, user_profile, session_id)
        agent_name = "Therapist"
    
    # Cleanup old sessions periodically
    memory_manager.cleanup_old_sessions()
    
    return {
        "agent": agent_name,
        "response": answer
    }

# Utility functions for session management
def clear_user_session(user_profile: dict):
    """Clear conversation history for a specific user"""
    session_id = memory_manager.generate_session_id(user_profile)
    memory_manager.clear_session(session_id)