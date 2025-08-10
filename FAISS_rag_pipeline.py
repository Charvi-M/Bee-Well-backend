from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from fastembed.embedding import TextEmbedding  
from langchain_core.embeddings import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Dict, List
import time

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
        self.session_metadata: Dict[str, Dict] = {}
        self.session_user_mapping: Dict[str, str] = {}
    
    def create_user_identifier(self, user_profile: dict) -> str:
        """Create a unique user identifier from profile"""
        name = user_profile.get('name', 'unknown')
        timestamp = user_profile.get('timestamp', str(time.time()))
        return f"{name}_{timestamp}"
    
    def get_or_create_memory(self, session_id: str, user_profile: dict) -> ConversationBufferMemory:
        """Get existing memory for session or create new one"""
        if session_id not in self.user_memories:
            self.user_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.session_metadata[session_id] = user_profile.copy()
            user_id = self.create_user_identifier(user_profile)
            self.session_user_mapping[session_id] = user_id
            print(f"[DEBUG] Created new memory for session {session_id}, user: {user_id}")
        return self.user_memories[session_id]
    
    def load_chat_history_from_client(self, session_id: str, chat_history: List[Dict], user_profile: dict) -> None:
        """Load chat history from client's local storage, excluding welcome messages"""
        if not chat_history:
            return
            
        memory = self.get_or_create_memory(session_id, user_profile)
        
        non_welcome_messages = []
        for message in chat_history:
            content = message.get('content', '')
            sender = message.get('sender', '')
            
            if sender == 'bot' and any(greeting in content.lower() for greeting in 
                ['hello', 'hi', 'welcome', 'i\'m bee', 'mental health companion', 'how are you feeling today']):
                continue
                
            non_welcome_messages.append(message)
        
        if non_welcome_messages:
            memory.clear()
            
            for message in non_welcome_messages:
                content = message.get('content', '')
                sender = message.get('sender', '')
                
                if sender == 'user':
                    memory.chat_memory.add_user_message(content)
                elif sender == 'bot':
                    memory.chat_memory.add_ai_message(content)
            
            print(f"[DEBUG] Loaded {len(non_welcome_messages)} non-welcome messages for session {session_id}")
    
    def add_interaction(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Add new interaction to memory"""
        if session_id in self.user_memories:
            memory = self.user_memories[session_id]
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(ai_response)
    
    def get_formatted_history(self, session_id: str, max_messages: int = 8) -> str:
        """Get formatted chat history for prompts with more context"""
        if session_id not in self.user_memories:
            return "No previous conversation."
            
        memory = self.user_memories[session_id]
        messages = memory.chat_memory.messages
        
        if not messages:
            return "No previous conversation."
        
        formatted = []
        # Get last N messages but ensure we have pairs
        recent_messages = messages[-max_messages:]
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted) if formatted else "No previous conversation."
    
    def get_user_profile(self, session_id: str) -> Dict:
        """Get user profile for a specific session"""
        return self.session_metadata.get(session_id, {})
    
    
    def get_user_name(self, session_id: str) -> str:
        """Get the user name for this session"""
        profile = self.get_user_profile(session_id)
        return profile.get('name', 'Unknown')
    
    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session"""
        if session_id in self.user_memories:
            self.user_memories[session_id].clear()
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        if session_id in self.session_user_mapping:
            del self.session_user_mapping[session_id]
    
    def cleanup_old_sessions(self, max_sessions: int = 100) -> None:
        """Clean up old sessions to prevent memory bloat"""
        if len(self.user_memories) > max_sessions:
            # Remove oldest sessions (simple FIFO)
            sessions_to_remove = len(self.user_memories) - max_sessions
            for session_id in list(self.user_memories.keys())[:sessions_to_remove]:
                self.clear_session(session_id)
            print(f"[DEBUG] Cleaned up {sessions_to_remove} old sessions")

# Global memory manager instance
memory_manager = SessionMemoryManager()

embedding_model = FastEmbedLangChainWrapper(model_name="BAAI/bge-small-en-v1.5")

# Updated prompt templates - Enhanced to handle personal questions naturally
therapist_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history", "session_id"], template="""
You are a compassionate clinical psychologist speaking directly with a client.

Client Profile: {user_profile}
Session ID: {session_id}
Previous Conversation: {chat_history}
Client's Current Question: {question}
Knowledge Base Information: "{raw_answer}"

IMPORTANT: You have access to this client's profile information and conversation history. Use this information to answer personal questions naturally.
You are talking to multiple clients at once so do not mix up the information and conversation history of two different clients.
Personal Questions Guidelines:
- If asked about their name/ age/ country etc. : Use information from the Client Profile
- If asked about their last message/what they said before: Check the Previous Conversation history.
Remember to refer the conversation history to sound coherent and to tell client that you are paying attention.
For Mental Health Questions:
If the user provides symptoms, you must:
1. Give a clear list of all possible disorders having those symptoms in bullet points.
2. Specify which among these could be the most probable diagnosis.
3. Include a disclaimer that you are an AI agent, not a professional.
4. Ask if they want professional help or want to talk about it
5. If they want to talk, provide gentle and supportive guidance.

INSTRUCTIONS:
1. For personal questions (name, last message, profile info): Answer directly using the provided context above
2. For mental health questions: You MUST respond based on information in the "Knowledge Base Information" 
3. If the Knowledge Base Information is empty for mental health questions, respond with: "I don't have specific information about that in my knowledge base. Could you ask something else related to mental health that I might be able to help with?"
4. NEVER make up information
5. Use the client's name from their profile naturally in conversation
6. Reference their previous messages when relevant and appropriate

Be gentle, friendly, empathetic and compassionate. Avoid using greetings like hello etc. after the first greeting.
Your response:
""")

resource_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history", "session_id"], template="""
User asked: "{question}"
User Profile: {user_profile}
Session ID: {session_id}
Previous Conversation: {chat_history}
Resources retrieved: "{raw_answer}"

You are a mental health assistant. You have access to the user's profile and conversation history.
You are talking to multiple clients at once so do not mix up the information and conversation history of two different clients.
Personal Questions: If the user asks personal questions (name, last message, profile info), answer using the information provided above.

For Resource Questions: Suggest *only country-specific and free (if user is financially struggling or on a limited budget otherwise suggest paid resources too)* support links or helpline numbers.
Keep it short, practical, and clear.

INSTRUCTIONS:
1. For personal questions: Answer using User Profile and Previous Conversation information
2. For resource requests: You MUST base your response on the resources retrieved above
3. If no relevant resources are found, respond with: "I don't have specific resource information for your request. Could you try asking for mental health resources in a different way?"
4. NEVER provide generic advice or make up resources
5. Prioritize free resources if user has financial constraints

Keep responses practical and country-specific when available.
Your response:
""")

classification_prompt = PromptTemplate(input_variables=["question"], template="""
You are an intent classifier for a multi-agent mental health system.
Classify the user input:
- therapist: if the user is asking for a definition, explanation, symptoms, needs emotional support, asks personal questions about themselves or family, wants to chat, asks about their name, last message, or profile information
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
    try:
        docs = retrieverTherapy.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context.strip()
    except Exception as e:
        print(f"Error retrieving therapy context: {e}")
        return ""

def get_resource_context(question: str) -> str:
    """Get relevant resource context from vector store"""
    try:
        docs = retrieverResource.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context.strip()
    except Exception as e:
        print(f"Error retrieving resource context: {e}")
        return ""

def classify_agent(user_input: str) -> str:
    """Classify user input to determine appropriate agent"""
    try:
        prompt = classification_prompt.format(question=user_input)
        response = llm.invoke(prompt)
        result = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()
        return result if result in ["therapist", "resource"] else "therapist"
    except Exception as e:
        print(f"Error in classification: {e}")
        return "therapist"

def is_personal_question(user_input: str) -> bool:
    """Check if the question is personal and doesn't need knowledge base lookup"""
    user_input_lower = user_input.lower()
    personal_indicators = [
        "what is my name", "my name", "what's my name", "who am i",
        "last message", "previous message", "what did i say", "my last question",
        "my age", "how old am i", "my country", "where am i from",
        "what did we talk about", "what did i tell you", "remember when i"
    ]
    
    return any(indicator in user_input_lower for indicator in personal_indicators)

def therapist_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Name: {user_profile.get('name', 'Unknown')}, Age: {user_profile.get('age', 'Unknown')}, Country: {user_profile.get('country', 'Unknown')}, Financial Status: {user_profile.get('financial', 'Unknown')}"
    
    # Check if it's a personal question
    if is_personal_question(user_input):
        # For personal questions, we don't need knowledge base - LLM uses context
        raw_answer = ""  
    else:
        # For mental health questions, get knowledge base context
        raw_answer = get_therapy_context(user_input)
        
        if not raw_answer:
            fallback_response = "I don't have specific information about that in my knowledge base. Could you ask something else related to mental health that I might be able to help with?"
            memory_manager.add_interaction(session_id, user_input, fallback_response)
            return fallback_response
    
    try:
        prompt = therapist_prompt.format(
            question=user_input, 
            raw_answer=raw_answer, 
            user_profile=profile_summary,
            chat_history=chat_history,
            session_id=session_id
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Update memory for this session
        memory_manager.add_interaction(session_id, user_input, response_text)
        return response_text
        
    except Exception as e:
        print(f"Error in therapist wrapper: {e}")
        fallback_response = "I'm experiencing some technical difficulties. Please try again in a moment."
        memory_manager.add_interaction(session_id, user_input, fallback_response)
        return fallback_response

def resource_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    """Resource agent with session-specific memory - LLM handles personal questions"""
    
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Name: {user_profile.get('name', 'Unknown')}, Age: {user_profile.get('age', 'Unknown')}, Country: {user_profile.get('country', 'Unknown')}, Financial Status: {user_profile.get('financial', 'Unknown')}"
    
    # Check if it's a personal question
    if is_personal_question(user_input):
        # For personal questions, we don't need resource knowledge base
        raw_answer = ""  
    else:
        # For resource questions, get resource context
        raw_answer = get_resource_context(user_input)
        
        # Strict document adherence check for resource questions only
        if not raw_answer:
            fallback_response = "I don't have specific resource information for your request. Could you try asking for mental health resources in a different way?"
            memory_manager.add_interaction(session_id, user_input, fallback_response)
            return fallback_response
    
    try:
        prompt = resource_prompt.format(
            question=user_input, 
            raw_answer=raw_answer, 
            user_profile=profile_summary,
            chat_history=chat_history,
            session_id=session_id
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Update memory for this session
        memory_manager.add_interaction(session_id, user_input, response_text)
        return response_text
        
    except Exception as e:
        print(f"Error in resource wrapper: {e}")
        fallback_response = "I'm experiencing some technical difficulties with resource lookup. Please try again."
        memory_manager.add_interaction(session_id, user_input, fallback_response)
        return fallback_response

def multiagent_chain(user_input: str, user_profile: dict, chat_history: List[Dict] = None, session_id: str = None) -> dict:
    """
    Main entry point for multi-agent chain with session management
    
    Args:
        user_input: Current user message
        user_profile: User profile information  
        chat_history: Chat history from client's local storage
        session_id: Session ID from client (should be consistent per user)
    
    Returns:
        dict with agent, response, and session_id
    """
    
    # Use the provided session_id (from client's localStorage)
    if not session_id:
        print("[ERROR] No session ID provided by client")
        return {
            "agent": "System",
            "response": "Session error. Please refresh and try again.",
            "session_id": None
        }
    
    print(f"[DEBUG] Processing request for session ID: {session_id}")
    print(f"[DEBUG] User: {user_profile.get('name', 'Unknown')}")
    print(f"[DEBUG] Question type: {'Personal' if is_personal_question(user_input) else 'Mental Health/Resource'}")
    
    # Load chat history from client if provided
    if chat_history:
        memory_manager.load_chat_history_from_client(session_id, chat_history, user_profile)
        print(f"[DEBUG] Loaded {len(chat_history)} messages from client history")
    
    # Ensure user profile is stored for this session
    memory_manager.get_or_create_memory(session_id, user_profile)
    
    # Classify and route to appropriate agent
    agent = classify_agent(user_input)
    print(f"[DEBUG] Classified as: {agent}")
    
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
        "response": answer,
        "session_id": session_id  # Return the same session_id
    }

# Utility functions for session management
def clear_user_session(session_id: str):
    """Clear conversation history for a specific session"""
    memory_manager.clear_session(session_id)

def get_session_info(session_id: str):
    """Get information about a session (for debugging)"""
    if session_id in memory_manager.user_memories:
        memory = memory_manager.user_memories[session_id]
        profile = memory_manager.get_user_profile(session_id)
        user_id = memory_manager.session_user_mapping.get(session_id, "Unknown")
        return {
            "session_exists": True,
            "message_count": len(memory.chat_memory.messages),
            "user_profile": profile,
            "user_identifier": user_id
        }
    return {"session_exists": False}