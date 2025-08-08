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
import uuid
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
        self.session_metadata: Dict[str, Dict] = {}  #Store user profile per session
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID using UUID and timestamp"""
        return f"{uuid.uuid4().hex[:12]}_{int(time.time())}"
    
    def get_or_create_memory(self, session_id: str, user_profile: dict) -> ConversationBufferMemory:
        """Get existing memory for session or create new one"""
        if session_id not in self.user_memories:
            self.user_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            #Store user profile for this session
            self.session_metadata[session_id] = user_profile.copy()
        return self.user_memories[session_id]
    
    def load_chat_history_from_client(self, session_id: str, chat_history: List[Dict], user_profile: dict) -> None:
        """Load chat history from client's local storage"""
        if not chat_history:
            return
            
        memory = self.get_or_create_memory(session_id, user_profile)
        #Clear existing memory first
        memory.clear()
        
        #Add messages from local storage
        for message in chat_history:
            #Skip system messages and welcome messages
            content = message.get('content', '')
            sender = message.get('sender', '')
            
            #Skip welcome messages (contain greeting patterns)
            if sender == 'bot' and any(greeting in content.lower() for greeting in ['hello', 'hi', 'welcome', 'i\'m bee']):
                continue
                
            if sender == 'user':
                memory.chat_memory.add_user_message(content)
            elif sender == 'bot':
                memory.chat_memory.add_ai_message(content)
    
    def add_interaction(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Add new interaction to memory"""
        if session_id in self.user_memories:
            memory = self.user_memories[session_id]
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
        for msg in messages[-max_messages:]:  #Last N messages
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted)
    
    def get_user_profile(self, session_id: str) -> Dict:
        """Get user profile for a specific session"""
        return self.session_metadata.get(session_id, {})
    
    def get_last_user_message(self, session_id: str) -> str:
        """Get the last user message from this session"""
        if session_id not in self.user_memories:
            return "No previous messages found."
            
        memory = self.user_memories[session_id]
        messages = memory.chat_memory.messages
        
        #Find the last user message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        
        return "No previous messages found."
    
    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session"""
        if session_id in self.user_memories:
            self.user_memories[session_id].clear()
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
    
    def cleanup_old_sessions(self, max_sessions: int = 100) -> None:
        """Clean up old sessions to prevent memory bloat"""
        if len(self.user_memories) > max_sessions:
            #Remove oldest sessions (simple FIFO)
            sessions_to_remove = len(self.user_memories) - max_sessions
            for session_id in list(self.user_memories.keys())[:sessions_to_remove]:
                if session_id in self.user_memories:
                    del self.user_memories[session_id]
                if session_id in self.session_metadata:
                    del self.session_metadata[session_id]

#Global memory manager instance
memory_manager = SessionMemoryManager()

embedding_model = FastEmbedLangChainWrapper(model_name="BAAI/bge-small-en-v1.5")

#Updated prompt templates with stricter document adherence
therapist_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history"], template="""
You are a compassionate clinical psychologist speaking directly with a client.

Client Profile: {user_profile}
Previous Conversation: {chat_history}
Client's Current Question: {question}
Knowledge Base Information: "{raw_answer}"

Remember this client's information from their profile. You can refer to their name, age, country, and other details naturally in conversation.
                                  
If the user provides symptoms, you must:
1. List all possible diagnoses in bullet points
2. Include a disclaimer that you are an AI agent, not a professional
3. Ask if they want professional help or want to talk about it
4. If they want to talk, provide gentle and supportive guidance.

INSTRUCTIONS:
1. You MUST respond based on information in the "Knowledge Base Information" above
2. If the Knowledge Base Information is empty respond with: "I don't have specific information about that in my knowledge base. Could you ask something else related to mental health that I might be able to help with?"
3. NEVER make up information.
4. Use the client's name from their profile naturally in conversation
5. Reference their previous messages when relevant and appropriate

Special cases you CAN answer from context:
- If asked about their name, use the name from their profile
- If asked about their last message, refer to the previous conversation history
- If asked about personal information, use only what's in their profile

If the knowledge base has relevant mental health information, provide a helpful response based ONLY on that information. Avoid using greetings like hello etc. or saying stuff like of course here's a gentle response. 
Be gentle, friendly empathetic and compassionate.
Your response:
""")

resource_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile", "chat_history"], template="""
User asked: "{question}"
User Profile: {user_profile}
Previous Conversation: {chat_history}
Resources retrieved: "{raw_answer}"

You are a mental health assistant. Suggest *only country-specific and free (if user is financially struggling or on a limited budget otherwise suggest paid resources too)* support links or helpline numbers.
Keep it short, practical, and clear.
Output only contact options, links, or phone numbers:                                 

INSTRUCTIONS:
1. You MUST base your response on the resources retrieved above
2. If no relevant resources are found or retrieved information is insufficient, respond with: "I don't have specific resource information for your request. Could you try asking for mental health resources in a different way?"
3. NEVER provide generic advice or make up resources
4. Prioritize free resources if user has financial constraints.

Keep responses practical and country-specific when available.

Your response:
""")

classification_prompt = PromptTemplate(input_variables=["question"], template="""
You are an intent classifier for a multi-agent mental health system.
Classify the user input:
- therapist: if the user is asking for a definition, explanation, symptoms, needs emotional support, asks personal questions about themselves or family, or wants to chat
- resource: if the user is asking for support links, professional help, helpline numbers, or country-specific services

Input: "{question}"

Respond with one word only: therapist or resource
""")

#Load databases
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


def therapist_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    
    #Get context from documents
    raw_answer = get_therapy_context(user_input)
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Name: {user_profile.get('name', 'Unknown')}, Age: {user_profile.get('age', 'Unknown')}, Country: {user_profile.get('country', 'Unknown')}, Financial Status: {user_profile.get('financial', 'Unknown')}"
    
    #Strict document adherence check
    if not raw_answer :
        fallback_response = "I don't have specific information about that in my knowledge base. Could you ask something else related to mental health that I might be able to help with?"
        memory_manager.add_interaction(session_id, user_input, fallback_response)
        return fallback_response
    
    try:
        prompt = therapist_prompt.format(
            question=user_input, 
            raw_answer=raw_answer, 
            user_profile=profile_summary,
            chat_history=chat_history
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        #Double-check if LLM followed instructions
        if "I don't have specific information" in response_text:
            memory_manager.add_interaction(session_id, user_input, response_text)
            return response_text
        
        #Update memory for this session
        memory_manager.add_interaction(session_id, user_input, response_text)
        return response_text
        
    except Exception as e:
        print(f"Error in therapist wrapper: {e}")
        fallback_response = "I'm experiencing some technical difficulties. Please try again in a moment."
        memory_manager.add_interaction(session_id, user_input, fallback_response)
        return fallback_response

def resource_wrapper(user_input: str, user_profile: dict, session_id: str) -> str:
    """Resource agent with session-specific memory and strict document adherence"""
    
    raw_answer = get_resource_context(user_input)
    chat_history = memory_manager.get_formatted_history(session_id)
    profile_summary = f"Name: {user_profile.get('name', 'Unknown')}, Age: {user_profile.get('age', 'Unknown')}, Country: {user_profile.get('country', 'Unknown')}, Financial Status: {user_profile.get('financial', 'Unknown')}"
    
    #Strict document adherence check for resources
    if not raw_answer :
        fallback_response = "I don't have specific resource information for your request. Could you try asking for mental health resources in a different way?"
        memory_manager.add_interaction(session_id, user_input, fallback_response)
        return fallback_response
    
    try:
        prompt = resource_prompt.format(
            question=user_input, 
            raw_answer=raw_answer, 
            user_profile=profile_summary,
            chat_history=chat_history
        )
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        #Update memory for this session
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
        session_id: Existing session ID from client (if available)
    
    Returns:
        dict with agent, response, and session_id
    """
    #Generate new session ID if not provided
    if not session_id:
        session_id = memory_manager.generate_session_id()
        print(f"[DEBUG] Generated new session ID: {session_id}")
    else:
        print(f"[DEBUG] Using existing session ID: {session_id}")
    
    #Load chat history from client if provided
    if chat_history:
        memory_manager.load_chat_history_from_client(session_id, chat_history, user_profile)
        print(f"[DEBUG] Loaded {len(chat_history)} messages from client history")
    
    #Ensure user profile is stored for this session
    memory_manager.get_or_create_memory(session_id, user_profile)
    
    #Classify and route to appropriate agent
    agent = classify_agent(user_input)
    print(f"[DEBUG] Classified as: {agent}")
    
    if agent == "resource":
        answer = resource_wrapper(user_input, user_profile, session_id)
        agent_name = "Resource Assistant"
    else:
        answer = therapist_wrapper(user_input, user_profile, session_id)
        agent_name = "Therapist"
    
    #Cleanup old sessions periodically
    memory_manager.cleanup_old_sessions()
    
    return {
        "agent": agent_name,
        "response": answer,
        "session_id": session_id
    }

#Utility functions for session management
def clear_user_session(session_id: str):
    """Clear conversation history for a specific session"""
    memory_manager.clear_session(session_id)

def get_session_info(session_id: str):
    """Get information about a session (for debugging)"""
    if session_id in memory_manager.user_memories:
        memory = memory_manager.user_memories[session_id]
        profile = memory_manager.get_user_profile(session_id)
        return {
            "session_exists": True,
            "message_count": len(memory.chat_memory.messages),
            "user_profile": profile
        }
    return {"session_exists": False}