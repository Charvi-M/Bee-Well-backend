from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from fastembed.embedding import TextEmbedding  
from langchain_core.embeddings import Embeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class FastEmbedLangChainWrapper(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return next(self.model.embed([text]))



embedding_model = FastEmbedLangChainWrapper(model_name="BAAI/bge-small-en-v1.5")

#prompt templates

therapist_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile"], template="""
You are a compassionate clinical psychologist speaking directly with a client.

Client Profile: {user_profile}
Client's Question: {question}
Knowledge Base Insights: "{raw_answer}"

Remember this client's information from their profile. You can refer to their name, age, country, and other details naturally in conversation.

If the user provides symptoms, you must:
1. List possible diagnoses in bullet points
2. Include a disclaimer that you are an AI agent, not a professional
3. Ask if they want professional help or want to talk about it
4. If they want to talk, provide gentle and supportive guidance

Respond naturally and directly to the client. Avoid saying stuff like of course here is a gentle and supportive reply because then the user will feel that you are not talking to them directly.

Your response:
""")

resource_prompt = PromptTemplate(input_variables=["question", "raw_answer", "user_profile"], template="""
User asked: "{question}"
User Profile: {user_profile}
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


dbTherapy = FAISS.load_local("faiss_therapy_index", embedding_model, allow_dangerous_deserialization=True)
dbResources = FAISS.load_local("faiss_resource_index", embedding_model, allow_dangerous_deserialization=True)

retrieverTherapy = dbTherapy.as_retriever()
retrieverResource = dbResources.as_retriever()

   
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

# --- Memory ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

therapy_base_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retrieverTherapy,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
    verbose=True
)
# --- Classifier ---
def classify_agent(user_input: str) -> str:
    prompt = classification_prompt.format(question=user_input)
    response = llm.invoke(prompt)
    result = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()
    return result if result in ["therapist", "resource"] else "therapist"

# --- Therapist Agent ---
def therapist_wrapper(user_input, raw_answer, user_profile):
    prompt = therapist_prompt.format(question=user_input, raw_answer=raw_answer, user_profile=user_profile)
    styled = llm.invoke(prompt)
    return styled.content if hasattr(styled, "content") else str(styled)

# --- Resource Agent ---
def resource_wrapper(user_input, user_profile):
    docs = retrieverResource.invoke(user_input)
    # print("\n[RESOURCE] Retrieved documents:")
    # for i, doc in enumerate(docs):
    #     print(f"  [{i+1}] {doc.page_content[:300]}...\n")

    raw_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = resource_prompt.format(question=user_input, raw_answer=raw_text, user_profile=user_profile)
    styled = llm.invoke(prompt)
    return styled.content if hasattr(styled, "content") else str(styled)

# --- Multiagent Entry Point ---

def multiagent_chain(user_input: str, user_profile: dict) -> dict:
    agent = classify_agent(user_input)
    profile_summary = f"Country: {user_profile.get('country', 'unknown')}, Financial: {user_profile.get('financial', 'unknown')}, Name: {user_profile.get('name', 'unknown')}, Age: {user_profile.get('age', 'unknown')}"

    if agent == "resource":
        answer = resource_wrapper(user_input, profile_summary)
        return {"agent": "Gemini (Resource Assistant)", "response": answer}
    else:
        # Include user profile context in the question
        contextualized_question = f"User Profile: {profile_summary}\nUser Question: {user_input}"
        
        result = therapy_base_chain.invoke({"question": contextualized_question})
        raw_answer = result.get("answer", "")
        #print("\n[THERAPIST] Retrieved documents:")
        #retrieved_docs = result.get("source_documents", [])
        #for doc in retrieved_docs:
        #    print(doc.page_content[:200])
        
        if not raw_answer:
            return {"agent": "Gemini (Therapist)", "response": "I'm here for you. Please share a bit more."}

        styled = therapist_wrapper(user_input, raw_answer, profile_summary)
        return {"agent": "Gemini (Therapist)", "response": styled}