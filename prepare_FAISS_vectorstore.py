import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from fastembed.embedding import TextEmbedding

class FastEmbedLangChainWrapper(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return next(self.model.embed([text]))
    
embedding_model = FastEmbedLangChainWrapper(model_name="BAAI/bge-small-en-v1.5")

def check_vectorstore_exists():
    """Check if FAISS vectorstores already exist"""
    therapy_exists = os.path.exists("faiss_therapy_index") and os.path.isdir("faiss_therapy_index")
    resource_exists = os.path.exists("faiss_resource_index") and os.path.isdir("faiss_resource_index")
    return therapy_exists, resource_exists

def load_documents(path):
    """Load scraped documents"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARNING] Document file not found: {path}")
        return ""
    except Exception as e:
        print(f"[ERROR] Failed to load document {path}: {e}")
        return ""

def create_vectorstores():
    """Create FAISS vectorstores from documents"""
    print("[INFO] Creating FAISS vectorstores...")
    
    #Load documents
    therapyDocuments = [
        Document(page_content=load_documents("data/therapy/meditations.txt"), metadata={"source": "Meditations"}),
        Document(page_content=load_documents("data/therapy/cleaveland.txt"), metadata={"source": "Cleaveland Clinic"}),
        Document(page_content=load_documents("data/therapy/who_psych_guidelines.txt"), metadata={"source": "WHO Guidelines"}),
    ]

    resourceDocuments = [
        Document(page_content=load_documents("data/resources/global_helplines.txt"), metadata={"source": "Global Helplines"}),
        Document(page_content=load_documents("data/resources/india_helplines.txt"), metadata={"source": "India Helplines"}),
        Document(page_content=load_documents("data/resources/free_resources_india.txt"), metadata={"source": "Free Resources India"}),
        Document(page_content=load_documents("data/resources/who_psych_guidelines.txt"), metadata={"source": "WHO Guidelines"}),
    ]

    #Filter out empty documents
    therapyDocuments = [doc for doc in therapyDocuments if doc.page_content.strip()]
    resourceDocuments = [doc for doc in resourceDocuments if doc.page_content.strip()]

    if not therapyDocuments:
        print("[WARNING] No therapy documents found!")
        return False
    
    if not resourceDocuments:
        print("[WARNING] No resource documents found!")
        return False

    try:
        #Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_therapy_docs = splitter.split_documents(therapyDocuments)
        split_resource_docs = splitter.split_documents(resourceDocuments)

        print(f"[INFO] Created {len(split_therapy_docs)} therapy chunks and {len(split_resource_docs)} resource chunks")

        #Initialize embedding model
        embedding_model = embedding_model

        #Build FAISS vectorstores
        print("[INFO] Building therapy vectorstore...")
        therapy_vectorstore = FAISS.from_documents(split_therapy_docs, embedding_model)
        
        print("[INFO] Building resource vectorstore...")
        resource_vectorstore = FAISS.from_documents(split_resource_docs, embedding_model)

        #Save vectorstores locally
        print("[INFO] Saving vectorstores...")
        therapy_vectorstore.save_local("faiss_therapy_index")
        resource_vectorstore.save_local("faiss_resource_index")
        
        print("[SUCCESS] FAISS vectorstores created and saved successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create vectorstores: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to check and create vectorstores if needed"""
    print("[INFO] Checking for existing FAISS vectorstores...")
    
    therapy_exists, resource_exists = check_vectorstore_exists()
    
    if therapy_exists and resource_exists:
        print("[INFO] Both FAISS vectorstores already exist. Skipping creation.")
        print("[INFO] Therapy index:", os.path.abspath("faiss_therapy_index"))
        print("[INFO] Resource index:", os.path.abspath("faiss_resource_index"))
        return True
    else:
        print("[INFO] FAISS vectorstores not found or incomplete.")
        print(f"[INFO] Therapy exists: {therapy_exists}, Resource exists: {resource_exists}")
        
        #Create vectorstores
        success = create_vectorstores()
        if success:
            print("[INFO] Vectorstore preparation completed successfully!")
            return True
        else:
            print("[ERROR] Vectorstore preparation failed!")
            return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)  #Exit with error code for build failure
