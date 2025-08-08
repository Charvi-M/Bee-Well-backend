from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from FAISS_rag_pipeline import multiagent_chain
import os

app = Flask(__name__)

#Cross Origin Resource Sharing for Vercel frontend
CORS(app, 
     origins=[
         "https://bee-well-ai.vercel.app",
         "http://localhost:3000",
         "http://127.0.0.1:5500"
     ],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True
)

def normalize_user_profile(user_data):
    return {
        'name': user_data.get('userName', ''),
        'age': user_data.get('userAge', ''),
        'country': user_data.get('userCountry', ''),
        'financial': user_data.get('financialStatus', ''),
        'hasDiagnosis': user_data.get('hasDiagnosis', False),
        'timestamp': user_data.get('timestamp', '')
    }

def convert_frontend_chat_to_backend_format(frontend_chat_history):
    """Convert frontend chat history format to backend expected format"""
    if not frontend_chat_history:
        return []
    
    backend_format = []
    for message in frontend_chat_history:
        backend_format.append({
            'sender': message.get('sender', ''),
            'content': message.get('content', ''),
            'agentType': message.get('agentType', ''),
            'timestamp': message.get('timestamp', '')
        })
    
    return backend_format

@app.route("/")
def index():
    return {"message": "BeeWell Backend API is running!"}

@app.route('/api/chat', methods=['OPTIONS'])
def handle_preflight():
    return '', 200

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    data = request.get_json()
    user_input = data.get("message", "")
    raw_user_profile = data.get("user_data", {})
    frontend_chat_history = data.get("chat_history", [])  # Get chat history from frontend
    existing_session_id = data.get("session_id", None)  # Get existing session ID if available
    
    user_profile = normalize_user_profile(raw_user_profile)
    chat_history = convert_frontend_chat_to_backend_format(frontend_chat_history)

    print(f"[BeeWell] Received message: {user_input}") 
    print(f"[BeeWell] User profile for chat: {user_profile}")
    print(f"[BeeWell] Chat history length: {len(chat_history)}")
    print(f"[BeeWell] Existing session ID: {existing_session_id}")

    try:
        # Pass chat history and session ID to the multiagent chain
        result = multiagent_chain(user_input, user_profile, chat_history, existing_session_id)
        print(f"[BeeWell] AI Response: {result}")
        
        return jsonify({
            "agent": result.get("agent", "Therapist"),
            "response": result.get("response", "I'm here to help. Could you tell me more about what's on your mind?"),
            "session_id": result.get("session_id")  # Return session ID to client
        })
    except Exception as e:
        print(f"[BeeWell] Error: {str(e)}") 
        return jsonify({
            "agent": "System", 
            "response": f"I'm experiencing some technical difficulties right now. Please try again in a moment."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)