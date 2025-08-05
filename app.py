from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from FAISS_rag_pipeline import multiagent_chain
import os

app = Flask(__name__)

# Configure CORS for Vercel frontend
CORS(app, origins=[
    "https://bee-well-ai.vercel.app/",  # Replace with your Vercel URL
    "http://localhost:3000",  # For local development
    "http://127.0.0.1:5500"   # For local development
])

# Global user session state (in-memory for now)
user_session_data = {}

@app.route("/")
def index():
    return {"message": "BeeWell Backend API is running!"}

@app.route("/api/userdata", methods=["POST"])
def receive_user_data():
    data = request.get_json()
    name = data.get("userName", "unknown")
    
    # Store user profile in session data
    user_session_data["user"] = {
        "name": name,
        "age": data.get("userAge", ""),
        "country": data.get("userCountry", ""),
        "financial": data.get("financialStatus", ""),  
        "diagnosis": data.get("hasDiagnosis", False),   
        "is_minor": data.get("isMinor", False)
    }

    print(f"[BeeWell] New session started for {name}")
    print(f"[BeeWell] User profile: {user_session_data['user']}") 
    return jsonify({"status": "success"})

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    data = request.get_json()
    user_input = data.get("message", "")
    user_profile = user_session_data.get("user", {}) 

    print(f"[BeeWell] Received message: {user_input}") 
    print(f"[BeeWell] User profile for chat: {user_profile}") 

    try:
        result = multiagent_chain(user_input, user_profile)
        print(f"[BeeWell] AI Response: {result}")
        
        return jsonify({
            "agent": result.get("agent", "Therapist"),
            "response": result.get("response", "I'm here to help. Could you tell me more about what's on your mind?")
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