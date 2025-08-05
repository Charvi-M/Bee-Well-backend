from flask import Flask, request, jsonify
from flask_cors import CORS
from FAISS_rag_pipeline import multiagent_chain
import os

app = Flask(__name__)

# Configure CORS for Vercel frontend with your link
CORS(app, origins=["https://your-vercel-app.vercel.app", "http://localhost:3000"])

user_session_data = {}

@app.route("/api/userdata", methods=["POST"])
def receive_user_data():
    try:
        data = request.get_json()
        name = data.get("userName", "unknown")
        
        user_session_data["user"] = {
            "name": name,
            "age": data.get("userAge", ""),
            "country": data.get("userCountry", ""),
            "financial": data.get("financialStatus", ""),
            "diagnosis": data.get("hasDiagnosis", False),
            "is_minor": data.get("isMinor", False)
        }
        
        print(f"[BeeWell] New session: {name}")
        return jsonify({"status": "success"})
        
    except Exception as e:
        print(f"[ERROR] User data error: {e}")
        return jsonify({"status": "error"}), 500

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        user_profile = user_session_data.get("user", {})
        
        print(f"[BeeWell] Processing: {user_input[:50]}...")
        
        result = multiagent_chain(user_input, user_profile)
        
        return jsonify({
            "agent": result.get("agent", "Therapist"),
            "response": result.get("response", "I'm here to help.")
        })
        
    except Exception as e:
        print(f"[BeeWell] Error: {str(e)}")
        return jsonify({
            "agent": "System",
            "response": "Technical difficulties. Please try again."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
