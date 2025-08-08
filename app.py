from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from FAISS_rag_pipeline import multiagent_chain
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cross Origin Resource Sharing for Vercel frontend
CORS(app, 
     origins=[
         "https://bee-well-ai.vercel.app",
         "http://localhost:3000",
         "http://127.0.0.1:5500",
         "http://localhost:5500"  # Added for development
     ],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True
)

def normalize_user_profile(user_data):
    """Normalize user profile data from frontend format"""
    if not user_data:
        return {}
        
    normalized = {
        'name': user_data.get('userName', '').strip(),
        'age': user_data.get('userAge', '').strip(),
        'country': user_data.get('userCountry', '').strip(),
        'financial': user_data.get('financialStatus', '').strip(),
        'hasDiagnosis': user_data.get('hasDiagnosis', False),
        'timestamp': user_data.get('timestamp', '').strip()
    }
    
    # Validate required fields
    if not normalized['name'] or not normalized['timestamp']:
        logger.warning("Missing required user profile fields")
        return {}
    
    return normalized

def convert_frontend_chat_to_backend_format(frontend_chat_history):
    """Convert frontend chat history format to backend expected format"""
    if not frontend_chat_history or not isinstance(frontend_chat_history, list):
        return []
    
    backend_format = []
    for message in frontend_chat_history:
        if not isinstance(message, dict):
            continue
            
        backend_format.append({
            'sender': message.get('sender', ''),
            'content': message.get('content', ''),
            'agentType': message.get('agentType', ''),
            'timestamp': message.get('timestamp', '')
        })
    
    return backend_format

def validate_request_data(data):
    """Validate incoming request data"""
    if not data:
        return False, "No data provided"
    
    if not data.get("message", "").strip():
        return False, "No message provided"
    
    user_data = data.get("user_data", {})
    if not user_data.get('userName') or not user_data.get('timestamp'):
        return False, "Invalid user profile data"
    
    return True, None

@app.route("/")
def index():
    return {"message": "BeeWell Backend API is running!", "version": "2.0"}

@app.route("/health")
def health_check():
    return {"status": "healthy", "service": "BeeWell Backend"}

@app.route('/api/chat', methods=['OPTIONS'])
def handle_preflight():
    """Handle CORS preflight requests"""
    return '', 200

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    """Main chat endpoint with enhanced session management"""
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
        
        # Validate request data
        is_valid, error_message = validate_request_data(data)
        if not is_valid:
            logger.error(f"Invalid request data: {error_message}")
            return jsonify({"error": error_message}), 400
        
        # Extract request components
        user_input = data.get("message", "").strip()
        raw_user_profile = data.get("user_data", {})
        frontend_chat_history = data.get("chat_history", [])
        existing_session_id = data.get("session_id", None)
        
        # Normalize and validate user profile
        user_profile = normalize_user_profile(raw_user_profile)
        if not user_profile:
            logger.error("Failed to normalize user profile")
            return jsonify({
                "error": "Invalid user profile data"
            }), 400
        
        # Convert chat history format
        chat_history = convert_frontend_chat_to_backend_format(frontend_chat_history)
        
        # Log request details (without sensitive data)
        logger.info(f"[BeeWell] Processing request for user: {user_profile.get('name', 'Unknown')}")
        logger.info(f"[BeeWell] Message length: {len(user_input)} chars")
        logger.info(f"[BeeWell] Chat history: {len(chat_history)} messages")
        logger.info(f"[BeeWell] Session ID provided: {bool(existing_session_id)}")
        
        # Process request through multiagent chain
        result = multiagent_chain(
            user_input=user_input,
            user_profile=user_profile,
            chat_history=chat_history,
            session_id=existing_session_id
        )
        
        if not result:
            logger.error("Multiagent chain returned empty result")
            return jsonify({
                "agent": "System",
                "response": "I'm experiencing technical difficulties. Please try again.",
                "session_id": existing_session_id
            }), 500
        
        # Log successful response
        logger.info(f"[BeeWell] Response generated by {result.get('agent', 'Unknown')} agent")
        logger.info(f"[BeeWell] Session ID: {result.get('session_id', 'None')}")
        
        # Return response
        response_data = {
            "agent": result.get("agent", "Therapist"),
            "response": result.get("response", "I'm here to help. Could you tell me more about what's on your mind?"),
            "session_id": result.get("session_id", existing_session_id)
        }
        
        return jsonify(response_data)
        
    except ValueError as ve:
        logger.error(f"[BeeWell] Validation error: {str(ve)}")
        return jsonify({
            "agent": "System",
            "response": "I'm having trouble understanding your request. Please try again.",
            "error": "validation_error"
        }), 400
        
    except Exception as e:
        logger.error(f"[BeeWell] Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "agent": "System",
            "response": "I'm experiencing technical difficulties right now. Please try again in a moment.",
            "error": "internal_error"
        }), 500

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session_info(session_id):
    """Debug endpoint to get session information"""
    try:
        from FAISS_rag_pipeline import get_session_info
        info = get_session_info(session_id)
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        return jsonify({"error": "Failed to get session info"}), 500

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """Debug endpoint to list active sessions"""
    try:
        from FAISS_rag_pipeline import memory_manager
        sessions = list(memory_manager.user_memories.keys())
        return jsonify({
            "active_sessions": len(sessions),
            "sessions": sessions[:10]  # Return first 10 for privacy
        })
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({"error": "Failed to list sessions"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# Middleware to log all requests
@app.before_request
def log_request_info():
    if request.path not in ['/health', '/']:  # Skip health checks
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting BeeWell Backend on port {port}, debug={debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)