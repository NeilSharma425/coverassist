# app.py - Flask Backend for Real-time Conversation Assistant
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import os
import requests
import json
import uuid
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change this in production
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration - In production, use environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# In-memory storage for conversation sessions (use Redis/DB in production)
conversation_sessions = {}

class ConversationSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.conversation_history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_message(self, sender, text):
        message = {
            'sender': sender,
            'text': text,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_context_for_ai(self, max_messages=10):
        """Get recent conversation context for AI processing"""
        recent_messages = self.conversation_history[-max_messages:]
        context = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in recent_messages])
        return context

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    conversation_sessions[session_id] = ConversationSession(session_id)
    logger.info(f"Created new session: {session_id}")
    return jsonify({'session_id': session_id})

@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    """Generate AI response using Gemini API"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_input = data.get('user_input')
        
        if not session_id or session_id not in conversation_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400
        
        session_obj = conversation_sessions[session_id]
        
        # Add user message to conversation
        session_obj.add_message('user', user_input)
        
        # Get conversation context
        context = session_obj.get_context_for_ai()
        
        # Generate AI response
        ai_response = call_gemini_api(user_input, context)
        
        if ai_response:
            # Add AI response to conversation
            session_obj.add_message('assistant', ai_response)
            return jsonify({
                'response': ai_response,
                'conversation_history': session_obj.conversation_history
            })
        else:
            return jsonify({'error': 'Failed to generate AI response'}), 500
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def call_gemini_api(user_input, context):
    """Call Gemini API to generate response"""
    try:
        # Create prompt with context
        prompt = f"""You are a helpful conversation assistant. You're helping someone in a live conversation by suggesting appropriate responses.

Current conversation context:
{context}

The user just said: "{user_input}"

Please provide a natural, contextually appropriate response that the user could say in reply. Keep it conversational and helpful. Don't mention that you're an AI assistant - just provide the suggested response naturally.

Response:"""

        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        # Make request to Gemini API
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"Generated AI response: {generated_text[:100]}...")
                return generated_text
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        logger.error("Gemini API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request error: {str(e)}")
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        
    # Fallback response if API fails
    return generate_fallback_response(user_input)

def generate_fallback_response(user_input):
    """Generate a fallback response if Gemini API fails"""
    input_lower = user_input.lower()
    
    fallback_responses = {
        'greeting': [
            "Hello! It's great to connect with you. How has your day been going?",
            "Hi there! I'm excited to chat with you. What brings you here today?",
            "Good to see you! I hope you're having a wonderful day. What's on your mind?"
        ],
        'question': [
            "That's a really interesting question. From my perspective, I think we should explore this further.",
            "I appreciate you asking that. Let me share my thoughts on this topic.",
            "Great question! I'd love to dive deeper into that with you."
        ],
        'agreement': [
            "I completely agree with you on that point. It's fascinating how these things work.",
            "You're absolutely right! That's exactly what I was thinking about.",
            "I couldn't agree more. That perspective really resonates with me."
        ],
        'default': [
            "That's really insightful. I think you're onto something important there.",
            "I find that perspective really compelling. Can you tell me more about it?",
            "That's a thoughtful point. It reminds me of something I've been considering."
        ]
    }
    
    # Simple keyword matching
    if any(word in input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        response_type = 'greeting'
    elif any(word in input_lower for word in ['what', 'how', 'why', 'where', 'when', '?']):
        response_type = 'question'
    elif any(word in input_lower for word in ['yes', 'agree', 'exactly', 'right', 'correct', 'true']):
        response_type = 'agreement'
    else:
        response_type = 'default'
    
    import random
    response = random.choice(fallback_responses[response_type])
    logger.info(f"Using fallback response: {response}")
    return response

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to conversation assistant'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('speech_result')
def handle_speech_result(data):
    """Handle real-time speech recognition results"""
    try:
        session_id = data.get('session_id')
        transcript = data.get('transcript')
        is_final = data.get('is_final', False)
        
        if session_id and transcript:
            # Emit the transcript to all clients in the room
            emit('transcript_update', {
                'session_id': session_id,
                'transcript': transcript,
                'is_final': is_final,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Speech result - Session: {session_id}, Final: {is_final}, Text: {transcript[:50]}...")
                
    except Exception as e:
        logger.error(f"Error handling speech result: {str(e)}")
        emit('error', {'message': 'Error processing speech'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sessions': len(conversation_sessions)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-gemini-api-key-here':
        logger.warning("⚠️  GEMINI_API_KEY not set! Please set it in your .env file")
        logger.warning("   Get your API key from: https://makersuite.google.com/app/apikey")
    
    # Get port from environment or default to 5000
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info("🚀 Starting Flask Conversation Assistant...")
    logger.info(f"📍 Server running on: http://localhost:{port}")
    logger.info(f"🔧 Debug mode: {debug}")
    
    # Run the application with Socket.IO support
    socketio.run(app, debug=debug, port=port, host='0.0.0.0')