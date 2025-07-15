# app_api.py - Flask Backend with ChatGPT & Free APIs
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import os
import requests
import json
import uuid
from datetime import datetime
import logging
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# API Configuration - Choose your preferred option
API_PROVIDER = os.getenv('API_PROVIDER', 'gemini')  # 'gemini', 'openai', 'huggingface'

# Gemini API (FREE)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.0-flash-exp'

# OpenAI API (CHEAP)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # Cheapest option

# Hugging Face API (FREE with limits)
HF_API_KEY = os.getenv('HF_API_KEY')
HF_MODEL = 'microsoft/DialoGPT-large'

# In-memory storage for conversation sessions
conversation_sessions = {}

class ConversationSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.conversation_history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.api_call_count = 0  # Track usage for cost monitoring
    
    def add_message(self, sender, text):
        message = {
            'sender': sender,
            'text': text,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_context_for_ai(self, max_messages=6):
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
    """Generate AI response using selected API"""
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
        
        # Generate AI response using selected API
        ai_response = call_selected_api(user_input, context, session_obj)
        
        if ai_response:
            # Add AI response to conversation
            session_obj.add_message('assistant', ai_response)
            return jsonify({
                'response': ai_response,
                'api_calls_used': session_obj.api_call_count,
                'provider': API_PROVIDER,
                'conversation_history': session_obj.conversation_history
            })
        else:
            return jsonify({'error': 'Failed to generate AI response'}), 500
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def call_selected_api(user_input, context, session_obj):
    """Call the selected API provider"""
    session_obj.api_call_count += 1
    
    try:
        if API_PROVIDER == 'gemini':
            return call_gemini_api(user_input, context)
        elif API_PROVIDER == 'openai':
            return call_openai_api(user_input, context)
        elif API_PROVIDER == 'huggingface':
            return call_huggingface_api(user_input, context)
        else:
            logger.error(f"Unknown API provider: {API_PROVIDER}")
            return generate_fallback_response(user_input)
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return generate_fallback_response(user_input)

def call_gemini_api(user_input, context):
    """Call Google Gemini API (FREE)"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-gemini-api-key':
        logger.error("Gemini API key not configured")
        return None
    
    try:
        prompt = create_conversation_prompt(user_input, context)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 200,
            }
        }
        
        logger.info(f"🤖 Calling Gemini API (FREE)")
        
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                cleaned = validate_and_clean_response(generated_text)
                if cleaned:
                    logger.info(f"✅ Gemini API success")
                    return cleaned
        else:
            logger.error(f"❌ Gemini API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error calling Gemini API: {str(e)}")
    
    return None

def call_openai_api(user_input, context):
    """Call OpenAI ChatGPT API (CHEAP)"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-openai-api-key':
        logger.error("OpenAI API key not configured")
        return None
    
    try:
        prompt = create_conversation_prompt(user_input, context)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a social anxiety conversation assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.3,
        }
        
        cost_estimate = estimate_openai_cost(len(prompt.split()) + 200)
        logger.info(f"💰 Calling OpenAI API ({OPENAI_MODEL}) - Est. cost: ${cost_estimate:.4f}")
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            cleaned = validate_and_clean_response(generated_text)
            if cleaned:
                logger.info(f"✅ OpenAI API success")
                return cleaned
        else:
            logger.error(f"❌ OpenAI API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error calling OpenAI API: {str(e)}")
    
    return None

def call_huggingface_api(user_input, context):
    """Call Hugging Face API (FREE with limits)"""
    if not HF_API_KEY:
        logger.error("Hugging Face API key not configured")
        return None
    
    try:
        prompt = f"Context: {context}\nUser heard: {user_input}\nSuggest a brief, standard, and detailed response:"
        
        headers = {
            'Authorization': f'Bearer {HF_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 150,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        logger.info(f"🤗 Calling Hugging Face API (FREE)")
        
        response = requests.post(
            f'https://api-inference.huggingface.co/models/{HF_MODEL}',
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '').strip()
                if generated_text:
                    logger.info(f"✅ Hugging Face API success")
                    # Convert to our format since HF might not follow our exact format
                    return convert_to_format(generated_text)
        else:
            logger.error(f"❌ Hugging Face API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Error calling Hugging Face API: {str(e)}")
    
    return None

def estimate_openai_cost(total_tokens):
    """Estimate OpenAI API cost"""
    if OPENAI_MODEL == 'gpt-4o-mini':
        return total_tokens * 0.00015 / 1000  # $0.15 per 1K tokens
    elif OPENAI_MODEL == 'gpt-3.5-turbo':
        return total_tokens * 0.0015 / 1000   # $1.50 per 1K tokens
    else:
        return total_tokens * 0.03 / 1000     # Default estimate

def create_conversation_prompt(user_input, context):
    """Create the prompt for any API"""
    if not validate_and_sanitize_input(user_input):
        user_input = "[Invalid input detected]"
    
    examples = get_response_examples(user_input.lower())
    
    prompt = f"""You are a social anxiety conversation assistant. Provide response suggestions for what the user should SAY BACK.

WHAT THE PERSON HEARD: "{user_input}"

CONVERSATION CONTEXT:
{context if context else "Start of conversation"}

RULES:
1. Suggest what the user should SAY BACK in response to "{user_input}"
2. Use EXACT format: Brief: / Standard: / Detailed: / 💡 Tip:
3. Responses must be appropriate replies to what was heard
4. Keep responses natural and helpful for social anxiety

{examples}

Provide appropriate REPLY suggestions:
Brief: [Short reply to "{user_input}"]
Standard: [Natural reply to "{user_input}"]
Detailed: [Engaging reply to "{user_input}"]
💡 Tip: [Confidence tip for this situation]"""

    return prompt

def get_response_examples(user_input_lower):
    """Get examples based on input type"""
    if any(phrase in user_input_lower for phrase in ['how about you', 'what about you']) and any(word in user_input_lower for word in ['good', 'fine', 'great']):
        return """EXAMPLE - Someone says "Things are good, how about you?":
Brief: I'm doing well too!
Standard: I'm doing really well, thanks! Glad to hear you're doing good.
Detailed: I'm having a great day too, thank you for asking! It's nice when we're both doing well.
💡 Tip: Acknowledge they shared first, then share about yourself."""
    
    elif any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
        return """EXAMPLE - Someone says "Hello":
Brief: Hi there!
Standard: Hello! How are you doing?
Detailed: Hi! It's great to see you, how has your day been?
💡 Tip: Smile and make eye contact."""
    
    elif any(phrase in user_input_lower for phrase in ['how are you', 'how\'s your day']):
        return """EXAMPLE - Someone asks "How are you?":
Brief: Pretty good, thanks!
Standard: I'm doing well, thanks for asking! How about you?
Detailed: It's been a really nice day, thanks for asking! How has yours been?
💡 Tip: Always return the question to show interest."""
    
    return """EXAMPLE - General response:
Brief: [Acknowledge what they said]
Standard: [Show interest and engage naturally]
Detailed: [Thoughtful response with follow-up]
💡 Tip: Listen actively and respond genuinely."""

def convert_to_format(text):
    """Convert free-form response to our format"""
    # Simple conversion for APIs that don't follow our format exactly
    brief = text[:30] + "..." if len(text) > 30 else text
    return f"""Brief: {brief}
Standard: That's interesting, tell me more about that.
Detailed: That's really fascinating, I'd love to hear more about your perspective on that.
💡 Tip: Show genuine interest in what they're sharing."""

def validate_and_sanitize_input(user_input):
    """Basic input validation"""
    if not user_input or not isinstance(user_input, str) or len(user_input.strip()) == 0:
        return False
    return len(user_input) <= 500  # Basic length check

def validate_and_clean_response(response):
    """Validate API response format"""
    if not response:
        return None
    
    # Check if response has our expected format
    if 'Brief:' in response and 'Standard:' in response:
        return response
    
    # If not in format, try to extract useful content
    lines = response.split('\n')
    useful_content = [line.strip() for line in lines if line.strip() and not line.startswith(('Note:', 'Remember:'))]
    
    if useful_content:
        # Create a basic format from the response
        content = ' '.join(useful_content[:3])  # Take first few lines
        return f"""Brief: {content[:50]}
Standard: {content[:100] if len(content) > 50 else content}
Detailed: {content}
💡 Tip: Stay engaged and show genuine interest."""
    
    return None

def generate_fallback_response(user_input):
    """Fallback when all APIs fail"""
    input_lower = user_input.lower().strip()
    
    if any(phrase in input_lower for phrase in ['how about you', 'what about you']) and any(word in input_lower for word in ['good', 'fine', 'great']):
        return """Brief: I'm doing well too!
Standard: I'm doing really well, thanks! Glad to hear you're doing good.
Detailed: I'm having a great day too, thank you for asking! It's always nice when things are going well.
💡 Tip: Acknowledge they shared first, then share about yourself."""
    
    elif any(word in input_lower for word in ['hello', 'hi', 'hey']):
        return """Brief: Hi there!
Standard: Hello! How are you doing today?
Detailed: Hi! It's really nice to see you, how has your day been going?
💡 Tip: Smile warmly and make eye contact."""
    
    elif any(phrase in input_lower for phrase in ['how are you', 'how\'s your day']):
        return """Brief: Pretty good, thanks!
Standard: I'm doing well, thanks for asking! How about you?
Detailed: It's been a really good day actually, thanks for asking! How has yours been?
💡 Tip: Always return the question to show interest."""
    
    else:
        return """Brief: That's interesting.
Standard: That sounds really interesting, tell me more about that.
Detailed: That's fascinating, I'd love to hear more about your thoughts on that.
💡 Tip: Show genuine curiosity and interest in what they're sharing."""

@app.route('/api/llm-status')
def llm_status():
    """Check API status (compatible with frontend)"""
    status_info = {
        'provider': API_PROVIDER,
        'status': 'disconnected',
        'message': 'API not configured',
        'details': {}
    }
    
    try:
        if API_PROVIDER == 'gemini':
            if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key':
                # Test the API with a simple request
                test_response = test_gemini_api()
                if test_response:
                    status_info.update({
                        'status': 'ready',
                        'message': 'Gemini API connected and working',
                        'details': {
                            'api_url': 'https://generativelanguage.googleapis.com',
                            'model': GEMINI_MODEL,
                            'cost': 'FREE',
                            'daily_limit': '1500 requests',
                            'test_successful': True
                        }
                    })
                else:
                    status_info.update({
                        'status': 'connected',
                        'message': 'Gemini API key configured but test failed',
                        'details': {
                            'model': GEMINI_MODEL,
                            'cost': 'FREE'
                        }
                    })
            else:
                status_info['message'] = 'Gemini API key not configured'
                status_info['details'] = {
                    'help': 'Get free API key from: https://makersuite.google.com/app/apikey'
                }
                
        elif API_PROVIDER == 'openai':
            if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key':
                test_response = test_openai_api()
                if test_response:
                    status_info.update({
                        'status': 'ready',
                        'message': f'OpenAI API connected - Model: {OPENAI_MODEL}',
                        'details': {
                            'api_url': 'https://api.openai.com',
                            'model': OPENAI_MODEL,
                            'cost_per_1k_tokens': '$0.15' if OPENAI_MODEL == 'gpt-4o-mini' else '$1.50',
                            'estimated_per_conversation': '$0.001-0.003',
                            'test_successful': True
                        }
                    })
                else:
                    status_info.update({
                        'status': 'connected',
                        'message': 'OpenAI API key configured but test failed',
                        'details': {
                            'model': OPENAI_MODEL
                        }
                    })
            else:
                status_info['message'] = 'OpenAI API key not configured'
                status_info['details'] = {
                    'help': 'Get API key from: https://platform.openai.com/api-keys'
                }
                
        elif API_PROVIDER == 'huggingface':
            if HF_API_KEY:
                status_info.update({
                    'status': 'connected',
                    'message': 'Hugging Face API configured',
                    'details': {
                        'api_url': 'https://api-inference.huggingface.co',
                        'model': HF_MODEL,
                        'cost': 'FREE',
                        'monthly_limit': '30,000 requests'
                    }
                })
            else:
                status_info['message'] = 'Hugging Face API key not configured'
                status_info['details'] = {
                    'help': 'Get free API key from: https://huggingface.co/settings/tokens'
                }
                
    except Exception as e:
        status_info['message'] = f'Error checking {API_PROVIDER} API: {str(e)}'
    
    return jsonify(status_info)

def test_gemini_api():
    """Test Gemini API with a simple request"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": "Say 'OK' if you can understand this test."}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
        
        response = requests.post(
            f"{url}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
    except:
        return False

def test_openai_api():
    """Test OpenAI API with a simple request"""
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": "Say 'OK' if you can understand this test."}],
            "max_tokens": 10
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
    except:
        return False

@app.route('/api/status')
def api_status():
    """Check API status and costs"""
    status = {
        'provider': API_PROVIDER,
        'status': 'unknown',
        'cost_info': {}
    }
    
    if API_PROVIDER == 'gemini':
        status['status'] = 'FREE' if GEMINI_API_KEY else 'Not configured'
        status['cost_info'] = {
            'daily_limit': '1500 requests',
            'rate_limit': '15 requests/minute',
            'cost': 'FREE'
        }
    elif API_PROVIDER == 'openai':
        status['status'] = 'PAID' if OPENAI_API_KEY else 'Not configured'
        status['cost_info'] = {
            'model': OPENAI_MODEL,
            'cost_per_1k_tokens': '$0.15' if OPENAI_MODEL == 'gpt-4o-mini' else '$1.50',
            'estimated_per_conversation': '$0.001-0.003'
        }
    elif API_PROVIDER == 'huggingface':
        status['status'] = 'FREE' if HF_API_KEY else 'Not configured'
        status['cost_info'] = {
            'monthly_limit': '30,000 requests',
            'cost': 'FREE'
        }
    
    return jsonify(status)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sessions': len(conversation_sessions),
        'api_provider': API_PROVIDER
    })

if __name__ == '__main__':
    logger.info("🚀 Starting API-Based Conversation Assistant")
    logger.info(f"🤖 API Provider: {API_PROVIDER}")
    
    if API_PROVIDER == 'gemini':
        logger.info("💰 Using Gemini API (FREE)")
        logger.info("   Get free API key: https://makersuite.google.com/app/apikey")
    elif API_PROVIDER == 'openai':
        logger.info(f"💰 Using OpenAI API ({OPENAI_MODEL}) - CHEAP")
        logger.info("   Get API key: https://platform.openai.com/api-keys")
    elif API_PROVIDER == 'huggingface':
        logger.info("💰 Using Hugging Face API (FREE)")
        logger.info("   Get free API key: https://huggingface.co/settings/tokens")
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"📍 Server running on: http://localhost:{port}")
    
    socketio.run(app, debug=debug, port=port, host='0.0.0.0')