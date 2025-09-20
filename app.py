import streamlit as st
import json
import openai
import anthropic
from typing import List, Dict, Tuple, Optional
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Intent Classification and Routing through LLMs",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_data' not in st.session_state:
    st.session_state.qa_data = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'question_vectors' not in st.session_state:
    st.session_state.question_vectors = None

class IntentMatcher:
    def __init__(self, qa_data: List[Dict]):
        self.qa_data = qa_data
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            max_features=5000
        )
        self._prepare_vectors()
    
    def _prepare_vectors(self):
        """Prepare TF-IDF vectors for all questions"""
        all_questions = []
        self.question_map = {}
        
        for idx, item in enumerate(self.qa_data):
            questions = item.get('q', [])
            for q in questions:
                all_questions.append(q.lower().strip())
                self.question_map[len(all_questions) - 1] = idx
        
        if all_questions:
            self.question_vectors = self.vectorizer.fit_transform(all_questions)
        else:
            self.question_vectors = None
    
    def find_best_match(self, user_query: str, threshold: float = 0.3) -> Tuple[Optional[Dict], float]:
        """Find the best matching Q&A pair for the user query"""
        if self.question_vectors is None or self.question_vectors.shape[0] == 0:
            return None, 0.0
        
        # Vectorize user query
        user_vector = self.vectorizer.transform([user_query.lower().strip()])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(user_vector, self.question_vectors).flatten()
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            qa_idx = self.question_map[best_idx]
            return self.qa_data[qa_idx], best_score
        
        return None, best_score

class APIClient:
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider
        
        if provider == "OpenAI":
            openai.api_key = api_key
            self.client = openai
        elif provider == "Anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def get_available_models(self) -> List[str]:
        """Get available models for the provider"""
        try:
            if self.provider == "OpenAI":
                models = self.client.models.list()
                # Filter for chat models
                chat_models = [
                    model.id for model in models.data 
                    if any(x in model.id.lower() for x in ['gpt-3.5', 'gpt-4'])
                ]
                return sorted(chat_models)
            
            elif self.provider == "Anthropic":
                # Claude models available via API
                return [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022", 
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
        except Exception as e:
            st.error(f"Error fetching models for {self.provider}: {str(e)}")
            return []

def load_qa_data(uploaded_file) -> List[Dict]:
    """Load Q&A data from uploaded JSON file"""
    try:
        data = json.load(uploaded_file)
        if 'qna' in data:
            return data['qna']
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return []

def display_header():
    """Display the app header with centered DMV logo"""
    # Center the logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Try to display DMV logo if it exists
        dmv_logo_path = "static/icons/dmv.jpeg"
        if os.path.exists(dmv_logo_path):
            st.image(dmv_logo_path, width=150)
        else:
            # Fallback with emoji
            st.markdown("""
            <div style="text-align: center; font-size: 60px;">
                🚗
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h1>Intent Classification and Routing through LLMs</h1>
        </div>
        """, unsafe_allow_html=True)

def clean_template_syntax(text: str) -> str:
    """Clean template syntax from answer text"""
    if not text:
        return "No answer available"
    
    # Remove handlebars-style conditionals and variables
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    
    # Convert markdown links to plain text with URLs
    text = re.sub(r'\* \[([^\]]+)\]\(([^)]+)\)', r'• \1: \2', text)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def process_user_query(user_query: str, threshold: float) -> str:
    """Process user query and return response as string"""
    
    # Check if we have Q&A data loaded
    if not st.session_state.qa_data or 'intent_matcher' not in st.session_state:
        return "Please upload a Q&A JSON file first to enable intent matching."
    
    # Find best matching intent
    best_match, confidence = st.session_state.intent_matcher.find_best_match(user_query, threshold)
    
    response_parts = []
    
    if best_match:
        # Found a good match in Q&A data
        response_parts.append(f"**Intent Match Found** (Confidence: {confidence:.2f})")
        
        # Get and clean the answer
        raw_answer = best_match.get('a', 'No answer available')
        cleaned_answer = clean_template_syntax(raw_answer)
        
        response_parts.append(cleaned_answer)
        
        # Add button information if available
        r_field = best_match.get('r', {})
        buttons = r_field.get('buttons', [])
        
        if buttons:
            response_parts.append("**Available Options:**")
            for button in buttons[:5]:  # Limit to 5 buttons for display
                button_text = button.get('text', 'Unknown')
                response_parts.append(f"• {button_text}")
        
        # Add technical details if available
        title = best_match.get('title', '')
        qid = best_match.get('qid', '')
        if title or qid:
            tech_details = []
            if title:
                tech_details.append(f"Title: {title}")
            if qid:
                tech_details.append(f"Question ID: {qid}")
            response_parts.append(f"*Technical Details: {' | '.join(tech_details)}*")
    
    else:
        # No good match found
        response_parts.append(f"**No intent match found** (Best confidence: {confidence:.2f})")
        response_parts.append("Please try rephrasing your question or contact support for assistance.")
        
        # Show some sample topics
        if st.session_state.qa_data:
            sample_topics = []
            for item in st.session_state.qa_data[:3]:  # Show first 3
                questions = item.get('q', [])
                if questions:
                    sample_topics.append(questions[0])
            
            if sample_topics:
                response_parts.append("**Sample topics you can ask about:**")
                for topic in sample_topics:
                    response_parts.append(f"• {topic}")
    
    return "\n\n".join(response_parts)

def main():
    display_header()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        st.subheader("Upload Q&A JSON File")
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload your Q&A JSON file"
        )
        
        if uploaded_file is not None:
            qa_data = load_qa_data(uploaded_file)
            if qa_data:
                st.session_state.qa_data = qa_data
                # Initialize intent matcher
                st.session_state.intent_matcher = IntentMatcher(qa_data)
                st.success(f"✓ Loaded {len(qa_data)} Q&A pairs")
        
        # Model Selection
        st.subheader("Model Selection")
        
        # Get API keys from environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Provider selection - only show providers with available keys
        available_providers = []
        if openai_key:
            available_providers.append("OpenAI")
        if anthropic_key:
            available_providers.append("Anthropic")
        
        if available_providers:
            provider = st.selectbox(
                "Select AI Provider",
                available_providers,
                help="Choose between OpenAI and Anthropic models"
            )
            
            # Get the appropriate API key
            api_key = openai_key if provider == "OpenAI" else anthropic_key
            
            # Model selection
            if api_key:
                try:
                    client = APIClient(api_key, provider)
                    models = client.get_available_models()
                    
                    if models:
                        selected_model = st.selectbox(
                            f"Select {provider} Model",
                            models,
                            help=f"Choose from available {provider} models"
                        )
                        st.session_state.api_client = client
                        st.session_state.selected_model = selected_model
                        
                        # Show selected configuration
                        st.success(f"✓ Using {provider}: {selected_model}")
                    else:
                        st.warning(f"No models found for {provider}")
                except Exception as e:
                    st.error(f"Error connecting to {provider}: {str(e)}")
        else:
            st.info("Add API keys to .env file to enable model selection")
        
        # Intent matching threshold
        st.subheader("Intent Matching")
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum similarity score for intent matching"
        )
    
    # Main chat interface
    st.markdown("### Ask your question:")
    
    # Display all chat history in sequence with proper responses
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            # User message with avatar/icon
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**You:** {message['content']}")
        else:
            # Assistant message with avatar/icon  
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown("**Assistant:**")
                # Display the stored response content
                if 'content' in message and message['content'] != "Response provided above":
                    st.markdown(message['content'])
                else:
                    st.markdown("*Previous response*")
    
    # User input at the bottom
    if prompt := st.chat_input("What's your question?"):
        # Process the query and get response
        with st.spinner("Finding matching information..."):
            response = process_user_query(prompt, threshold)
        
        # Add both user message and assistant response to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Trigger a rerun to display the new messages
        st.rerun()

if __name__ == "__main__":
    main()