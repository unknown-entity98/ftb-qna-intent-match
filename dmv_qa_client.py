import streamlit as st
import json
import openai
import anthropic
from typing import List, Dict, Tuple, Optional
import re
import requests
import base64
from pathlib import Path
from PIL import Image
import os
from dotenv import load_dotenv
from io import BytesIO
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Intent Classification and Routing through LLMs",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS - same as before but removed main-file-uploader styles
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4286f4 0%, #397dd2 100%);
        color: #fff !important;
        min-width: 330px !important;
        padding: 0 0 0 0 !important;
    }
    [data-testid="stSidebar"] .sidebar-title {
        color: #fff !important;
        font-weight: bold;
        font-size: 2.2rem;
        letter-spacing: -1px;
        text-align: center;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    .sidebar-block {
        width: 94%;
        margin: 0 auto 18px auto;
    }
    .sidebar-block label {
        color: #fff !important;
        font-weight: 500;
        font-size: 1.07rem;
        margin-bottom: 4px;
        margin-left: 2px;
        display: block;
        text-align: left;
    }
    .sidebar-block .stSelectbox>div {
        background: #fff !important;
        color: #222 !important;
        border-radius: 13px !important;
        font-size: 1.13rem !important;
        min-height: 49px !important;
        box-shadow: 0 3px 14px #0002 !important;
        padding: 3px 10px !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
    }
    .sidebar-block .stTextInput>div>div>input {
        background: #fff !important;
        color: #222 !important;
        border-radius: 13px !important;
        font-size: 1.13rem !important;
        min-height: 49px !important;
        box-shadow: 0 3px 14px #0002 !important;
        padding: 3px 10px !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        border: none !important;
    }
    .sidebar-block .stRadio>div>label {
        background: #fff !important;
        color: #222 !important;
        border-radius: 13px !important;
        font-size: 1.13rem !important;
        min-height: 49px !important;
        box-shadow: 0 3px 14px #0002 !important;
        padding: 3px 10px !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    .sidebar-block .stSlider {
        padding: 10px !important;
        background: #fff !important;
        border-radius: 13px !important;
        margin-top: 4px !important;
        box-shadow: 0 3px 14px #0002 !important;
    }
    .sidebar-block .stSlider label {
        color: #222 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        background: #39e639;
        color: #222;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #32d932;
        transform: translateY(-1px);
    }
    .sidebar-logo-label {
        margin-top: 30px !important;
        margin-bottom: 10px;
        font-size: 1.13rem !important;
        font-weight: 600;
        text-align: center;
        color: #fff !important;
        letter-spacing: 0.1px;
    }
    .sidebar-logo-row {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .sidebar-logo-row img {
        width: 42px;
        height: 42px;
        border-radius: 9px;
        background: #fff;
        padding: 6px 8px;
        object-fit: contain;
        box-shadow: 0 2px 8px #0002;
    }
    .sidebar-block .stRadio > div {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 13px !important;
    padding: 8px !important;
    color: white !important;
    margin-top: 4px !important;
    }

    .sidebar-block .stRadio > div > label {
        color: #fff !important;
        font-weight: 500 !important;
        font-size: 1.07rem !important;
        margin-bottom: 4px !important;
    }

    .sidebar-block .stRadio > div > div > label {
        color: #fff !important;
        font-weight: 500 !important;
    }

    .sidebar-block .stRadio input[type="radio"] + div {
        color: #fff !important;
    }

    .sidebar-block .stRadio label span {
        color: #fff !important;
    }
    .sidebar-block2 label {
    color: white !important;
    }
    
    /* Style the text for the individual radio options */
    .stRadio p {
        color: white !important;
    }
    /* Chat area needs bottom padding so sticky bar does not overlap */
    .stChatPaddingBottom { padding-bottom: 98px; }
    /* Responsive sticky chatbar */
    .sticky-chatbar {
        position: fixed;
        left: 330px;
        right: 0;
        bottom: 0;
        z-index: 100;
        background: #f8fafc;
        padding: 0.6rem 2rem 0.8rem 2rem;
        box-shadow: 0 -2px 24px #0001;

    }
    @media (max-width: 800px) {
        .sticky-chatbar { left: 0; right: 0; padding: 0.6rem 0.5rem 0.8rem 0.5rem; }
        [data-testid="stSidebar"] { display: none !important; }
    }
    .chat-bubble {
        padding: 13px 20px;
        margin: 8px 0;
        border-radius: 18px;
        max-width: 75%;
        font-size: 1.09rem;
        line-height: 1.45;
        box-shadow: 0 1px 4px #0001;
        display: inline-block;
        word-break: break-word;
    }
    .user-msg {
        background: #e6f0ff;
        color: #222;
        margin-left: 24%;
        text-align: right;
        border-bottom-right-radius: 6px;
        border-top-right-radius: 24px;
    }
    .agent-msg {
        background: #f5f5f5;
        color: #222;
        margin-right: 24%;
        text-align: left;
        border-bottom-left-radius: 6px;
        border-top-left-radius: 24px;
    }
    .chat-row {
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.6rem;
    }
    .avatar {
        height: 36px;
        width: 36px;
        border-radius: 50%;
        margin: 0 8px;
        object-fit: cover;
        box-shadow: 0 1px 4px #0002;
    }
    .user-avatar { order: 2; }
    .agent-avatar { order: 0; }
    .user-bubble { order: 1; }
    .agent-bubble { order: 1; }
    .right { justify-content: flex-end; }
    .left { justify-content: flex-start; }
    .chatbar-claude {
        display: flex;
        align-items: center;
        gap: 12px;
        width: 100%;
        max-width: 850px;
        margin: 0 auto;
        border-radius: 20px;
        background: #fff;
        box-shadow: 0 2px 8px #0002;
        padding: 8px 14px;
        margin-bottom: 0;
    }
    .claude-hamburger {
        background: #f2f4f9;
        border: none;
        border-radius: 11px;
        font-size: 1.35rem;
        font-weight: bold;
        width: 38px; height: 38px;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.13s;
    }
    .claude-hamburger:hover { background: #e6f0ff; }
    .claude-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 1.12rem;
        padding: 0.45rem 0.5rem;
        background: #f5f7fa;
        border-radius: 8px;
        min-width: 60px;
    }
    .claude-send {
        background: #fe3044 !important;
        color: #fff !important;
        border: none;
        border-radius: 50%;
        width: 40px; height: 40px;
        font-size: 1.4rem !important;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.17s;
    }
    .claude-send:hover { background: #d91d32 !important; }

    /* File uploader styling */
    .stFileUploader>div {
        background: #fff !important;
        color: #222 !important;
        border-radius: 13px !important;
        font-size: 1.13rem !important;
        box-shadow: 0 3px 14px #0002 !important;
        padding: 10px !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        border: 1px dashed #4286f4 !important;
    }

    /* Status indicators */
    .status-success {
        background: #39e639;
        color: #fff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-weight: 500;
    }
    .status-warning {
        background: #ff9800;
        color: #fff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-weight: 500;
    }
    .status-error {
        background: #f44336;
        color: #fff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    .stDeployButton, #MainMenu, footer, .stActionButton {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'server_connected' not in st.session_state:
    st.session_state.server_connected = False
if 'qa_data_loaded' not in st.session_state:
    st.session_state.qa_data_loaded = False
if 'agent_data_loaded' not in st.session_state:
    st.session_state.agent_data_loaded = False
if 'agent_qa_data' not in st.session_state:
    st.session_state.agent_qa_data = []
if 'classification_method' not in st.session_state:
    st.session_state.classification_method = "MCP Server"
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = "GPT"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.3
if 'server_url' not in st.session_state:
    st.session_state.server_url = "http://localhost:8000"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_input_box" not in st.session_state:
    st.session_state["chat_input_box"] = ""


class HTTPIntentClient:
    """HTTP client for intent classification server"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30

    def health_check(self) -> Dict:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to server: {e}")

    def load_qa_data(self, qa_data: Dict) -> Dict:
        """Load Q&A data to server"""
        try:
            response = self.session.post(
                f"{self.base_url}/load_data",
                json=qa_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")

    def classify_intent(self, query: str, threshold: float = 0.3) -> Dict:
        """Classify user intent"""
        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                json={"query": query, "threshold": threshold}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to classify intent: {e}")

    def get_stats(self) -> Dict:
        """Get server statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get stats: {e}")


class AIModelClient:
    """AI model client for enhanced responses"""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.client = None

        if provider == "OpenAI":
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "Anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)

    def enhance_response(self, intent_response: str, user_query: str) -> str:
        """Enhance intent response with AI model"""
        if not self.client:
            return intent_response

        try:
            prompt = f"""Based on this intent classification result, provide a more conversational and helpful response:

User Query: {user_query}
Intent Response: {intent_response}

Make the response more natural and helpful while keeping all the technical information."""

            if self.provider == "OpenAI":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content

            elif self.provider == "Anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

        except Exception as e:
            return f"{intent_response}\n\n*(AI enhancement failed: {e})*"

        return intent_response


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


def get_image_base64(img_path):
    """Get base64 encoded image"""
    try:
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
    except:
        return ""


def display_header():
    """Display the app header with DMV logo"""
    # Use the GitHub raw URL for the DMV logo
    dmv_logo_url = "https://raw.githubusercontent.com/unknown-entity98/ftb-qna-intent-match/main/static/icons/dmv.jpeg"

    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center; margin-bottom:20px;'>
            <img src='{dmv_logo_url}' width='180' style='border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 18px;
            padding: 10px 0 10px 0;
        ">
            <span style="
                font-size: 2.5rem;
                font-weight: bold;
                letter-spacing: -2px;
                color: #222;
            ">
                Intent Classification and Routing through LLMs
            </span>
            <span style="
                font-size: 1.15rem;
                color: #555;
                margin-top: 0.35rem;
            ">
            </span>
            <hr style="
            width: 80%;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #4286f4, transparent);
            margin: 20px auto;
            ">
        </div>
        """,
        unsafe_allow_html=True
    )


def display_qa_upload_section():
    """Display Q&A upload section on main page - only when Agent is selected"""
    if st.session_state.classification_method == "Agent":
        st.markdown("### 📄 Upload Q&A JSON File")

        uploaded_file = st.file_uploader(
            "",
            type=['json'],
            help="Upload your Q&A JSON file to enable intent matching",
            key="main_qa_upload",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            qa_data = load_qa_data(uploaded_file)
            if qa_data:
                # Store in session state for agent processing
                st.session_state.agent_qa_data = qa_data
                st.session_state.agent_data_loaded = True
                st.success(f"✅ Successfully loaded {len(qa_data)} Q&A pairs for Agent processing")
            else:
                st.error("❌ Failed to load Q&A data. Please check your file format.")

        st.markdown("---")


def clean_template_syntax(text: str) -> str:
    """Clean template syntax from answer text"""
    if not text:
        return "No answer available"

    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\* \[([^\]]+)\]\(([^)]+)\)', r'• \1: \2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()

    return text


def connect_to_server(server_url: str):
    """Connect to HTTP intent server"""
    try:
        client = HTTPIntentClient(server_url)
        health = client.health_check()

        st.session_state.http_client = client
        st.session_state.server_connected = True
        st.session_state.qa_data_loaded = health.get('data_loaded', False)

        return health

    except Exception as e:
        st.session_state.server_connected = False
        st.session_state.qa_data_loaded = False
        raise e


def process_user_query_via_server(user_query: str, threshold: float) -> str:
    """Process user query via MCP server"""
    if not st.session_state.server_connected:
        return "❌ Not connected to server"

    if not st.session_state.qa_data_loaded:
        return "❌ No Q&A data loaded on server. Please upload a file first."

    try:
        result = st.session_state.http_client.classify_intent(user_query, threshold)

        response_parts = []

        if result["status"] == "match_found":
            intent = result["intent"]
            response_parts.append(f"**Intent Match Found** (Confidence: {result['confidence']:.2f})")

            cleaned_answer = clean_template_syntax(intent.get('answer', ''))
            response_parts.append(cleaned_answer)

            buttons = intent.get('buttons', [])
            if buttons:
                response_parts.append("**Available Options:**")
                for button in buttons[:3]:
                    button_text = button.get('text', 'Unknown')
                    response_parts.append(f"• {button_text}")

            title = intent.get('title', '')
            qid = intent.get('qid', '')
            if title or qid:
                tech_details = []
                if title:
                    tech_details.append(f"Title: {title}")
                if qid:
                    tech_details.append(f"Question ID: {qid}")
                response_parts.append(f"*Technical Details: {' | '.join(tech_details)}*")

        else:
            response_parts.append(f"**No intent match found** (Best confidence: {result['confidence']:.2f})")
            response_parts.append("Please try rephrasing your question or contact support for assistance.")

        return "\n\n".join(response_parts)

    except Exception as e:
        return f"❌ Error processing query: {str(e)}"


def process_user_query_via_agent(user_query: str, threshold: float) -> str:
    """Process user query via local agent"""
    if not st.session_state.agent_data_loaded:
        return "No Q&A data loaded. Please upload a JSON file first."

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        qa_data = st.session_state.agent_qa_data

        # Extract all questions for similarity matching (handle 'q' array structure)
        all_questions = []
        question_map = {}

        for idx, item in enumerate(qa_data):
            questions = item.get('q', [])  # 'q' contains array of questions
            for q in questions:
                all_questions.append(q.lower().strip())
                question_map[len(all_questions) - 1] = idx

        if not all_questions:
            return "No questions found in the uploaded data."

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            max_features=5000
        )

        # Fit vectorizer on all questions and user query
        all_text = [user_query.lower().strip()] + all_questions
        tfidf_matrix = vectorizer.fit_transform(all_text)

        # Calculate similarity scores (user query vs all questions)
        user_vector = tfidf_matrix[0:1]
        question_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(user_vector, question_vectors).flatten()

        # Find best match
        best_match_idx = np.argmax(similarity_scores)
        best_score = similarity_scores[best_match_idx]

        response_parts = []

        if best_score >= threshold:
            # Found a good match
            qa_idx = question_map[best_match_idx]
            matched_item = qa_data[qa_idx]

            response_parts.append(f"**Intent Match Found** (Confidence: {best_score:.2f})")

            # Get and clean the answer (handle 'a' field)
            raw_answer = matched_item.get('a', 'No answer available')
            cleaned_answer = clean_template_syntax(raw_answer)
            response_parts.append(cleaned_answer)

            # Add button information if available (handle 'r' field structure)
            r_field = matched_item.get('r', {})
            buttons = r_field.get('buttons', [])

            if buttons:
                response_parts.append("**Available Options:**")
                for button in buttons[:5]:  # Limit to 5 buttons for display
                    button_text = button.get('text', 'Unknown')
                    response_parts.append(f"• {button_text}")

            # Add technical details if available
            title = matched_item.get('title', '')
            qid = matched_item.get('qid', '')
            if title or qid:
                tech_details = []
                if title:
                    tech_details.append(f"Title: {title}")
                if qid:
                    tech_details.append(f"Question ID: {qid}")
                response_parts.append(f"*Technical Details: {' | '.join(tech_details)}*")

            return "\n\n".join(response_parts)
        else:
            response_parts.append(f"**No intent match found** (Best confidence: {best_score:.2f})")
            response_parts.append("Please try rephrasing your question or contact support for assistance.")

            # Show some sample topics
            sample_topics = []
            for item in qa_data[:3]:  # Show first 3
                questions = item.get('q', [])
                if questions:
                    sample_topics.append(questions[0])

            if sample_topics:
                response_parts.append("**Sample topics you can ask about:**")
                for topic in sample_topics:
                    response_parts.append(f"• {topic}")

            return "\n\n".join(response_parts)

    except ImportError:
        return "Agent processing requires scikit-learn. Please install: `pip install scikit-learn`"
    except Exception as e:
        return f"Error processing query with agent: {str(e)}"


def get_model_options(model_type: str) -> List[str]:
    """Get model options based on selected type"""
    if model_type == "GPT":
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
    elif model_type == "Claude":
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    return []


def main():
    # Sidebar content - modified to show server connection only for MCP Server
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Solution Scope</div>", unsafe_allow_html=True)



        with st.container():
            st.markdown('<div class="sidebar-block"><label>Classification By</label></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
            st.selectbox(
                "Classification By",
                options=["MCP Server", "Agent"],
                key="classification_method",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Server Connection Section - only show for MCP Server
        if st.session_state.classification_method == "MCP Server":
            with st.container():
                st.markdown('<div class="sidebar-block"><label>Server Connection</label></div>', unsafe_allow_html=True)
                st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
                server_url = st.text_input(
                    "Server URL:",
                    value=st.session_state.server_url,
                    help="Enter the MCP server URL (IP:Port)",
                    label_visibility="collapsed"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Connection button
            if st.button("🔌 Connect to Server"):
                try:
                    with st.spinner("Connecting to server..."):
                        health = connect_to_server(server_url)
                        st.session_state.server_url = server_url
                    st.success("✅ Connected to server!")
                    if health.get('data_loaded'):
                        st.success("📄 Server has Q&A data loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

            # Connection status
            status_icon = "🟢" if st.session_state.server_connected else "🔴"
            data_icon = "📄" if st.session_state.qa_data_loaded else "📄"

            # Single line status display
            status_text = f"{status_icon} Server"
            if st.session_state.server_connected and st.session_state.qa_data_loaded:
                status_text += f" {data_icon} Data"
            elif st.session_state.server_connected:
                status_text += " (No Data)"

            st.markdown(
                f'<div style="text-align: center; color: #fff; font-size: 0.9rem; margin: 10px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px;">{status_text}</div>',
                unsafe_allow_html=True)

        # Agent Status - only show for Agent
        elif st.session_state.classification_method == "Agent":
            agent_status_icon = "📄" if st.session_state.agent_data_loaded else "❌"
            agent_status_text = f"{agent_status_icon} Agent Data: {'Loaded' if st.session_state.agent_data_loaded else 'Not Loaded'}"

            st.markdown(
                f'<div style="text-align: center; color: #fff; font-size: 0.9rem; margin: 10px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px;">{agent_status_text}</div>',
                unsafe_allow_html=True)

        # AI Model Section
        with st.container():
            st.markdown('<div class="sidebar-block"><label>Select Model</label></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
            model_type = st.selectbox(
                "Model Type",
                options=["None", "GPT", "Claude"],
                key="selected_model_type",
                index=0,
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if model_type != "None":
            model_options = get_model_options(model_type)
            if model_options:
                with st.container():
                    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
                    st.selectbox(
                        "Select Model",
                        options=model_options,
                        key="selected_model",
                        label_visibility="collapsed"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                # API Key input
                api_key_env = f"{model_type.upper()}_API_KEY" if model_type == "GPT" else "ANTHROPIC_API_KEY"
                api_key = os.getenv(api_key_env if model_type == "GPT" else "ANTHROPIC_API_KEY")

                if not api_key:
                    with st.container():
                        st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
                        api_key = st.text_input(
                            f"{model_type} API Key:",
                            type="password",
                            help=f"Enter your {model_type} API key",
                            label_visibility="collapsed"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-success">✅ API key loaded from environment</div>',
                                unsafe_allow_html=True)

        # Similarity Threshold Section
        with st.container():
            st.markdown('<div class="sidebar-block"><label>Similarity Threshold</label></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
            st.slider(
                "",
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                help="Minimum similarity score for intent matching",
                key="similarity_threshold",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Logo section
        st.markdown('<div class="sidebar-logo-label">Built with</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-logo-row">
            <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" title="Streamlit">
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg" title="OpenAI">
            <img src="https://anthropic.com/favicon.ico" title="Anthropic">
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    display_header()

    # Q&A Upload Section on main page - only shows for Agent method
    display_qa_upload_section()

    # Avatar URLs
    user_avatar_url = "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"
    agent_avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

    # Render chat messages
    st.markdown('<div class="stChatPaddingBottom">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row right">
                    <div class="chat-bubble user-msg user-bubble">{msg['content']}</div>
                    <img src="{user_avatar_url}" class="avatar user-avatar" alt="User">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row left">
                    <img src="{agent_avatar_url}" class="avatar agent-avatar" alt="Agent">
                    <div class="chat-bubble agent-msg agent-bubble">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Claude-style sticky chat bar - always present
    st.markdown('<div class="sticky-chatbar"><div class="chatbar-claude">', unsafe_allow_html=True)
    with st.form("chatbar_form", clear_on_submit=True):
        chatbar_cols = st.columns([16, 1])

        # Left: Input Box
        with chatbar_cols[0]:
            placeholder_text = "What's your intent?" if st.session_state.classification_method == "MCP Server" else "Upload JSON file above, then ask your question..."
            if st.session_state.classification_method == "Agent" and st.session_state.agent_data_loaded:
                placeholder_text = "What's your intent?"

            user_query_input = st.text_input(
                "",
                placeholder=placeholder_text,
                label_visibility="collapsed",
                key="chat_input_box"
            )

        # Right: Send Button
        with chatbar_cols[1]:
            send_clicked = st.form_submit_button("➤", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Process chat input
    if send_clicked and user_query_input:
        user_query = user_query_input

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
        })

        # Process query based on classification method
        if st.session_state.classification_method == "MCP Server":
            if st.session_state.server_connected and st.session_state.qa_data_loaded:
                with st.spinner("Processing your query via server..."):
                    try:
                        response = process_user_query_via_server(
                            user_query.strip(),
                            st.session_state.similarity_threshold
                        )

                        # Check if AI enhancement is enabled
                        if (st.session_state.selected_model_type != "None" and
                                st.session_state.selected_model and
                                "Intent Match Found" in response):

                            # Get API key
                            if st.session_state.selected_model_type == "GPT":
                                api_key = os.getenv('OPENAI_API_KEY')
                            else:
                                api_key = os.getenv('ANTHROPIC_API_KEY')

                            if api_key:
                                try:
                                    ai_client = AIModelClient(
                                        "OpenAI" if st.session_state.selected_model_type == "GPT" else "Anthropic",
                                        api_key,
                                        st.session_state.selected_model
                                    )
                                    enhanced_response = ai_client.enhance_response(response, user_query.strip())
                                    response = f"{enhanced_response}\n\n---\n**Original Server Response:**\n{response}"
                                except Exception as e:
                                    response = f"{response}\n\n*(AI enhancement failed: {e})*"

                        # Add agent response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                        })

                    except Exception as e:
                        error_response = f"❌ Error processing query: {str(e)}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_response,
                        })
            else:
                if not st.session_state.server_connected:
                    response = "❌ Please connect to server first using the sidebar."
                else:
                    response = "❌ No Q&A data loaded on server. Please upload a file first."

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })

        else:  # Agent method
            with st.spinner("Processing your query via local agent..."):
                try:
                    response = process_user_query_via_agent(
                        user_query.strip(),
                        st.session_state.similarity_threshold
                    )

                    # Check if AI enhancement is enabled
                    if (st.session_state.selected_model_type != "None" and
                            st.session_state.selected_model and
                            "Intent Match Found" in response):

                        # Get API key
                        if st.session_state.selected_model_type == "GPT":
                            api_key = os.getenv('OPENAI_API_KEY')
                        else:
                            api_key = os.getenv('ANTHROPIC_API_KEY')

                        if api_key:
                            try:
                                ai_client = AIModelClient(
                                    "OpenAI" if st.session_state.selected_model_type == "GPT" else "Anthropic",
                                    api_key,
                                    st.session_state.selected_model
                                )
                                enhanced_response = ai_client.enhance_response(response, user_query.strip())
                                response = f"{enhanced_response}\n\n---\n**Original Agent Response:**\n{response}"
                            except Exception as e:
                                response = f"{response}\n\n*(AI enhancement failed: {e})*"

                    # Add agent response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })

                except Exception as e:
                    error_response = f"❌ Error processing query with agent: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_response,
                    })

        st.rerun()

    # Auto-scroll to bottom
    components.html("""
        <script>
          setTimeout(() => { window.scrollTo(0, document.body.scrollHeight); }, 80);
        </script>
    """)


if __name__ == "__main__":
    main()
