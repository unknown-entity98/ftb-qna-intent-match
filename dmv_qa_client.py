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

# Custom CSS - with sidebar fix
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

    .stChatPaddingBottom { padding-bottom: 98px; }

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
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = "GPT"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.6
if 'server_url' not in st.session_state:
    # Get server URL from environment variable
    st.session_state.server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_input_box" not in st.session_state:
    st.session_state["chat_input_box"] = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

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

    def classify_intent(self, query: str, threshold: float = 0.3, 
                       model_provider: str = "openai", model_name: str = "gpt-4o-mini") -> Dict:
        """Classify user intent"""
        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                json={
                    "query": query, 
                    "threshold": threshold,
                    "model_provider": model_provider,
                    "model_name": model_name
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to classify intent: {e}")

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

def display_header():
    """Display the app header with DMV logo"""
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

def auto_connect_to_server():
    """Auto-connect to server on startup"""
    if not st.session_state.server_connected:
        try:
            client = HTTPIntentClient(st.session_state.server_url)
            health = client.health_check()
            
            st.session_state.http_client = client
            st.session_state.server_connected = True
            st.session_state.qa_data_loaded = health.get('data_loaded', False)
            
            return True
        except Exception:
            st.session_state.server_connected = False
            st.session_state.qa_data_loaded = False
            return False

def upload_data_to_server(qa_data):
    """Upload data to server"""
    if st.session_state.server_connected:
        try:
            result = st.session_state.http_client.load_qa_data({"qna": qa_data})
            st.session_state.qa_data_loaded = True
            return True
        except Exception as e:
            st.error(f"Failed to upload data to server: {e}")
            return False
    return False

def clean_template_syntax(text: str) -> str:
    """Clean template syntax from answer text"""
    if not text:
        return "No answer available"

    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'\* \[([^\]]+)\]\(([^)]+)\)', r'• \1: \2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()

    return text

def process_user_query_via_server(user_query: str, threshold: float) -> str:
    """Process user query via MCP server"""
    if not st.session_state.server_connected:
        return "❌ Not connected to server"

    if not st.session_state.qa_data_loaded:
        return "❌ No Q&A data loaded on server. Please upload a file first."

    try:
        # Get model configuration from session state with proper defaults
        if st.session_state.selected_model_type == "GPT":
            model_provider = "openai"
            model_name = st.session_state.selected_model or "gpt-4o-mini"
        else:  # Claude
            model_provider = "anthropic"
            # Make sure we use a valid Claude model
            if st.session_state.selected_model and "claude" in st.session_state.selected_model.lower():
                model_name = st.session_state.selected_model
            else:
                model_name = "claude-3-haiku-20240307"  # Safe Claude default
        
        # Add debug logging
        print(f"DEBUG: Sending query '{user_query}' with threshold {threshold}")
        print(f"DEBUG: Model type selected: {st.session_state.selected_model_type}")
        print(f"DEBUG: Model name selected: {st.session_state.selected_model}")
        print(f"DEBUG: Using model {model_provider}/{model_name}")
        print(f"DEBUG: Server URL: {st.session_state.server_url}")
        
        # Test server health first
        try:
            health = st.session_state.http_client.health_check()
            print(f"DEBUG: Server health: {health}")
        except Exception as health_error:
            print(f"DEBUG: Health check failed: {health_error}")
            return f"❌ Server health check failed: {health_error}"
        
        result = st.session_state.http_client.classify_intent(
            user_query, threshold, model_provider, model_name
        )
        print(f"DEBUG: Received result: {result}")

        response_parts = []

        if result["status"] == "match_found":
            intent = result["intent"]
            model_used = result.get("model_used", "unknown")
            response_parts.append(f"**Intent Match Found** (Confidence: {result['confidence']:.2f})")
            response_parts.append(f"*Model: {model_used}*")

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

            # Show reasoning if available
            reasoning = result.get("reasoning", "")
            if reasoning:
                response_parts.append(f"*Reasoning: {reasoning}*")

        else:
            model_used = result.get("model_used", "unknown")
            response_parts.append(f"**No intent match found** (Best confidence: {result.get('confidence', 0.0):.2f})")
            response_parts.append(f"*Model: {model_used}*")
            
            reasoning = result.get("reasoning", "")
            if reasoning:
                response_parts.append(f"*Reasoning: {reasoning}*")
                
            response_parts.append("Please try rephrasing your question or contact support for assistance.")

        final_response = "\n\n".join(response_parts)
        print(f"DEBUG: Final response: {final_response}")
        return final_response

    except requests.exceptions.HTTPError as e:
        error_msg = f"❌ HTTP Error: {e}\nServer URL: {st.session_state.server_url}\nCheck if server is running and API format matches."
        print(f"DEBUG: HTTP Exception: {e}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        print(f"DEBUG: General Exception: {e}")
        return error_msg

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
    # Auto-connect to server on startup
    auto_connect_to_server()
    
    # Sidebar content
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>MCP Server Client</div>", unsafe_allow_html=True)

        # Server status display
        server_status_icon = "✅" if st.session_state.server_connected else "❌"
        data_status_icon = "✅" if st.session_state.qa_data_loaded else "❌"
        
        server_status = f"{server_status_icon} Server: {'Connected' if st.session_state.server_connected else 'Disconnected'}"
        if st.session_state.server_connected:
            server_status += f"\n{data_status_icon} Data: {'Loaded' if st.session_state.qa_data_loaded else 'Not Loaded'}"
        
        st.markdown(
            f'<div style="text-align: center; color: #fff; font-size: 0.9rem; margin: 10px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px; white-space: pre-line;">{server_status}</div>',
            unsafe_allow_html=True
        )

        # Show server URL
        st.markdown(
            f'<div style="text-align: center; color: #fff; font-size: 0.8rem; margin: 5px 0; padding: 5px; background: rgba(255,255,255,0.05); border-radius: 5px;">Server: {st.session_state.server_url}</div>',
            unsafe_allow_html=True
        )

        # AI Model Section
        with st.container():
            st.markdown('<div class="sidebar-block"><label>AI Model Selection</label></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
            model_type = st.selectbox(
                "Model Type",
                options=["GPT", "Claude"],
                key="selected_model_type",
                index=0,
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

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

            # API Key status
            api_key_env = f"{model_type.upper()}_API_KEY" if model_type == "GPT" else "ANTHROPIC_API_KEY"
            api_key = os.getenv(api_key_env if model_type == "GPT" else "ANTHROPIC_API_KEY")

            if api_key:
                st.markdown('<div class="status-success">✅ API key loaded from environment</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">❌ API key not found in environment</div>',
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

    # Main content area
    display_header()

    # File Upload Section
    st.markdown("### 📄 Upload Q&A JSON File")
    
    uploaded_file = st.file_uploader(
        "",
        type=['json'],
        help="Upload your Q&A JSON file to send to the MCP server for intent matching",
        key="qa_upload",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Only process if we haven't uploaded this file yet
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            qa_data = load_qa_data(uploaded_file)
            if qa_data:
                with st.spinner("Uploading data to server..."):
                    if upload_data_to_server(qa_data):
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.success(f"✅ Successfully uploaded {len(qa_data)} Q&A pairs to server")
                        st.info("🧠 Server is ready for intent classification!")
                        st.rerun()
            else:
                st.error("❌ Failed to load Q&A data. Please check your file format.")
        else:
            st.success(f"✅ File '{uploaded_file.name}' already uploaded to server")
            st.info("🧠 Server is ready for intent classification!")

    st.markdown("---")

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

    # Sticky chat bar
    st.markdown('<div class="sticky-chatbar"><div class="chatbar-claude">', unsafe_allow_html=True)
    with st.form("chatbar_form", clear_on_submit=True):
        chatbar_cols = st.columns([16, 1])

        # Input Box
        with chatbar_cols[0]:
            placeholder_text = "What's your question?" if st.session_state.qa_data_loaded else "Upload JSON file above, then ask your question..."
            
            user_query_input = st.text_input(
                "",
                placeholder=placeholder_text,
                label_visibility="collapsed",
                key="chat_input_box"
            )

        # Send Button
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

        # Process query via server
        if st.session_state.server_connected and st.session_state.qa_data_loaded:
            with st.spinner("Processing your query via MCP server..."):
                try:
                    response = process_user_query_via_server(
                        user_query.strip(),
                        st.session_state.similarity_threshold
                    )

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
                response = "❌ Server not connected. Please check the MCP_SERVER_URL in your .env file."
            else:
                response = "❌ No Q&A data loaded on server. Please upload a JSON file first."

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
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