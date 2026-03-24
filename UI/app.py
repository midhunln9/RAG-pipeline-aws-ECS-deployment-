"""
Streamlit chatbot interface for Financial Compliance RAG.

Provides a clean, professional UI for multi-turn conversations with source citations.
"""

import streamlit as st
import requests
import uuid
import os
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Compliance RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-container h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .header-container p {
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .message-container {
        margin-bottom: 1.5rem;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    .user-bubble {
        background-color: #2563eb;
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        max-width: 70%;
        word-wrap: break-word;
        font-size: 0.95rem;
    }
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    .assistant-bubble {
        background-color: #f0f4f8;
        color: #1f2937;
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #1e40af;
        max-width: 85%;
        word-wrap: break-word;
        font-size: 0.95rem;
    }
    .sources-container {
        background-color: #faf9f7;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        margin-left: 0;
    }
    .source-title {
        font-weight: 600;
        color: #374151;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .source-item {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .source-content {
        color: #6b7280;
        margin-top: 0.5rem;
        line-height: 1.4;
    }
    .source-metadata {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    .error-box {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #dc2626;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #dbeafe;
        color: #0c4a6e;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #0284c7;
        margin-bottom: 1rem;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e7eb;
        padding: 1rem;
        z-index: 100;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_error" not in st.session_state:
        st.session_state.api_error = None


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def send_message(query: str) -> Optional[dict]:
    """Send a message to the API and get a response."""
    try:
        st.session_state.api_error = None
        
        payload = {
            "query": query,
            "session_id": st.session_state.session_id,
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=30,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.session_state.api_error = f"API Error: {response.status_code} - {response.text}"
            return None
    except requests.exceptions.Timeout:
        st.session_state.api_error = "API request timed out. Please try again."
        return None
    except requests.exceptions.ConnectionError:
        st.session_state.api_error = f"Cannot connect to API at {API_BASE_URL}. Make sure the API server is running."
        return None
    except Exception as e:
        st.session_state.api_error = f"Error: {str(e)}"
        return None


def render_sources(sources: list) -> None:
    """Render source documents in a clean format."""
    if not sources:
        return
    
    with st.container():
        st.markdown(
            '<div class="sources-container">'
            '<div class="source-title">📚 Sources</div>',
            unsafe_allow_html=True,
        )
        
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            content = source.get("content", "")
            
            source_name = metadata.get("source", f"Document {i}")
            page = metadata.get("page", None)
            
            source_label = source_name
            if page:
                source_label += f" (Page {page})"
            
            with st.expander(f"📄 {source_label}", expanded=False):
                if content:
                    st.markdown(f"**Content:** {content[:500]}..." if len(content) > 500 else f"**Content:** {content}")
                
                if metadata:
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.markdown(f"- **{key}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_message(msg_type: str, content: str, sources: list = None, timestamp: str = None) -> None:
    """Render a single message with sources if applicable."""
    if msg_type == "user":
        st.markdown(
            f'<div class="user-message"><div class="user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
        if timestamp:
            st.markdown(
                f'<div style="text-align: right; margin-top: -0.75rem;"><span class="timestamp">{timestamp}</span></div>',
                unsafe_allow_html=True,
            )
    
    elif msg_type == "assistant":
        st.markdown(
            f'<div class="assistant-message"><div class="assistant-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
        if timestamp:
            st.markdown(
                f'<div class="timestamp">{timestamp}</div>',
                unsafe_allow_html=True,
            )
        if sources:
            render_sources(sources)


def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown(
        '''
        <div class="header-container">
            <h1>💼 Financial Compliance RAG Chatbot</h1>
            <p>Ask questions about financial compliance policies and regulations</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"Session ID: `{st.session_state.session_id}`")
    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    
    api_healthy = check_api_health()
    if not api_healthy:
        st.warning(
            f"⚠️ Cannot connect to API at {API_BASE_URL}. Please ensure the API server is running.",
            icon="⚠️"
        )
    
    if st.session_state.api_error:
        st.error(st.session_state.api_error)
    
    st.divider()
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            render_message(
                msg_type=message["role"],
                content=message["content"],
                sources=message.get("sources"),
                timestamp=message.get("timestamp"),
            )
    
    st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1], gap="small")
    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask about financial compliance...",
            label_visibility="collapsed",
        )
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    if send_button and user_input:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp,
        })
        
        with st.spinner("Thinking..."):
            response_data = send_message(user_input)
        
        if response_data:
            assistant_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data.get("response", "No response received"),
                "sources": response_data.get("sources", []),
                "timestamp": assistant_timestamp,
            })
            st.rerun()
        else:
            st.error("Failed to get response from the API. Please check the error message above and try again.")


if __name__ == "__main__":
    main()
