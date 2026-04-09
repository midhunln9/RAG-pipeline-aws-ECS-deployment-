"""
Streamlit frontend for the RAG Chatbot API.

Run with:
    streamlit run app.py

Make sure your FastAPI backend is running at API_URL below.
"""
import os
import uuid
import requests
import streamlit as st

# ---------- Configuration ----------
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/ask")
REQUEST_TIMEOUT = 120  # seconds

# ---------- Page Config ----------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 820px;
    }
    .source-card {
        background-color: rgba(128, 128, 128, 0.12);
        border-left: 3px solid #4a6cf7;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
        white-space: pre-wrap;
        color: inherit;
    }
    .source-meta {
        color: rgba(156, 163, 175, 0.95);
        font-size: 0.78rem;
        margin-bottom: 0.4rem;
    }
    .session-pill {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session State ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    # Each message: {"role": "user"/"assistant", "content": str, "sources": list}
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Session")
    st.markdown(
        f"**Session ID:** <span class='session-pill'>{st.session_state.session_id}</span>",
        unsafe_allow_html=True,
    )
    st.caption("This ID is sent with every request to maintain conversation context.")

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 🔌 API Endpoint")
    st.code(API_URL, language="text")
    st.caption("Set the `RAG_API_URL` env variable to override.")

# ---------- Header ----------
st.title("💬 RAG Chatbot")
st.caption("Ask questions and get answers grounded in your documents.")

# ---------- Helper: render a single source ----------
def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            meta = src.get("metadata", {}) or {}
            page = meta.get("page", "N/A")
            source_path = meta.get("source", "Unknown")
            source_name = os.path.basename(source_path) if source_path != "Unknown" else "Unknown"

            with st.expander(f"Source {i} — {source_name} (page {page})", expanded=False):
                st.markdown(
                    f"<div class='source-meta'>📄 <b>{source_name}</b> &nbsp;•&nbsp; Page {page}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='source-card'>{src.get('content', '').strip()}</div>",
                    unsafe_allow_html=True,
                )

# ---------- Render Chat History ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# ---------- Chat Input ----------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Append + render user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API and render assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")

        try:
            response = requests.post(
                API_URL,
                json={
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                },
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "I don't know.")
            sources = data.get("sources", [])

            placeholder.markdown(answer)
            render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except requests.exceptions.ConnectionError:
            err = "❌ **Connection error** — could not reach the API. Is the backend running?"
            placeholder.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
        except requests.exceptions.Timeout:
            err = f"⏱️ **Timeout** — the API did not respond within {REQUEST_TIMEOUT}s."
            placeholder.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
        except requests.exceptions.HTTPError as e:
            err = f"❌ **HTTP {e.response.status_code}** — {e.response.text}"
            placeholder.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )
        except Exception as e:
            err = f"❌ **Unexpected error:** {str(e)}"
            placeholder.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )