import streamlit as st
import chromadb
import style
from util.embedder import Embedder
from util.context import Context
from util.llm_client import LLMClient
from datetime import datetime
from util import agent

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="EasyRAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(style.style_sheet, unsafe_allow_html=True)




# ─── Session state initialisation ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],           # list of {"role": ..., "content": ...}
        "context_tokens": 0,      # tokens used so far
        "context_limit": 128_000, # model context window size
        "uploaded_files": [],     # list of collection names already embedded
        "embedded_collections": [],
        "index": None,            # your vector store / index object goes here
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    # ── Embedder ──────────────────────────────────────────────────
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder()
    # ── Context ───────────────────────────────────────────────────
    if "context" not in st.session_state:
        st.session_state.context = Context()
    # ── LLMClient ─────────────────────────────────────────────────
    if "llm" not in st.session_state:
        st.session_state.agent = agent.build_agent()


def refresh_embedded_collections() -> list[str]:
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        names = sorted(collection.name for collection in collections)
    except Exception:
        names = []

    EMBEDDER.collection_names = names
    st.session_state.uploaded_files = names
    st.session_state.embedded_collections = names
    return names


init_state()
EMBEDDER = st.session_state.embedder
CONTEXT = st.session_state.context
AGENT = st.session_state.agent
refresh_embedded_collections()


# ─── Backend stubs (replace with your real implementations) ───────────────────
 
def embed_file(uploaded_file) -> bool:
    success , _ = EMBEDDER.embed_file(uploaded_file)
    # ── stub: pretend it worked ──
    return success
 
def query_rag(prompt: str, history: list | None = None) -> tuple[str, int]:
    history = history if history is not None else CONTEXT.history

    response = AGENT.run(prompt, history=history)
    total_tokens = len(prompt.split()) + len(response.split()) + len(Embedded_response.split())

    CONTEXT.history.append({"role": "user", "content": prompt})
    CONTEXT.history.append({"role": "assistant", "content": response})

    return [response, total_tokens]
 
 
def stream_query_rag(prompt: str):
    """
    Streaming version of query_rag — yields string chunks.
    Use this with st.write_stream().
 
    Example with Anthropic streaming:
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                yield text
 
    For now we yield the stub response word by word.
    """
    response, _ = query_rag(prompt)
    for word in response.split(" "):
        yield word + " "
 


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⬡ EasyRAG</div>', unsafe_allow_html=True)
 
    # ── File uploader ──
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Upload files to embed",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
 
    if uploaded:
        new_files = [f for f in uploaded if not EMBEDDER.check_duplicate(f)]
        if new_files:
            with st.spinner(f"Embedding {len(new_files)} file(s)…"):
                for f in new_files:
                    success = embed_file(f)
                    if success:
                        refresh_embedded_collections()
 
    # Show embedded collection list
    embedded_collections = refresh_embedded_collections()
    if embedded_collections:
        for collection_name in embedded_collections:
            st.markdown(f'<span class="file-pill">🗂️ {collection_name}</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span style="font-size:12px;color:#444;">No collections embedded yet.</span>',
            unsafe_allow_html=True,
        )
 
    # ── Context window meter ──
    st.markdown('<div class="section-label">Context window</div>', unsafe_allow_html=True)
 
    used  = st.session_state.context_tokens
    limit = st.session_state.context_limit
    pct   = used / limit
 
    st.markdown(
        f'<div class="ctx-numbers">'
        f'{used:,} <span style="color:#444">/ {limit:,} tokens</span>'
        f' &nbsp;<span style="color:#555;font-size:11px;">({pct:.1%})</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
 
    # Colour the bar based on fill level
    bar_color = "#4a9eff"        # blue  — normal
    if pct > 0.6:
        bar_color = "#f5a623"    # amber — getting full
    if pct > 0.85:
        bar_color = "#e25c5c"    # red   — nearly full
 
    st.markdown(
        f"""
        <div style="background:#1e1e1e;border-radius:3px;height:6px;overflow:hidden;margin-bottom:8px;">
          <div style="width:{pct*100:.1f}%;height:100%;background:{bar_color};
                      border-radius:3px;transition:width 0.4s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    if pct > 0.85:
        st.warning("Context nearly full — consider starting a new conversation.")
    elif pct > 0.6:
        st.info("Context over 60% used.")
 
    # ── Controls ──
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)
 
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_tokens = 0
        st.rerun()
 
    use_streaming = st.toggle("Stream responses", value=True)

# ─── Main chat area ────────────────────────────────────────────────────────────
st.markdown('<div class="chat-header">Ask your documents.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-sub">Upload files in the sidebar, then ask anything about them.</div>',
    unsafe_allow_html=True,
)
 
# Render message history
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-icon">⬡</div>
          No messages yet.<br>
          Upload a document and start asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your documents…"):
    print(f"User prompt: {prompt}")
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Generate assistant response
    with st.chat_message("assistant"):
        if use_streaming:
            # Stream token by token
            full_response = st.write_stream(
                stream_query_rag(prompt)
            )
            # Approximate token update for streaming
            # (replace with real token count from your API)
            st.session_state.context_tokens = min(
                st.session_state.context_tokens + len(prompt.split()) * 3,
                st.session_state.context_limit,
            )
        else:
            # Wait for full response
            with st.spinner("Thinking…"):
                full_response, total_tokens = query_rag(
                    prompt, st.session_state.messages
                )
            st.markdown(full_response)
            st.session_state.context_tokens = total_tokens
 
    # Save assistant response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
 
    # Rerun so sidebar context meter updates immediately
    st.rerun()
