"""
app.py - NeoStats Research Assistant

Streamlit UI. Combines RAG, live web search, and multi-provider LLM support.
API keys are read from .env - configure them before running.
"""

import os
import sys
import tempfile
import logging

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    EMBEDDING_API_KEY,
    EXA_API_KEY,
    PROVIDER_PRESETS,
)
from models.llm import get_llm_model
from models.embeddings import get_doc_embeddings, get_query_embeddings
from utils.rag import (
    load_documents,
    build_vectorstore,
    load_vectorstore,
    add_to_vectorstore,
    query_vectorstore,
)
from utils.search import web_search, extract_urls_from_prompt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = {
    "Concise": (
        "You are a research assistant. Answer in 2-3 sentences - be direct and skip preamble. "
        "Use the provided context when relevant and cite sources inline."
    ),
    "Detailed": (
        "You are a research assistant. Give a thorough, well-structured answer. "
        "Use headings or bullet points where helpful, include examples, and cite sources. "
        "Use the provided context when relevant."
    ),
}


def get_response(llm, history: list, mode: str, rag_ctx: str, web_ctx: str) -> str:
    try:
        # Build context block to prepend to the system prompt
        ctx_parts = []
        if rag_ctx:
            ctx_parts.append(f"--- Documents ---\n{rag_ctx}")
        if web_ctx:
            ctx_parts.append(f"--- Web Search ---\n{web_ctx}")

        system = SYSTEM_PROMPT[mode]
        if ctx_parts:
            system += "\n\n" + "\n\n".join(ctx_parts)

        messages: list[BaseMessage] = [SystemMessage(content=system)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        return llm.invoke(messages).content
    except Exception as e:
        logger.error("LLM error: %s", e)
        return f"Something went wrong: {e}"


def get_vectorstore(embedding_key: str):
    """Return vectorstore from session state or load from disk."""
    if "vectorstore" in st.session_state:
        return st.session_state.vectorstore
    if not embedding_key:
        return None
    try:
        vs = load_vectorstore(get_query_embeddings(embedding_key))
        if vs:
            st.session_state.vectorstore = vs
        return vs
    except Exception as e:
        logger.warning("Could not load vectorstore: %s", e)
        return None


def index_files(files, embedding_key: str):
    """Index uploaded files into the FAISS store."""
    with st.spinner("Indexing..."):
        try:
            doc_emb = get_doc_embeddings(embedding_key)
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for f in files:
                    p = os.path.join(tmp, f.name)
                    with open(p, "wb") as fp:
                        fp.write(f.read())
                    paths.append(p)
                docs = load_documents(paths)

            existing = get_vectorstore(embedding_key)
            if existing:
                vs = add_to_vectorstore(existing, docs, doc_emb)
            else:
                vs = build_vectorstore(docs, doc_emb)

            st.session_state.vectorstore = vs
            st.success(f"Indexed {len(files)} file(s).")
            st.rerun()
        except Exception as e:
            st.error(f"Indexing failed: {e}")


def clear_index():
    import shutil

    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.session_state.pop("vectorstore", None)
        st.success("Index cleared.")
        st.rerun()
    except Exception as e:
        st.error(f"Could not clear index: {e}")


def sidebar():
    """Render sidebar and return runtime config."""
    with st.sidebar:
        st.title("Research Assistant")
        st.caption("Configure your session below.")
        st.divider()

        # Provider picker - switches base_url + model automatically
        st.subheader("LLM Provider")
        provider = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()) + ["Custom"])

        if provider != "Custom":
            preset = PROVIDER_PRESETS[provider]
            api_key = preset["api_key"]
            base_url = preset["base_url"]
            model = preset["model"]
        else:
            api_key = LLM_API_KEY
            base_url = LLM_BASE_URL
            model = LLM_MODEL

        # Show which model is active; let user override if needed
        active_model = st.text_input("Model", value=model)

        # Warn only about keys relevant to the active configuration
        missing = []
        if not api_key:
            missing.append(
                "LLM_API_KEY"
                if provider == "Custom"
                else f"{provider.upper().replace('.', '').replace(' ', '_')}_API_KEY"
            )
        if not EMBEDDING_API_KEY:
            missing.append("EMBEDDING_API_KEY")
        if not EXA_API_KEY:
            missing.append("EXA_API_KEY")

        if missing:
            st.warning(f"Missing in .env: {', '.join(missing)}")

        st.divider()

        # Response style
        st.subheader("Response Mode")
        mode = st.radio("Style", ["Detailed", "Concise"], index=0)

        st.divider()

        # Web search toggle
        st.subheader("Web Search")
        web_enabled = st.checkbox(
            "Search the web on every query",
            help="Also triggers automatically when you paste a URL in your message.",
        )

        st.divider()

        # Document upload for RAG
        st.subheader("Knowledge Base")
        indexed = os.path.exists(os.path.join("faiss_index", "index.faiss"))
        if indexed:
            st.success("Index ready.")
        else:
            st.info("No documents indexed yet.")

        uploads = st.file_uploader(
            "Upload files", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
        if st.button("Index Documents", use_container_width=True):
            if not EMBEDDING_API_KEY:
                st.error("EMBEDDING_API_KEY not set in .env.")
            elif not uploads:
                st.warning("Select files first.")
            else:
                index_files(uploads, EMBEDDING_API_KEY)

        if st.button("Clear Index", use_container_width=True):
            clear_index()

        st.divider()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": active_model,
        "mode": mode,
        "web_enabled": web_enabled,
    }


def chat_page(cfg: dict):
    st.title("Research Assistant")
    st.caption("Ask anything - I'll search your documents and the web to answer.")

    # Validate required keys before doing anything
    if not cfg["api_key"]:
        st.error("LLM API key is not set. Add it to your .env file and restart.")
        return

    # Re-use the LLM instance unless the config changed
    llm_key = (cfg["api_key"], cfg["base_url"], cfg["model"])
    if st.session_state.get("_llm_key") != llm_key:
        try:
            st.session_state.llm = get_llm_model(
                cfg["api_key"], cfg["base_url"], cfg["model"]
            )
            st.session_state["_llm_key"] = llm_key
        except Exception as e:
            st.error(f"Could not initialise LLM: {e}")
            return

    llm = st.session_state.llm

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a research question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                # If the user pasted a URL, fetch and index it first
                urls = extract_urls_from_prompt(prompt)
                if urls and EMBEDDING_API_KEY:
                    try:
                        with st.status(f"Fetching {len(urls)} URL(s)..."):
                            doc_emb = get_doc_embeddings(EMBEDDING_API_KEY)
                            url_docs = load_documents(urls)
                            existing = get_vectorstore(EMBEDDING_API_KEY)
                            vs = (
                                add_to_vectorstore(existing, url_docs, doc_emb)
                                if existing
                                else build_vectorstore(url_docs, doc_emb)
                            )
                            st.session_state.vectorstore = vs
                    except Exception as e:
                        st.warning(f"Could not fetch URL(s): {e}")

                # RAG retrieval
                rag_ctx = ""
                vs = get_vectorstore(EMBEDDING_API_KEY)
                if vs:
                    rag_ctx = query_vectorstore(vs, prompt)

                # Web search - runs when enabled or when URLs were in the prompt
                web_ctx = ""
                if (cfg["web_enabled"] or bool(urls)) and EXA_API_KEY:
                    web_ctx = web_search(prompt, EXA_API_KEY)

                response = get_response(
                    llm, st.session_state.messages, cfg["mode"], rag_ctx, web_ctx
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def about_page():
    st.title("About")
    st.markdown("""
## NeoStats Research Assistant

A chatbot built for researchers, analysts, and curious people who need answers
grounded in real sources - not just what the model remembers.

### How it works

1. **Ask a question** - the assistant searches your uploaded documents and/or the web.
2. **RAG (Retrieval-Augmented Generation)** - relevant passages from your files are
   retrieved and injected into the context before the LLM responds.
3. **Exa web search** - when enabled (or when you paste a URL), Exa's neural search
   fetches live, high-quality sources and adds them to the context.
4. **Response modes** - switch between Concise (quick 2-3 sentence answers) and
   Detailed (structured, sourced responses).

### Setup

1. Copy `.env.example` to `.env` and fill in your API keys.
2. Run with `uv run streamlit run app.py`.

### API keys needed

| Key | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini LLM |
| `ZAI_API_KEY` | Z.ai LLM |
| `EMBEDDING_API_KEY` | Gemini embeddings for RAG (same as `GEMINI_API_KEY`) |
| `EXA_API_KEY` | Exa web search |
    """)


def main():
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        page = st.radio("Page", ["Chat", "About"], label_visibility="collapsed")
        st.divider()

    cfg = sidebar()

    if page == "Chat":
        chat_page(cfg)
    else:
        about_page()


if __name__ == "__main__":
    main()
