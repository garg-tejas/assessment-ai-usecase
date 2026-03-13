import os
from dotenv import load_dotenv

load_dotenv()

# Identify outbound HTTP requests (suppresses LangChain USER_AGENT warning)
os.environ.setdefault("USER_AGENT", "neostats-research-assistant/1.0")

# Per-provider API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")

# Generic vars - used only when "Custom" is selected in the sidebar
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Embeddings - Gemini gemini-embedding-001
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# Web search - Exa
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# FAISS vector store path (persisted between sessions)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RAG_TOP_K = 4

# Provider presets - picked in the sidebar dropdown.
# To add a new provider just add an entry here.
PROVIDER_PRESETS = {
    "Gemini": {
        "api_key": GEMINI_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-flash-latest",
    },
    "Z.ai": {
        "api_key": ZAI_API_KEY,
        "base_url": "https://api.z.ai/api/paas/v4/",
        "model": "glm-4.5-flash",
    },
}