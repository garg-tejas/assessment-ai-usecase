# NeoStats Research Assistant

A chatbot that answers research questions using your own documents and live web search - not just what the model was trained on.

## What it does

- Upload PDFs, text files, or paste a URL in the chat - relevant passages get retrieved before the model responds (RAG via FAISS)
- Live web search via Exa's neural search, on demand or triggered automatically when you paste a URL
- Switch between Gemini and Z.ai from the sidebar dropdown, no code changes needed
- Concise mode (2-3 sentence answers) or Detailed mode (structured with sources)

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure API keys
cp .env.example .env
# fill in your keys

# 3. Run
uv run streamlit run app.py
```

| Key                 | Where to get it                                               |
| ------------------- | ------------------------------------------------------------- |
| `GEMINI_API_KEY`    | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `ZAI_API_KEY`       | [z.ai](https://api.z.ai)                                      |
| `EMBEDDING_API_KEY` | Same as `GEMINI_API_KEY`                                      |
| `EXA_API_KEY`       | [exa.ai](https://exa.ai)                                      |

## Providers

Pick a provider from the sidebar. The presets:

| Provider | Base URL                                                   | Default model         |
| -------- | ---------------------------------------------------------- | --------------------- |
| Gemini   | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-flash-latest` |
| Z.ai     | `https://api.z.ai/api/paas/v4/`                            | `glm-4.5-flash`       |

You can also pick Custom and supply your own base URL and model for any OpenAI-compatible endpoint.

## Project structure

```
├── app.py              # Streamlit UI
├── config/config.py    # Settings loaded from .env
├── models/
│   ├── llm.py          # LLM wrapper (provider-agnostic)
│   └── embeddings.py   # Gemini gemini-embedding-001 embeddings
├── utils/
│   ├── rag.py          # Document loading, FAISS indexing, retrieval
│   └── search.py       # Exa web search
└── pyproject.toml      # uv-managed dependencies
```
