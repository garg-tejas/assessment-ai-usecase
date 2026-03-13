from langchain_google_genai import GoogleGenerativeAIEmbeddings

_MODEL = "models/gemini-embedding-001"


def _make(api_key: str, task_type: str) -> GoogleGenerativeAIEmbeddings:
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY is not set.")
    try:
        return GoogleGenerativeAIEmbeddings(
            model=_MODEL,
            google_api_key=api_key,
            task_type=task_type,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialise embeddings: {e}") from e


def get_doc_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Embeddings for indexing documents (RETRIEVAL_DOCUMENT task type)."""
    return _make(api_key, "retrieval_document")


def get_query_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Embeddings for querying (RETRIEVAL_QUERY task type)."""
    return _make(api_key, "retrieval_query")