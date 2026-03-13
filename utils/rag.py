import os
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


def load_documents(sources: List[str]) -> List[Document]:
    """Load documents from file paths or URLs. Skips failures, raises if nothing loaded."""
    docs: List[Document] = []
    errors: List[str] = []

    for source in sources:
        source = source.strip()
        if not source:
            continue
        try:
            if source.startswith("http://") or source.startswith("https://"):
                loader = WebBaseLoader(source)
            elif source.lower().endswith(".pdf"):
                loader = PyPDFLoader(source)
            else:
                loader = TextLoader(source, encoding="utf-8")

            loaded = loader.load()
            docs.extend(loaded)
            logger.info("Loaded %d page(s) from %s", len(loaded), source)
        except Exception as e:
            errors.append(f"{source}: {e}")
            logger.warning("Could not load %s: %s", source, e)

    if not docs and errors:
        raise RuntimeError("No documents loaded.\n" + "\n".join(errors))

    return docs


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def build_vectorstore(docs: List[Document], doc_embeddings) -> FAISS:
    """Chunk, embed, and save a new FAISS index to disk."""
    try:
        chunks = _splitter().split_documents(docs)
        vs = FAISS.from_documents(chunks, doc_embeddings)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vs.save_local(FAISS_INDEX_PATH)
        logger.info(
            "Saved %d chunks to FAISS index at %s", len(chunks), FAISS_INDEX_PATH
        )
        return vs
    except Exception as e:
        raise RuntimeError(f"Failed to build vector store: {e}") from e


def load_vectorstore(query_embeddings) -> Optional[FAISS]:
    """Load the persisted FAISS index from disk, or None if it doesn't exist."""
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return None
    try:
        vs = FAISS.load_local(
            FAISS_INDEX_PATH, query_embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Loaded FAISS index from %s", FAISS_INDEX_PATH)
        return vs
    except Exception as e:
        logger.warning("Could not load FAISS index: %s", e)
        return None


def add_to_vectorstore(vs: FAISS, docs: List[Document], doc_embeddings) -> FAISS:
    """Add new documents to an existing index and re-persist."""
    try:
        chunks = _splitter().split_documents(docs)
        vs.add_documents(chunks)
        vs.save_local(FAISS_INDEX_PATH)
        return vs
    except Exception as e:
        raise RuntimeError(f"Failed to update vector store: {e}") from e


def query_vectorstore(vs: FAISS, query: str, k: int = RAG_TOP_K) -> str:
    """Return the top-k relevant chunks as a single context string."""
    try:
        results: List[Document] = vs.similarity_search(query, k=k)
        if not results:
            return ""
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.warning("Vector store query failed: %s", e)
        return ""
