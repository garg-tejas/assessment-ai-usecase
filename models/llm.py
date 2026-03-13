from pydantic import SecretStr
from langchain_openai import ChatOpenAI


def get_llm_model(api_key: str, base_url: str, model: str) -> ChatOpenAI:
    """ChatOpenAI client for any OpenAI-compatible provider. Swap base_url + model to switch."""
    if not api_key:
        raise ValueError("LLM_API_KEY is not set.")
    if not base_url:
        raise ValueError("LLM_BASE_URL is not set.")
    if not model:
        raise ValueError("LLM_MODEL is not set.")

    try:
        return ChatOpenAI(api_key=SecretStr(api_key), base_url=base_url, model=model)
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LLM: {e}") from e
