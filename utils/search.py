import re
import logging
from typing import List

from exa_py import Exa

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(
    r"https?://"
    r"(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}"
    r"(?::\d+)?"
    r"(?:/[^\s]*)?"
)


def extract_urls_from_prompt(prompt: str) -> List[str]:
    """Return deduplicated list of http/https URLs found in the prompt."""
    seen = set()
    urls = []
    for url in _URL_PATTERN.findall(prompt):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def web_search(query: str, api_key: str, num_results: int = 5) -> str:
    """Search the web with Exa and return a formatted context string."""
    try:
        if not api_key:
            logger.warning("EXA_API_KEY not set; skipping web search.")
            return ""

        client = Exa(api_key=api_key)
        response = client.search_and_contents(
            query,
            num_results=num_results,
            text={"max_characters": 400},
            type="neural",
        )

        parts: List[str] = []
        for i, result in enumerate(response.results, 1):
            title = result.title or "No title"
            url = result.url or ""
            content = (result.text or "").strip()
            parts.append(f"{i}. {title}\n   {url}\n   {content}")

        return "\n\n".join(parts) if parts else ""

    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return ""
