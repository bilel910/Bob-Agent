from typing import Literal
import httpx
from langchain_core.tools import tool

from ai_companion.settings import settings

API_URL = "https://api.tavily.com"


_MAX_CONTENT_CHARS = 400
# A full article is often 20k+ characters. ~2000 is about 500-600 tokens: enough
# for the main content, small enough to stay inside Groq's per-minute budget.
_MAX_PAGE_CHARS = 2000


def _format_result(number: int, result: dict) -> str:
    """One result as two short lines instead of a JSON object."""
    # Excerpts often contain line breaks and runs of spaces from the scraped page.
    content = " ".join(result.get("content", "").split())[:_MAX_CONTENT_CHARS]
    header = f"{number}. {result['title']} ({result['url']})"
    if result.get("published_date"):
        header += f", veröffentlicht {result['published_date']}"
    return f"{header}\n   {content}"


@tool(parse_docstring=True)
def web_search(
    query: str,
    topic: Literal["general", "news"] = "general",
    time_range: Literal["day", "week", "month", "year"] | None = None,
) -> str:
    """Search the internet for current information: news, facts, opening hours, prices, events, weather.

    Use this when the answer depends on recent events or on facts you are not sure about.
    Do not use it for the user's calendar or for bus and train connections.

    Args:
        query: A short search query like you would type into Google, e.g. 'Wetter Berlin morgen'.
        topic: 'news' for current events and headlines, otherwise 'general'.
        time_range: Only results from the last 'day', 'week', 'month' or 'year'. Leave empty for no limit.
    """
    if not settings.TAVILY_API_KEY:
        return "Error: web search is not configured (TAVILY_API_KEY is missing)."

    payload = {"query": query, "topic": topic,
               "max_results": 5, "include_answer": True}
    if time_range:
        payload["time_range"] = time_range

    try:
        data = _post("search", payload)
    except httpx.HTTPError as e:
        return f"Error: the web search is not reachable right now ({e})."

    # data = response.json()
    results = data.get("results", [])
    if not results:
        return f"No web results found for '{query}'. Try a different query."

    parts = [f"Kurzantwort: {data['answer']}"] if data.get("answer") else []
    parts += [_format_result(number, result)
              for number, result in enumerate(results, start=1)]
    return "\n\n".join(parts)


def _post(endpoint: str, payload: dict, timeout: int = 20) -> dict:
    """Call one Tavily endpoint. Raises httpx.HTTPError, the tools turn it into text."""
    response = httpx.post(
        f"{API_URL}/{endpoint}", json=payload, timeout=timeout,
        headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"})
    response.raise_for_status()
    return response.json()


@tool(parse_docstring=True)
def read_webpage(url: str) -> str:
    """Read the full text of one web page.

    Use this when the short excerpt from web_search is not enough to answer, or when
    the user sends a link and wants to know what is on that page.

    Args:
        url: The full address of the page, starting with http:// or https://.
    """
    if not settings.TAVILY_API_KEY:
        return "Error: web search is not configured (TAVILY_API_KEY is missing)."
    if not url.startswith(("http://", "https://")):
        return f"Error: '{url}' is not a web address. It must start with http:// or https://."

    try:
        data = _post("extract", {"urls": [url]}, timeout=30)
    except httpx.HTTPError as e:
        return f"Error: the page could not be loaded ({e})."

    results = data.get("results", [])
    if not results:
        return f"Error: the page {url} could not be read (it may block bots or need a login)."

    text = " ".join(results[0].get("raw_content", "").split())
    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS] + " [... gekürzt]"
    return f"Inhalt von {url}:\n{text}"


WEB_SEARCH_TOOLS = [web_search, read_webpage]
