from typing import Literal
import httpx
from langchain_core.tools import tool

from ai_companion.settings import settings

API_URL = "https://api.tavily.com"


_MAX_CONTENT_CHARS = 400


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
        response = httpx.post(
            f"{API_URL}/search", json=payload, timeout=20,
            headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"})
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error: the web search is not reachable right now ({e})."

    data = response.json()
    results = data.get("results", [])
    if not results:
        return f"No web results found for '{query}'. Try a different query."

    parts = [f"Kurzantwort: {data['answer']}"] if data.get("answer") else []
    parts += [_format_result(number, result)
              for number, result in enumerate(results, start=1)]
    return "\n\n".join(parts)


WEB_SEARCH_TOOLS = [web_search]
