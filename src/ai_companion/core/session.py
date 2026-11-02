"""Lives in ``core`` rather than ``graph.utils.helpers`` so tools can use it too:
importing anything from ``ai_companion.graph`` builds the graph, which imports the
tools, so a tool importing helpers would be a circular import.
"""

from langchain_core.runnables import RunnableConfig


def get_session_id(config: RunnableConfig) -> str:
    """Return the id identifying whose conversation this is.

    Long-term memories and routines are scoped to this value, so it must never be
    empty: an empty scope would mix every user's data back together. Falls back to
    "default" for runs started without a thread_id (e.g. LangGraph Studio).
    """
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    return str(thread_id) if thread_id is not None else "default"
