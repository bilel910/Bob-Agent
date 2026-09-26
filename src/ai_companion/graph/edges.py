from langgraph.graph import END
from typing_extensions import Literal

from ai_companion.graph.state import AICompanionState
from ai_companion.settings import settings


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AICompanionState,
) -> Literal["conversation_node", "image_node", "audio_node", "action_node"]:
    workflow = state["workflow"]

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    elif workflow == "action":
        return "action_node"

    else:
        return "conversation_node"


def should_call_tools(
    state: AICompanionState,
) -> Literal["tools_node", "summarize_conversation_node", "__end__"]:
    last_message = state["messages"][-1]

    # The LLM asked for a tool → run it, then come back to action_node
    if getattr(last_message, "tool_calls", None):
        return "tools_node"

    # The LLM answered with text → finish like every other node
    return should_summarize_conversation(state)
