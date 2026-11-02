import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from ai_companion.core.session import get_session_id
from ai_companion.core.structured_output import get_structured_runnable
from ai_companion.modules.image.image_to_text import ImageToText
from ai_companion.modules.image.text_to_image import TextToImage
from ai_companion.modules.speech import TextToSpeech
from ai_companion.settings import settings


def chat_history(messages: list, keep_current_tool_calls: bool = False) -> list:
    """The conversation as the models should see it: without old tool traffic.

    Tool calls and their results (calendar lookups, journey plans) only matter to
    the turn that made them; Bob's text answer already carries what the user needs.
    Left in, they pile up in every later prompt: one oversized journey result made
    every following request exceed the context window, and gpt-oss on Groq rejects
    tool messages outright ("Tools should have a name!").

    keep_current_tool_calls keeps the tool calls made since the last user message,
    which the action node needs while it is still working through its tools.
    """
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    history = []
    for i, message in enumerate(messages):
        is_tool_traffic = message.type == "tool" or getattr(message, "tool_calls", None)
        if is_tool_traffic and not (keep_current_tool_calls and i > last_human):
            continue
        history.append(message)
    return history


def pending_tool_calls(messages: list) -> list:
    """The tool calls of the last AI message that have no result yet."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "ai":
            answered = {m.tool_call_id for m in messages[i + 1:] if m.type == "tool"}
            return [call for call in messages[i].tool_calls if call["id"] not in answered]
    return []


@lru_cache(maxsize=8)
def _build_chat_model(model_name: str, temperature: float, max_tokens: int | None = None):
    """Build (and reuse) a ChatGroq client.

    Cached because a fresh client per node meant a fresh connection pool per node,
    so nothing was ever kept warm between calls.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_chat_model(temperature: float = 0.7):
    """The model that writes the user-facing reply.

    Capped because Groq rejects any request whose expected output exceeds the
    model's output-tokens-per-minute limit (1000 for this model on the free tier).
    Bob's replies are at most ~100 words, so the cap never cuts a real answer.
    """
    return _build_chat_model(settings.TEXT_MODEL_NAME, temperature, settings.TEXT_MAX_TOKENS)


def get_small_chat_model(temperature: float = 0.3):
    """The model for internal helper calls (routing, memory analysis, scenarios).

    Deliberately a different model from get_chat_model so these calls draw on their
    own Groq rate-limit bucket and cannot throttle the reply the user is waiting on.
    """
    return _build_chat_model(settings.SMALL_TEXT_MODEL_NAME, temperature)


def get_text_to_speech_module():
    return TextToSpeech()


def get_text_to_image_module():
    return TextToImage()


def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


def get_pending_question(state) -> str | None:
    """Return the confirmation question if the graph is paused at an interrupt()."""
    for task in state.tasks:
        for pending in task.interrupts:
            return pending.value["question"]
    return None


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))


__all__ = [
    "AsteriskRemovalParser",
    "get_chat_model",
    "get_image_to_text_module",
    "get_session_id",
    "get_small_chat_model",
    "get_structured_runnable",
    "get_text_to_image_module",
    "get_text_to_speech_module",
    "remove_asterisk_content",
    "get_pending_question",
    "pending_tool_calls",
]
