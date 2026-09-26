import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from ai_companion.core.structured_output import get_structured_runnable
from ai_companion.modules.image.image_to_text import ImageToText
from ai_companion.modules.image.text_to_image import TextToImage
from ai_companion.modules.speech import TextToSpeech
from ai_companion.settings import settings


def get_session_id(config: RunnableConfig) -> str:
    """Return the id identifying whose conversation this is.

    Long-term memories are scoped to this value, so it must never be empty: an
    empty scope would mix every user's memories back together. Falls back to
    "default" for runs started without a thread_id (e.g. LangGraph Studio).
    """
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    return str(thread_id) if thread_id is not None else "default"


@lru_cache(maxsize=8)
def _build_chat_model(model_name: str, temperature: float):
    """Build (and reuse) a ChatGroq client.

    Cached because a fresh client per node meant a fresh connection pool per node,
    so nothing was ever kept warm between calls.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
    )


def get_chat_model(temperature: float = 0.7):
    """The model that writes the user-facing reply."""
    return _build_chat_model(settings.TEXT_MODEL_NAME, temperature)


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
]
