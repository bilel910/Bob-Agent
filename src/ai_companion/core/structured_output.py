"""Structured output that does not depend on Groq tool calling.

Lives in ``core`` rather than ``graph.utils.helpers`` so the image module can use
it too without importing helpers, which already imports the image module.

``with_structured_output`` routes through Groq's tool-calling API, and the small
helper models fail it constantly: for a schema as simple as MemoryAnalysis,
``openai/gpt-oss-20b`` returned the correct JSON but refused to wrap it in a tool
call in 7 of 8 calls, each one a wasted round trip ending in
``tool_use_failed``. Asking for plain JSON and parsing it ourselves is one call,
and parsed cleanly in every sample we measured.
"""

import json
import logging
import re

from groq import BadRequestError
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def _json_instruction(schema) -> str:
    """A compact reminder of the exact JSON object the model must return."""
    fields = ", ".join(f'"{name}"' for name in schema.model_fields)
    return (
        "\n\nAntworte ausschliesslich mit einem einzelnen JSON-Objekt mit genau "
        f"diesen Schluesseln: {fields}. Kein Text davor oder danach, keine "
        "Code-Fences."
    )


def _append_instruction(payload, instruction):
    """Add the JSON instruction to whatever the upstream chain handed us."""
    if isinstance(payload, str):
        return payload + instruction
    if isinstance(payload, PromptValue):
        return payload.to_messages() + [HumanMessage(content=instruction.strip())]
    if isinstance(payload, list):
        return list(payload) + [HumanMessage(content=instruction.strip())]
    return payload


def _parse(text: str, schema):
    """Pull the JSON object out of a completion and validate it."""
    cleaned = _FENCE_RE.sub("", str(text).strip()).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return schema.model_validate(json.loads(match.group(0) if match else cleaned))


def _recover_failed_generation(exc: BadRequestError, schema):
    """Rebuild a result from a Groq ``tool_use_failed`` error.

    Only reachable via the tool-calling fallback below, but worth keeping: Groq
    hands back the model's raw output in ``failed_generation``, which is usually
    the JSON we wanted.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    raw = (body.get("error") or {}).get("failed_generation")
    if not raw:
        return None
    try:
        return _parse(raw, schema)
    except Exception:
        return None


def get_structured_runnable(model, schema):
    """Return a Runnable that yields a validated ``schema`` instance.

    Asks the model for plain JSON. If the response cannot be parsed, falls back
    once to ``with_structured_output`` so a model that is better at tool calling
    than at following format instructions still works.
    """
    instruction = _json_instruction(schema)

    def _fallback(payload):
        structured = model.with_structured_output(schema)
        return structured, payload

    def _invoke(payload):
        try:
            return _parse(model.invoke(_append_instruction(payload, instruction)).content, schema)
        except BadRequestError:
            raise
        except Exception as exc:
            logger.warning("JSON parse failed for %s (%s); retrying via tool calling", schema.__name__, exc)
            structured, payload = _fallback(payload)
            try:
                return structured.invoke(payload)
            except BadRequestError as bad:
                recovered = _recover_failed_generation(bad, schema)
                if recovered is None:
                    raise
                return recovered

    async def _ainvoke(payload):
        try:
            result = await model.ainvoke(_append_instruction(payload, instruction))
            return _parse(result.content, schema)
        except BadRequestError:
            raise
        except Exception as exc:
            logger.warning("JSON parse failed for %s (%s); retrying via tool calling", schema.__name__, exc)
            structured, payload = _fallback(payload)
            try:
                return await structured.ainvoke(payload)
            except BadRequestError as bad:
                recovered = _recover_failed_generation(bad, schema)
                if recovered is None:
                    raise
                return recovered

    return RunnableLambda(_invoke, afunc=_ainvoke)
