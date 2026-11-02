import os
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.core.prompts import ROUTINE_RUN_PROMPT
from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
    get_action_chain,
    ACTION_TOOLS,
    ROUTINE_TOOLS,
)
from ai_companion.graph.utils.helpers import (
    chat_history,
    get_chat_model,
    get_session_id,
    get_text_to_image_module,
    get_text_to_speech_module,
    pending_tool_calls,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.routines import Routine, store as routine_store
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings
from datetime import datetime
from zoneinfo import ZoneInfo

from langgraph.prebuilt import ToolNode


# Cues that a user might be asking for a picture or a voice note. The router only
# ever returns 'image'/'audio' when the request is explicit, so a message with none
# of these words can go straight to 'conversation' without spending an LLM call.
_IMAGE_CUES = (
    "bild", "foto", "photo", "selfie", "zeig", "zeige", "male", "mal mir",
    "zeichne", "skizze", "aussehen", "siehst du aus", "picture", "image",
)
_AUDIO_CUES = (
    "stimme", "sprachnachricht", "sprich", "sag es", "sag das", "hören",
    "hoeren", "vorlesen", "audio", "voice", "sprachi",
)

# "Wenn ich 'Guten Morgen' sage, ...", "Welche Routinen habe ich?"
_ROUTINE_CUES = ("routine", "makro", "macro", "wenn ich", "sobald ich", "immer wenn")

# Sentences that can only be about routines skip the router model. Now and then it sent
# one to the chat model, which replied "Erledigt!" without having saved anything. A false
# match is harmless: saving a routine always asks first. "\b" keeps "Morgenroutine" out.
_ROUTINE_REQUEST = re.compile(r"\b(?:routinen?|makros?|macros?)\b|\b(?:wenn|sobald) ich\b.+\bsage\b", re.IGNORECASE)

_ACTION_CUES = _ROUTINE_CUES + (
    "termin", "meeting", "treffen", "kalender", "calendar", "schedule",
    "plan", "eintrag", "verabred", "was steht", "call",
    "verschieb", "änder", "aender", "umbuch", "verleg", "move", "reschedul", "update",
    "lösch", "loesch", "entfern", "absag", "stornier", "annullier",
    "delete", "remove", "cancel", "cancell",
    "frei", "zeit für", "lücke", "verfügbar", "slot", "free", "available",
    "bahn", "zug", "bus", "tram", "verbindung", "wie komm", "fahr", "haltestelle",
    "abfahrt", "umsteig", "öffis", "oeffis", "flughafen", "bahnhof", "route", "train",
    "such", "google", "internet", "online", "news", "nachricht", "aktuell", "neueste",
    "wetter", "öffnungszeit", "geöffnet", "preis", "kostet", "ergebnis", "wer ist", "was ist",
    "http", "www.", "webseite", "seite", "link", "artikel", "lies", "zusammenfass",
    # Tasks. A photographed list arrives as "[Bildanalyse: ...]", whose description
    # always says "Liste" (IMAGE_ANALYSIS_PROMPT asks for it).
    "liste", "aufgabe", "to-do", "todo", "task", "erledig", "notier", "zettel",
)

_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch",
             "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _last_user_text(state: AICompanionState) -> str:
    for message in reversed(state["messages"]):
        if message.type == "human":
            return str(message.content)
    return ""


def _needs_router_llm(state: AICompanionState) -> bool:
    """Whether the last user message is worth asking the router model about."""
    text = _last_user_text(state).lower()
    return any(cue in text for cue in _IMAGE_CUES + _AUDIO_CUES + _ACTION_CUES)


def _routine_for(state: AICompanionState, config: RunnableConfig) -> Routine | None:
    """The user's saved routine that their last message triggers, if any."""
    return routine_store.find(get_session_id(config), _last_user_text(state))


async def router_node(state: AICompanionState, config: RunnableConfig):
    # Checked first and in code: "Guten Morgen" looks like small talk, so no word filter
    # or model would ever send it to the action node.
    if _routine_for(state, config) or _ROUTINE_REQUEST.search(_last_user_text(state)):
        return {"workflow": "action"}

    if settings.ROUTER_FAST_PATH and not _needs_router_llm(state):
        return {"workflow": "conversation"}

    chain = get_router_chain()
    try:
        response = await chain.ainvoke({"messages": chat_history(state["messages"])[-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    except Exception:
        # The small model sometimes answers a bare word instead of JSON; a plain
        # reply is the safe default rather than failing the whole message.
        return {"workflow": "conversation"}
    return {"workflow": response.response_type}


def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False
    return {"apply_activity": apply_activity, "current_activity": schedule_context}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": chat_history(state["messages"]),
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


async def image_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(chat_history(state["messages"])[-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(
        content=f"<Bild von Bob angehängt, erzeugt aus dem Prompt: {scenario.image_prompt}>"
    )
    updated_messages = chat_history(state["messages"]) + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


async def audio_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": chat_history(state["messages"]),
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    return {"messages": response, "audio_buffer": output_audio}


async def summarize_conversation_node(state: AICompanionState):
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"Das ist die bisherige Zusammenfassung des Gesprächs zwischen Bob und dem Nutzer: {summary}\n\n"
            "Erweitere die Zusammenfassung unter Berücksichtigung der neuen Nachrichten oben. "
            "Antworte auf Deutsch:"
        )
    else:
        summary_message = (
            "Erstelle eine Zusammenfassung des obigen Gesprächs zwischen Bob und dem Nutzer. "
            "Die Zusammenfassung muss eine kurze Beschreibung des bisherigen Gesprächs sein, "
            "die aber alle relevanten Informationen erfasst, die zwischen Bob und dem Nutzer "
            "ausgetauscht wurden. Antworte auf Deutsch:"
        )

    messages = chat_history(state["messages"]) + \
        [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(
        id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}


async def memory_extraction_node(state: AICompanionState, config: RunnableConfig):
    """Extract and store important information from the last message."""
    if not state["messages"]:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1], get_session_id(config))
    return {}


def memory_injection_node(state: AICompanionState, config: RunnableConfig):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    recent_context = " ".join(
        [m.content for m in chat_history(state["messages"])[-3:]])
    memories = memory_manager.get_relevant_memories(
        recent_context, get_session_id(config))

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}


async def action_node(state: AICompanionState, config: RunnableConfig):
    now = datetime.now(ZoneInfo(settings.TIMEZONE))

    # chat_history strips old tool traffic, so earlier turns look like Bob answered
    # calendar questions without tools. With a long history the model copies that and
    # replies "Erledigt!" without calling anything. Keep only the current request and
    # the exchange before it (enough for follow-ups like "um 15 Uhr").
    messages = chat_history(state["messages"], keep_current_tool_calls=True)
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)

    routine = _routine_for(state, config)
    if routine:
        # A routine is a macro: the model sees the saved steps instead of "Guten Morgen",
        # and no older messages it could answer from instead of calling the tools.
        steps = "\n".join(f"{number}. {step}" for number, step in enumerate(routine.steps, start=1))
        request = HumanMessage(content=ROUTINE_RUN_PROMPT.format(trigger=routine.trigger, steps=steps))
        messages = [request] + messages[last_human + 1:]
    else:
        messages = messages[max(0, last_human - settings.ACTION_PREVIOUS_MESSAGES):]

    # The router only sends real tool requests here, so the first step of a turn must
    # call a tool. Once tool results are in, the model may answer in text. A routine needs
    # one result per step: forcing until then means the model cannot skip a step, and
    # because the count only goes up, it cannot loop forever either.
    tool_results = sum(m.type == "tool" for m in messages)
    results_needed = len(routine.steps) if routine else 1
    chain = get_action_chain(
        "" if routine else state.get("summary", ""),
        force_tool=tool_results < results_needed,
        # A finished routine only has its answer left to write, which saves the tokens
        # of the tool schemas. Without that, nearly every run hits the rate limit.
        use_tools=not routine or tool_results < results_needed,
        manage_routines=any(cue in _last_user_text(state).lower() for cue in _ROUTINE_CUES),
    )

    response = await chain.ainvoke(
        {
            "messages": messages,
            "current_activity": ScheduleContextGenerator.get_current_activity(),
            "memory_context": state.get("memory_context", ""),
            "today": now.strftime("%Y-%m-%d"),
            "weekday": _WEEKDAYS[now.weekday()],
            "time": now.strftime("%H:%M"),
            "timezone": settings.TIMEZONE,
        },
        config,
    )
    return {"messages": response}


# Runs whatever tool calls the LLM asked for and returns the results as ToolMessages
_run_tools = ToolNode(ACTION_TOOLS + ROUTINE_TOOLS)


async def tools_node(state: AICompanionState, config: RunnableConfig):
    """Run the next tool call that has no result yet, one per graph step.

    A routine makes the model ask for several tools at once. As one graph step, every
    "ja" to an interrupt() restarts all of them: with two confirmations in a batch, the
    first calendar entry was created twice. A finished graph step is never repeated.
    """
    next_call = pending_tool_calls(state["messages"])[0]
    return await _run_tools.ainvoke({"messages": [AIMessage(content="", tool_calls=[next_call])]}, config)
