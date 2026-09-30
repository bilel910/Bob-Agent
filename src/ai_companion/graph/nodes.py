import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
    get_action_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_session_id,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings
from datetime import datetime
from zoneinfo import ZoneInfo

from langgraph.prebuilt import ToolNode

from ai_companion.modules.calendar.tools import CALENDAR_TOOLS


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

_ACTION_CUES = (
    "termin", "meeting", "treffen", "kalender", "calendar", "schedule",
    "plan", "eintrag", "verabred", "was steht", "call",
    "verschieb", "änder", "aender", "umbuch", "verleg", "move", "reschedul", "update",
    "lösch", "loesch", "entfern", "absag", "stornier", "annullier",
    "delete", "remove", "cancel", "cancell",
    "frei", "zeit für", "lücke", "verfügbar", "slot", "free", "available",

)


_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch",
             "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _needs_router_llm(state: AICompanionState) -> bool:
    """Whether the last user message is worth asking the router model about."""
    for message in reversed(state["messages"]):
        if message.type != "human":
            continue
        text = str(message.content).lower()
        return any(cue in text for cue in _IMAGE_CUES + _AUDIO_CUES + _ACTION_CUES)
    return False


async def router_node(state: AICompanionState):
    if settings.ROUTER_FAST_PATH and not _needs_router_llm(state):
        return {"workflow": "conversation"}

    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
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
            "messages": state["messages"],
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

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(
        content=f"<Bild von Bob angehängt, erzeugt aus dem Prompt: {scenario.image_prompt}>"
    )
    updated_messages = state["messages"] + [scenario_message]

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
            "messages": state["messages"],
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

    messages = state["messages"] + [HumanMessage(content=summary_message)]
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
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(
        recent_context, get_session_id(config))

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}


async def action_node(state: AICompanionState, config: RunnableConfig):
    now = datetime.now(ZoneInfo(settings.TIMEZONE))
    chain = get_action_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
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
tools_node = ToolNode(CALENDAR_TOOLS)
