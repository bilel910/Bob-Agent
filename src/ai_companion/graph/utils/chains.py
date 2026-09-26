from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT, ACTION_PROMPT
from ai_companion.graph.utils.helpers import (
    AsteriskRemovalParser,
    get_chat_model,
    get_small_chat_model,
    get_structured_runnable,
)

from ai_companion.modules.calendar.tools import CALENDAR_TOOLS


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="Der Antworttyp für den Nutzer. Er muss einer der folgenden sein: "
        "'conversation', 'image' oder 'audio' oder 'action'"
    )


def get_router_chain():
    model = get_structured_runnable(
        get_small_chat_model(temperature=0.3), RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model


def get_character_response_chain(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nZusammenfassung des bisherigen Gesprächs zwischen Bob und dem Nutzer: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()


def get_action_chain(summary: str = ""):
    # bind_tools sends the tool schemas (the JSON you printed in Lesson 2) with every request
    model = get_chat_model(temperature=0.2).bind_tools(CALENDAR_TOOLS)
    system_message = ACTION_PROMPT

    if summary:
        system_message += f"\n\nZusammenfassung des bisherigen Gesprächs zwischen Bob und dem Nutzer: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model
