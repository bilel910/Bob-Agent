from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from ai_companion.graph.nodes import _needs_router_llm
from ai_companion.graph.utils.chains import ACTION_TOOLS, get_action_chain
from ai_companion.modules.web_search.tools import web_search
from ai_companion.settings import settings

# Layer ②: does the tool itself work?
print(web_search.invoke({"query": "Wetter Berlin morgen"}), "\n")

# Layer ④: is it registered?
print("Tools:", [t.name for t in ACTION_TOOLS], "\n")

messages = [
    "Wie wird das Wetter morgen in Berlin?",       # should search
    "Was gibt's Neues bei OpenAI?",                # should search, topic=news
    "Hat der Rewe am Alex sonntags offen?",        # should search
    # should NOT search (general knowledge)
    "Was ist eigentlich ein Transformer?",
    # should be plan_journey, not web_search!
    "Wie komme ich zum BER?",
]
now = datetime.now(ZoneInfo(settings.TIMEZONE))
chain = get_action_chain()
for text in messages:
    passes = _needs_router_llm(
        {"messages": [HumanMessage(content=text)]})   # Layer ①
    response = chain.invoke({                                                # Layer ③
        "messages": [HumanMessage(content=text)],
        "current_activity": "", "memory_context": "",
        "today": now.strftime("%Y-%m-%d"), "weekday": "Mittwoch",
        "time": now.strftime("%H:%M"), "timezone": settings.TIMEZONE,
    })
    print(f"{text}\n  filter passes: {passes}")
    print(
        f"  tool calls: {response.tool_calls or '— none, text: ' + str(response.content)[:120]}\n")
