# import json

# from ai_companion.modules.transport.tools import plan_journey

# print(json.dumps(plan_journey.args_schema.model_json_schema(), indent=2))
# print(plan_journey.invoke(
#     {"origin": "Alexanderplatz, Berlin", "destination": "Flughafen BER, Berlin"}))


# from ai_companion.modules.transport.tools import _find_place

# for query in ["Alexanderplatz, Berlin", "Flughafen BER, Berlin", "Kreuzberg, Berlin", "xqzzvy"]:
#     place = _find_place(query)
#     print(query, "→", place and (
#         place["type"], place["name"], place["lat"], place["lon"]))

# from ai_companion.modules.transport.tools import plan_journey

# tests = [
#     {"origin": "Alexanderplatz, Berlin", "destination": "Flughafen BER, Berlin"},
#     {"origin": "Kreuzberg, Berlin", "destination": "Potsdam Hauptbahnhof", "time": "2026-10-05T09:00", "arrive_by": True},
#     {"origin": "xqzzvy", "destination": "Berlin"},
#     {"origin": "Alexanderplatz, Berlin", "destination": "Flughafen BER, Berlin", "time": "next friday"},
# ]
# for args in tests:
#     print(args, "\n→", plan_journey.invoke(args)[:400], "\n")


from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from ai_companion.graph.nodes import _needs_router_llm
from ai_companion.graph.utils.chains import ACTION_TOOLS, get_action_chain
from ai_companion.settings import settings

messages = [
    "Wie komme ich vom Alexanderplatz zum BER?",
    "Ich muss morgen um 9 in Potsdam am Hauptbahnhof sein, ich wohne in Kreuzberg. Wann muss ich los?",
    "Wie komme ich zum Flughafen?",
    "Die Bahn war heute wieder zu spät",
]

# Layer ④: is the tool registered?
print("Tools:", [t.name for t in ACTION_TOOLS], "\n")

now = datetime.now(ZoneInfo(settings.TIMEZONE))
chain = get_action_chain()
for text in messages:
    # Layer ①: does the word filter let it through?
    passes = _needs_router_llm({"messages": [HumanMessage(content=text)]})

    # Layer ③: what does the action LLM decide to do?
    response = chain.invoke({
        "messages": [HumanMessage(content=text)],
        "current_activity": "", "memory_context": "",
        "today": now.strftime("%Y-%m-%d"), "weekday": "Samstag",
        "time": now.strftime("%H:%M"), "timezone": settings.TIMEZONE,
    })
    print(f"{text}\n  filter passes: {passes}")
    print(
        f"  tool calls: {response.tool_calls or '— none, text: ' + str(response.content)[:120]}\n")
