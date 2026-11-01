import asyncio
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

from ai_companion.graph.nodes import _WEEKDAYS, _needs_router_llm
from ai_companion.graph.utils.chains import ACTION_TOOLS, get_action_chain, get_router_chain
from ai_companion.modules.image import ImageToText
from ai_companion.modules.tasks.tools import list_tasks
from ai_companion.settings import settings

# Usage: uv run python scripts/test_tasks.py [path/to/photo.jpg]
photo = sys.argv[1] if len(sys.argv) > 1 else "docs/img/handwritten_list_example.jpg"

# Layer ①: can Bob read the photo? (OCR)
description = asyncio.run(ImageToText().analyze_image(photo))
print(description, "\n")

# The interfaces send the photo to the graph as text, exactly like this:
message = HumanMessage(content=f"\n[Bildanalyse: {description}]")

# Layer ②: does the router send it to the action node?
print("filter passes:", _needs_router_llm({"messages": [message]}))
print("router says:", get_router_chain().invoke({"messages": [message]}).response_type, "\n")

# Layer ③: are the tools registered?
print("Tools:", [t.name for t in ACTION_TOOLS], "\n")

# Layer ④: does the model turn the list into one create_tasks call?
now = datetime.now(ZoneInfo(settings.TIMEZONE))
response = get_action_chain(force_tool=True).invoke({
    "messages": [message],
    "current_activity": "", "memory_context": "",
    "today": now.strftime("%Y-%m-%d"), "weekday": _WEEKDAYS[now.weekday()],
    "time": now.strftime("%H:%M"), "timezone": settings.TIMEZONE,
})
for call in response.tool_calls:
    print(call["name"])
    for task in call["args"].get("tasks", []):
        print("  -", task)

# Layer ⑤: can Bob reach Google Tasks? (read only, creates nothing)
print("\n" + list_tasks.invoke({}))
