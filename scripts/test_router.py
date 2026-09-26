import asyncio

from langchain_core.messages import HumanMessage

from ai_companion.graph.nodes import _needs_router_llm
from ai_companion.graph.utils.chains import get_router_chain

TESTS = [
    ("Wie geht's dir heute?", "conversation"),
    ("Plane ein Meeting morgen um 14 Uhr mit Anna", "action"),
    ("Was steht morgen in meinem Kalender?", "action"),
    ("Trag mir einen Zahnarzttermin am Freitag um 9 ein", "action"),
    ("Mein Meeting heute war echt anstrengend", "conversation"),   # the tricky one
    ("Schick mir ein Foto von dir", "image"),
    ("Sag es mir mit deiner Stimme", "audio"),
]


async def main():
    chain = get_router_chain()
    for text, expected in TESTS:
        messages = [HumanMessage(content=text)]
        if _needs_router_llm({"messages": messages}):
            result = (await chain.ainvoke({"messages": messages})).response_type
            source = "LLM "
        else:
            result = "conversation"
            source = "fast"
        mark = "✅" if result == expected else "❌"
        print(
            f"{mark} [{source}] {text!r:55} → {result} (expected {expected})")


asyncio.run(main())
