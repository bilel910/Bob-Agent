import asyncio

from langchain_core.messages import HumanMessage

from ai_companion.graph.nodes import action_node, tools_node

TESTS = [
    "Was steht am Montag in meinem Kalender?",
    "Plane ein Meeting 'test 4' am Montag um 16 Uhr für eine Stunde",
    # "Trag mir einen Termin am Dienstag ein",   # no time given → Bob should ask
]


async def run(text: str):
    print(f"\n👤 {text}")
    state = {"messages": [HumanMessage(
        content=text)], "summary": "", "memory_context": ""}

    while True:
        ai_message = (await action_node(state, {}))["messages"]
        state["messages"].append(ai_message)

        if not ai_message.tool_calls:           # plain text → done
            print(f"🤖 {ai_message.content}")
            return

        for call in ai_message.tool_calls:       # the LLM's "request"
            print(f"🔧 {call['name']}({call['args']})")

        tool_result = await tools_node.ainvoke({"messages": state["messages"]})
        for msg in tool_result["messages"]:      # what the tool returned
            print(f"📋 {msg.content}")
        state["messages"].extend(tool_result["messages"])


async def main():
    for text in TESTS:
        await run(text)


asyncio.run(main())
