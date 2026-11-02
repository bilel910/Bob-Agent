"""Routines, tested from the inside out.

    uv run python scripts/test_routines.py          # layers 1-2: fake model, costs nothing
    uv run python scripts/test_routines.py --live   # adds layer 3: real model, web search and calendar (read only)
"""

import asyncio
import os
import sys
import tempfile
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from ai_companion.graph import graph_builder, nodes
from ai_companion.graph.utils.chains import (
    ACTION_TOOLS,
    ROUTINE_TOOLS,
    RouterResponse,
    get_action_chain,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import get_pending_question
from ai_companion.modules.routines import store
from ai_companion.modules.routines.tools import save_routine
from ai_companion.settings import settings

# Never touch real routines: everything below runs on a throwaway file.
settings.ROUTINES_DB_PATH = os.path.join(tempfile.mkdtemp(), "routines.db")

failures = []


def check(name: str, passed: bool, detail="") -> None:
    print(f"{'✅' if passed else '❌'} {name}" + (f"   {detail}" if detail else ""))
    if not passed:
        failures.append(name)


# Layer ①: the store, no graph and no model
def test_store() -> None:
    print("\nLayer ①: the store")
    for said in ("Guten Morgen", "guten morgen", "Guten Morgen!", "GUTEN MORGEN, Bob!", "guten morgen."):
        check(f"{said!r} is the trigger 'guten morgen'", store.normalize_trigger(said) == "guten morgen")

    store.save("anna", "Guten Morgen", ["Wetter?", "Termine?"])
    check("found again after saving", store.find("anna", "guten morgen!").steps == ["Wetter?", "Termine?"])
    check("umlauts survive", store.save("anna", "Grüß dich", ["Schönes Wetter?"]) or
          store.find("anna", "grüß dich").steps == ["Schönes Wetter?"])
    check("another user does not get it", store.find("ben", "Guten Morgen") is None)
    check("a longer sentence does not trigger it", store.find("anna", "Guten Morgen, ich bin müde") is None)
    check("an empty message does not trigger anything", store.find("anna", "?!") is None)

    store.save("anna", "guten MORGEN", ["Nur News"])
    check("same trigger replaces", store.find("anna", "Guten Morgen").steps == ["Nur News"])
    check("list shows both routines", len(store.list_all("anna")) == 2)
    store.delete("anna", "GUTEN morgen!")
    check("deleted", store.find("anna", "Guten Morgen") is None)
    check("the other routine is still there", len(store.list_all("anna")) == 1)

    check("the model cannot see or set the user", "config" not in save_routine.args)

    # Chatting must survive a damaged routines file (here: the path is a folder)
    real_path, settings.ROUTINES_DB_PATH = settings.ROUTINES_DB_PATH, tempfile.mkdtemp()
    check("a broken file means 'no routine', not a crash", store.find("anna", "Grüß dich") is None)
    settings.ROUTINES_DB_PATH = real_path

    # The graph must be able to run a routine tool even though normal calls don't offer them
    check("normal calls are not offered the routine tools", not set(t.name for t in ROUTINE_TOOLS) & set(t.name for t in ACTION_TOOLS))
    check("the graph can run the routine tools", all(t.name in nodes._run_tools.tools_by_name for t in ROUTINE_TOOLS))


# Layer ②: the real graph with a scripted model and fake tools
class FakeMemory:
    async def extract_and_store_memories(self, *_):
        pass

    def get_relevant_memories(self, *_):
        return []

    def format_memories_for_prompt(self, _):
        return ""


def ai_calls(*calls: tuple) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{name}_{i}"} for i, (name, args) in enumerate(calls)])


class FakeWorld:
    """Replaces the model and the tools, and records what happened to them."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.chain_calls = []  # one entry per model call: how it was called, what it saw
        self.router_calls = 0
        self.ran = []  # the tools that really ran
        self.written = []  # what the confirming tools really wrote

        @tool
        def fake_search(query: str = "") -> str:
            """Fake web search."""
            self.ran.append(query)
            return f"Ergebnis zu {query}"

        @tool
        def fake_write(what: str) -> str:
            """Fake tool that asks first, like schedule_meeting."""
            if interrupt({"question": f"{what} eintragen? (ja/nein)"}) == "ja":
                self.written.append(what)
            return f"{what} eingetragen"

        def fake_action_chain(summary="", force_tool=False, use_tools=True, manage_routines=False):
            async def run(payload, config=None):
                self.chain_calls.append({
                    "force": force_tool, "tools": use_tools, "manage": manage_routines,
                    "summary": summary, "messages": payload["messages"]})
                return self.replies.pop(0)
            return RunnableLambda(lambda p: None, afunc=run)

        async def router_says_action(_):
            self.router_calls += 1
            return RouterResponse(response_type="action")

        async def reply(_, config=None):
            return "normale Antwort"

        nodes.get_memory_manager = lambda: FakeMemory()
        nodes.get_action_chain = fake_action_chain
        nodes.get_router_chain = lambda: RunnableLambda(lambda p: None, afunc=router_says_action)
        nodes.get_character_response_chain = lambda summary="": RunnableLambda(lambda p: None, afunc=reply)
        nodes._run_tools = ToolNode([fake_search, fake_write, *ACTION_TOOLS, *ROUTINE_TOOLS])


async def say(text: str, user: str, world: FakeWorld, *answers: str):
    """Send a message through the real graph, answering each confirmation in turn."""
    graph = graph_builder.compile(checkpointer=world.memory)
    config = {"configurable": {"thread_id": user}}
    await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config)
    questions = []
    while question := get_pending_question(await graph.aget_state(config)):
        questions.append(question)
        await graph.ainvoke(Command(resume=answers[len(questions) - 1]), config)
    return (await graph.aget_state(config)).values, questions


async def test_graph() -> None:
    print("\nLayer ②: the real graph, scripted model")
    settings.TOTAL_MESSAGES_SUMMARY_TRIGGER = 1000  # summarizing would call the real model
    steps = ["Wie wird das Wetter heute?", "Was steht heute an?", "Was gibt es Neues?"]
    memory = MemorySaver()

    # Saving: Bob shows the steps and saves only after "ja"
    world = FakeWorld(ai_calls(("save_routine", {"trigger": "'Guten Morgen'", "steps": steps})), AIMessage(content="Gespeichert!"))
    world.memory = memory
    _, questions = await say("Wenn ich 'Guten Morgen' sage: Wetter, Termine, News", "anna", world, "ja")
    check("asks before saving", len(questions) == 1 and all(step in questions[0] for step in steps))
    check("saved after 'ja', without the quotes", store.find("anna", "Guten Morgen").trigger == "Guten Morgen")
    check("the save call ran no step", world.ran == [])
    check("the routine rules are sent when talking about routines", world.chain_calls[0]["manage"])
    check("saving does not depend on the router model", world.router_calls == 0, world.router_calls)

    for sentence, is_routine in [
        ("Wenn ich 'Guten Morgen' sage → Wetter + News", True),
        ("Wenn ich Feierabend sage, zeig mir die Bahn", True),
        ("Sobald ich Hallo sage, sag mir das Wetter", True),
        ("Welche Routinen habe ich?", True),
        ("Lösch die Routine Guten Morgen", True),
        ("Wenn ich müde bin, trinke ich Kaffee", False),
        ("Meine Morgenroutine ist langweilig", False),
    ]:
        found = bool(nodes._ROUTINE_REQUEST.search(sentence))
        check(f"{sentence!r} {'goes straight to the tools' if is_routine else 'is left to the router'}", found == is_routine)

    world = FakeWorld(ai_calls(("save_routine", {"trigger": "Feierabend", "steps": ["News"]})), AIMessage(content="Ok."))
    world.memory = memory
    await say("Wenn ich Feierabend sage: News", "anna", world, "nein")
    check("not saved after 'nein'", store.find("anna", "Feierabend") is None)

    too_many = [f"Schritt {i}" for i in range(settings.ROUTINE_MAX_STEPS + 1)]
    world = FakeWorld(ai_calls(("save_routine", {"trigger": "Zu viel", "steps": too_many})), AIMessage(content="Zu viele."))
    world.memory = memory
    _, questions = await say("Wenn ich 'Zu viel' sage: ...", "anna", world)
    check("too many steps are refused without asking", store.find("anna", "Zu viel") is None and not questions)

    # Listing and deleting
    store.save("erik", "Tschüss", ["Wetter?"])
    world = FakeWorld(ai_calls(("list_routines", {})), AIMessage(content="Hier."))
    world.memory = memory
    state, _ = await say("Welche Routinen habe ich?", "erik", world)
    listed = next(m.content for m in reversed(state["messages"]) if m.type == "tool")
    check("the list shows this user's routine", "Tschüss" in listed and "Wetter?" in listed and "Guten Morgen" not in listed, listed)

    world = FakeWorld(ai_calls(("delete_routine", {"trigger": "Tschüss"})), AIMessage(content="Bleibt."))
    world.memory = memory
    await say("Lösch die Routine Tschüss", "erik", world, "nein")
    check("kept after 'nein'", store.find("erik", "Tschüss") is not None)
    world = FakeWorld(ai_calls(("delete_routine", {"trigger": "Tschüss"})), AIMessage(content="Weg."))
    world.memory = memory
    await say("Lösch die Routine Tschüss", "erik", world, "ja")
    check("deleted after 'ja'", store.find("erik", "Tschüss") is None)

    # Running: the model plans everything at once, the normal case
    searches = [("fake_search", {"query": step}) for step in steps]
    world = FakeWorld(ai_calls(*searches), AIMessage(content="Dein Morgen: ..."))
    world.memory = memory
    state, _ = await say("Guten Morgen!", "anna", world)
    check("every step ran, in order", world.ran == steps, world.ran)
    check("two model calls in total", len(world.chain_calls) == 2, len(world.chain_calls))
    first, last = world.chain_calls[0], world.chain_calls[-1]
    check("the model saw the steps instead of 'Guten Morgen!'",
          all(step in first["messages"][0].content for step in steps))
    check("first call must call tools", first["force"] and first["tools"])
    check("the answer call has no tool schemas", not last["force"] and not last["tools"])
    check("running a routine does not pay for the routine rules", not first["manage"] and not last["manage"])
    check("the steps are not stored as chat messages",
          not any("Routine" in str(m.content) for m in state["messages"] if m.type == "human"))
    check("the final message is the answer", state["messages"][-1].content == "Dein Morgen: ...")

    # Running: the model forgets a step and is forced to go back for it
    world = FakeWorld(ai_calls(*searches[:2]), ai_calls(searches[2]), AIMessage(content="Fertig."))
    world.memory = memory
    await say("Guten Morgen", "anna", world)
    check("a skipped step still runs", world.ran == steps, world.ran)
    check("still forced while a step is missing", world.chain_calls[1]["force"] and world.chain_calls[1]["tools"])

    # A trigger with a word the router filter reacts to ("bahn") still skips the router model
    store.save("anna", "Bahn", ["Wann fährt meine Bahn?"])
    world = FakeWorld(ai_calls(("fake_search", {"query": "bahn"})), AIMessage(content="Um 8."))
    world.memory = memory
    await say("Bahn", "anna", world)
    check("a trigger never costs a router call", world.router_calls == 0 and world.ran == ["bahn"], world.router_calls)

    # Running: other users and other sentences are not affected
    world = FakeWorld()
    world.memory = memory
    state, _ = await say("Guten Morgen", "ben", world)
    check("another user just gets a normal chat", state["messages"][-1].content == "normale Antwort" and not world.chain_calls)
    world = FakeWorld()
    world.memory = memory
    state, _ = await say("Guten Morgen, wie geht's?", "anna", world)
    check("a longer sentence is a normal chat", state["messages"][-1].content == "normale Antwort" and not world.chain_calls)

    # The bug that sequential tools fix: two confirmations in one batch
    store.save("carla", "Feierabend", ["Sport eintragen", "Lesen eintragen"])
    world = FakeWorld(ai_calls(("fake_write", {"what": "Sport"}), ("fake_write", {"what": "Lesen"})), AIMessage(content="Beides drin."))
    world.memory = memory
    _, questions = await say("Feierabend", "carla", world, "ja", "ja")
    check("each write asks once", len(questions) == 2, questions)
    check("nothing is written twice", world.written == ["Sport", "Lesen"], world.written)

    # One "nein" in the middle does not stop the rest
    world = FakeWorld(ai_calls(("fake_write", {"what": "Sport"}), ("fake_write", {"what": "Lesen"})), AIMessage(content="Nur Lesen."))
    world.memory = memory
    await say("Feierabend", "carla", world, "nein", "ja")
    check("'nein' only skips that write", world.written == ["Lesen"], world.written)

    # The longest possible run: most steps, and a model that calls one tool at a time.
    store.save("dora", "Los", [f"Schritt {i}" for i in range(settings.ROUTINE_MAX_STEPS)])
    world = FakeWorld(*[ai_calls(("fake_search", {"query": str(i)})) for i in range(settings.ROUTINE_MAX_STEPS)], AIMessage(content="Fertig."))
    world.memory = memory
    state, _ = await say("Los", "dora", world)
    check("the longest run stays inside LangGraph's step limit", state["messages"][-1].content == "Fertig.")

    # Normal action turns must behave as before
    world = FakeWorld(ai_calls(("fake_search", {"query": "x"})), AIMessage(content="Erledigt."))
    world.memory = memory
    await say("Wie wird das Wetter?", "anna", world)
    first, last = world.chain_calls[0], world.chain_calls[-1]
    check("normal action: first call forced, answer call keeps its tools",
          first["force"] and first["tools"] and not last["force"] and last["tools"])
    check("normal action: does not pay for the routine rules", not first["manage"] and not last["manage"])


# Layer ③: the real model. Only reads: confirmations are never answered.
def chain_input(messages: list) -> dict:
    now = datetime.now(ZoneInfo(settings.TIMEZONE))
    return {
        "messages": messages, "current_activity": "", "memory_context": "",
        "today": now.strftime("%Y-%m-%d"), "weekday": nodes._WEEKDAYS[now.weekday()],
        "time": now.strftime("%H:%M"), "timezone": settings.TIMEZONE,
    }


def test_live() -> None:
    print("\nLayer ③: the real model")
    router = get_router_chain()
    for text, expected in [
        ("Wenn ich 'Guten Morgen' sage, zeig mir Wetter, Termine und News", "action"),
        ("Welche Routinen habe ich?", "action"),
        ("Lösch die Routine Guten Morgen", "action"),
        ("Wenn ich müde bin, trinke ich Kaffee", "conversation"),
    ]:
        said = router.invoke({"messages": [HumanMessage(content=text)]}).response_type
        check(f"router: {text!r} -> {expected}", said == expected, said)

    text = "Wenn ich 'Guten Morgen' sage → Wetter + Termine + News"
    response = get_action_chain(force_tool=True, manage_routines=True).invoke(chain_input([HumanMessage(content=text)]))
    calls = response.tool_calls
    check("the sentence becomes one save_routine call", [c["name"] for c in calls] == ["save_routine"], [c["name"] for c in calls])
    print("   trigger:", calls[0]["args"].get("trigger"))
    for step in calls[0]["args"].get("steps", []):
        print("   step:", step)


async def run_live() -> None:
    """A full run on the real model, with the real read-only tools."""
    nodes._run_tools = ToolNode(ACTION_TOOLS + ROUTINE_TOOLS)
    nodes.get_action_chain = get_action_chain
    nodes.get_memory_manager = lambda: FakeMemory()

    steps = ["Wie wird das Wetter heute in Berlin?", "Was steht heute in meinem Kalender?", "Was sind heute die wichtigsten Nachrichten?"]
    store.save("live", "Guten Morgen", steps)
    graph = graph_builder.compile(checkpointer=MemorySaver())
    started = time.time()
    await graph.ainvoke({"messages": [HumanMessage(content="Guten Morgen!")]}, {"configurable": {"thread_id": "live"}})
    state = await graph.aget_state({"configurable": {"thread_id": "live"}})
    tools = [c["name"] for m in state.values["messages"] if m.type == "ai" for c in m.tool_calls]
    tokens = [m.usage_metadata["total_tokens"] for m in state.values["messages"] if m.type == "ai" and m.usage_metadata]

    print(f"\n   tools: {tools}\n   tokens per model call: {tokens}, took {time.time() - started:.0f}s")
    print(f"\n   Bob: {state.values['messages'][-1].content}\n")
    check("one tool per step at least", len(tools) >= len(steps), tools)
    check("two model calls", len(tokens) == 2, tokens)
    check("the whole run fits the 8000 tokens/min", sum(tokens) <= 8000, sum(tokens))


test_store()
asyncio.run(test_graph())
if "--live" in sys.argv:
    test_live()
    asyncio.run(run_live())

print(f"\n{'❌ ' + str(len(failures)) + ' failed: ' + ', '.join(failures) if failures else '✅ all passed'}")
sys.exit(1 if failures else 0)
