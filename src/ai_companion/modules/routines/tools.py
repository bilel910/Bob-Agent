import sqlite3

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from ai_companion.core.session import get_session_id
from ai_companion.modules.routines import store
from ai_companion.settings import settings


def _numbered(steps: list[str]) -> str:
    return "\n".join(f"{number}. {step}" for number, step in enumerate(steps, start=1))


@tool(parse_docstring=True)
def save_routine(trigger: str, steps: list[str], config: RunnableConfig) -> str:
    """Save a routine: when the user says the trigger, Bob runs all the steps.

    Only saves, does not run the steps. Saving under an existing trigger replaces that
    routine. The user is asked to confirm first.

    Args:
        trigger: What the user will say, without quotes, e.g. 'Guten Morgen'.
        steps: One complete request per step, e.g. 'Wie wird das Wetter heute?'.
    """
    steps = [s.strip(" -•*\t") for s in steps if s.strip(" -•*\t")]
    if not store.normalize_trigger(trigger):
        return "Error: the trigger has no words in it. Ask the user what the trigger should be."
    if not steps:
        return "Error: no steps given. Pass at least one step."
    if len(steps) > settings.ROUTINE_MAX_STEPS:
        return f"Error: a routine can have at most {settings.ROUTINE_MAX_STEPS} steps. Ask the user which ones to keep."

    session_id = get_session_id(config)
    trigger = trigger.strip(" '\"„“”‚‘’")
    replaces = " Das ersetzt deine bisherige Routine dafür." if store.find(session_id, trigger) else ""
    # A saved routine keeps running for good, so a misread step (or one smuggled in by a
    # web page or photo) must not get in without the user having seen it.
    answer = interrupt(
        {"question": f"Soll ich diese Routine speichern?{replaces}\n"
                     f"Wenn du '{trigger}' sagst, mache ich:\n{_numbered(steps)}\n\n(ja/nein)"}
    )
    if str(answer).strip().lower() not in ("ja", "j", "yes", "y", "ok", "okay", "klar", "passt"):
        return (
            f"NICHT gespeichert. Der Nutzer hat geantwortet: '{answer}'. "
            "Wenn er etwas ändern möchte, ruf save_routine mit den neuen Angaben neu auf."
        )

    try:
        store.save(session_id, trigger, steps)
    except (sqlite3.Error, OSError) as e:
        return f"Error saving the routine: {e}"
    return f"Routine saved. When the user says '{trigger}', Bob runs these {len(steps)} steps."


@tool
def list_routines(config: RunnableConfig) -> str:
    """List the user's saved routines with their triggers and steps."""
    try:
        routines = store.list_all(get_session_id(config))
    except (sqlite3.Error, OSError) as e:
        return f"Error reading the routines: {e}"

    if not routines:
        return "The user has no routines yet."
    return "\n\n".join(f"Trigger '{r.trigger}':\n{_numbered(r.steps)}" for r in routines)


@tool(parse_docstring=True)
def delete_routine(trigger: str, config: RunnableConfig) -> str:
    """Delete one of the user's saved routines. The user is asked to confirm first.

    Args:
        trigger: The trigger of the routine, e.g. 'Guten Morgen'.
    """
    session_id = get_session_id(config)
    routine = store.find(session_id, trigger)
    if routine is None:
        return f"There is no routine with the trigger '{trigger}'. Call list_routines to see them."

    answer = interrupt({"question": f"Soll ich die Routine '{routine.trigger}' löschen? (ja/nein)"})
    if str(answer).strip().lower() not in ("ja", "j", "yes", "y", "ok", "okay", "klar", "passt"):
        return f"NICHT gelöscht. Der Nutzer hat geantwortet: '{answer}'."

    try:
        store.delete(session_id, routine.trigger)
    except (sqlite3.Error, OSError) as e:
        return f"Error deleting the routine: {e}"
    return f"Routine '{routine.trigger}' deleted."


ROUTINE_TOOLS = [save_routine, list_routines, delete_routine]
