import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langgraph.types import interrupt

from ai_companion.modules.tasks.google_tasks import get_tasks_service
from ai_companion.settings import settings

_WEEKDAYS = ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]
_RELATIVE_DAYS = {"heute": 0, "morgen": 1, "übermorgen": 2}


def _due_date(task: str, today: date) -> date | None:
    """Find a due date written in the task, e.g. 'bis 15.11.', 'Freitag' or 'morgen'.

    Done in code rather than by the model: asking the model for a separate date
    field per task made it break the tool call on almost every list with a date.
    """
    text = task.lower()
    if match := re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})?", text):
        day, month, year = int(match[1]), int(match[2]), match[3]
        try:
            if year:
                return date(int(year) + (2000 if len(year) == 2 else 0), month, day)
            due = date(today.year, month, day)
            # No year on the list: a date already past means next year.
            return due if due >= today else date(today.year + 1, month, day)
        except ValueError:  # e.g. 31.02.
            return None
    for word, days in _RELATIVE_DAYS.items():
        if re.search(rf"\b{word}\b", text):
            return today + timedelta(days=days)
    for weekday, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", text):
            return today + timedelta(days=(weekday - today.weekday()) % 7)
    return None


def _google_error(e: Exception) -> str:
    """Turn a Google API error into a message Bob can pass on to the user."""
    text = str(e)
    # invalid_scope: the token predates the tasks scope. invalid_grant: the token expired
    # (Google does this after 7 days while the OAuth app is in "Testing" mode).
    if "scope" in text.lower() or "invalid_grant" in text:
        return ("Error: Bob has no (valid) access to Google Tasks. The user must run "
                "'uv run python scripts/google_auth.py' once and allow access to Tasks.")
    if "has not been used" in text or "is disabled" in text:
        return "Error: the Google Tasks API is not enabled in the Google Cloud project."
    return f"Error talking to Google Tasks: {text}"


def _find_list(service, list_name: str | None) -> dict | None:
    """The task list called list_name, or the default list ('My Tasks') if no name is given."""
    if not list_name:
        return service.tasklists().get(tasklist="@default").execute()
    lists = service.tasklists().list(maxResults=100).execute().get("items", [])
    return next((tl for tl in lists if tl["title"].casefold() == list_name.casefold()), None)


@tool(parse_docstring=True)
def create_tasks(tasks: list[str], list_name: str | None = None) -> str:
    """Add one or more tasks to the user's Google Tasks, e.g. every line of a photographed to-do list.

    Put all tasks into ONE call. Keep dates in the text (e.g. 'Steuererklärung bis 15.11.'):
    the due date is read from it automatically. The user is asked to confirm before anything is saved.

    Args:
        tasks: One entry per task, as written on the list but without bullet points, in the same order.
        list_name: Name of the task list, e.g. 'Einkaufsliste'. Leave empty for the user's default list.
    """
    tasks = [t.strip(" -•*\t") for t in tasks if t.strip(" -•*\t")]
    if not tasks:
        return "Error: no tasks given. Pass at least one task."

    today = datetime.now(ZoneInfo(settings.TIMEZONE)).date()
    due_dates = [_due_date(t, today) for t in tasks]

    # Look up the list first: this also checks that Bob has access, before we ask the user anything.
    try:
        service = get_tasks_service()
        task_list = _find_list(service, list_name)
    except Exception as e:
        return _google_error(e)

    # Show the user exactly what was read from the photo: handwriting is easy to misread.
    lines = "\n".join(
        f"- {t}" + (f" (fällig {due:%d.%m.%Y})" if due else "") for t, due in zip(tasks, due_dates))
    where = f"in die Liste '{list_name}'" if list_name else "in deine Aufgaben"
    answer = interrupt(
        {"question": f"Soll ich diese {len(tasks)} Aufgaben {where} eintragen?\n{lines}\n(ja/nein)"}
    )
    if str(answer).strip().lower() not in ("ja", "j", "yes", "y", "ok", "okay", "klar", "passt"):
        return (
            f"NICHT eingetragen. Der Nutzer hat geantwortet: '{answer}'. "
            "Wenn er etwas korrigieren möchte (z. B. einen falsch gelesenen Punkt), ruf create_tasks mit der korrigierten Liste neu auf."
        )

    created = []
    try:
        if task_list is None:
            task_list = service.tasklists().insert(body={"title": list_name}).execute()

        # Google puts a new task at the top, so each task goes after the previous one
        # to keep the order of the paper list.
        previous = None
        for task, due in zip(tasks, due_dates):
            body = {"title": task}
            if due:
                body["due"] = f"{due.isoformat()}T00:00:00.000Z"  # Google Tasks only stores the date
            new = service.tasks().insert(tasklist=task_list["id"], body=body, previous=previous).execute()
            previous = new["id"]
            created.append(task)
    except Exception as e:
        done = f" These were already created: {', '.join(created)}." if created else ""
        return _google_error(e) + done

    return f"{len(created)} tasks created in '{task_list['title']}': {', '.join(created)}."


@tool(parse_docstring=True)
def list_tasks(list_name: str | None = None) -> str:
    """List the user's open (not completed) tasks.

    Args:
        list_name: Name of the task list, e.g. 'Einkaufsliste'. Leave empty for the user's default list.
    """
    try:
        service = get_tasks_service()
        task_list = _find_list(service, list_name)
        if task_list is None:
            return f"There is no task list called '{list_name}'."
        items = service.tasks().list(
            tasklist=task_list["id"], showCompleted=False, maxResults=100).execute().get("items", [])
    except Exception as e:
        return _google_error(e)

    if not items:
        return f"No open tasks in '{task_list['title']}'."

    lines = []
    for item in sorted(items, key=lambda t: t.get("position", "")):
        due = f" (due {item['due'][:10]})" if item.get("due") else ""
        lines.append(f"- {item['title']}{due}")
    return f"Open tasks in '{task_list['title']}':\n" + "\n".join(lines)


TASK_TOOLS = [create_tasks, list_tasks]
