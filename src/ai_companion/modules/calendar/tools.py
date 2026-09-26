from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from ai_companion.modules.calendar.google_calendar import get_calendar_service
from ai_companion.settings import settings
from langgraph.types import interrupt


def _parse_local(value: str) -> datetime:
    """Parse '2026-09-26T14:00' and attach the user's timezone if none is given."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(settings.TIMEZONE))
    return dt


@tool(parse_docstring=True)
def schedule_meeting(
    title: str,
    start: str,
    duration_minutes: int = 30,
    attendees: list[str] | None = None,
    description: str = "",
) -> str:
    """Create a meeting in the user's Google Calendar.

    Args:
        title: Short title of the meeting, e.g. 'Team Sync'.
        start: Start time in ISO format without timezone, e.g. '2026-09-26T14:00'.
        duration_minutes: Length of the meeting in minutes.
        attendees: Email addresses of people to invite.
        description: Optional notes for the meeting.
    """
    try:
        start_dt = _parse_local(start)
    except ValueError:
        return f"Error: '{start}' is not a valid ISO date. Use e.g. '2026-09-26T14:00'."

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": settings.TIMEZONE},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": settings.TIMEZONE},
        "attendees": [{"email": email} for email in attendees or []],
    }

    who = f" mit {', '.join(attendees)}" if attendees else ""
    answer = interrupt(
        {
            "question": f"Soll ich '{title}' am {start_dt:%d.%m.%Y} um {start_dt:%H:%M} "
            f"({duration_minutes} Min.){who} in deinen Kalender eintragen? (ja/nein)"
        }
    )
    if str(answer).strip().lower() not in ("ja", "j", "yes", "y", "ok", "okay", "klar", "passt"):
        return (
            f"NICHT eingetragen. Der Nutzer hat geantwortet: '{answer}'. "
            "Wenn er etwas ändern möchte (z. B. eine andere Uhrzeit), plane den Termin mit den neuen Angaben neu."
        )

    try:
        event = get_calendar_service().events().insert(
            calendarId="primary",
            body=body,
            sendUpdates="all" if attendees else "none",  # email invites to attendees
        ).execute()
    except Exception as e:
        return f"Error creating the meeting: {e}"

    return f"Meeting '{title}' created for {start_dt:%d.%m.%Y %H:%M}. Link: {event.get('htmlLink')}"


@tool(parse_docstring=True)
def list_events(day: str) -> str:
    """List the user's calendar events for one day.

    Args:
        day: The day in ISO format, e.g. '2026-09-26'.
    """
    try:
        start_dt = _parse_local(day)
    except ValueError:
        return f"Error: '{day}' is not a valid date. Use e.g. '2026-09-26'."
    end_dt = start_dt + timedelta(days=1)

    try:
        result = get_calendar_service().events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
    except Exception as e:
        return f"Error reading the calendar: {e}"

    events = result.get("items", [])
    if not events:
        return f"No events on {day}."

    lines = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        lines.append(
            f"- {start[11:16] or 'all day'}: {event.get('summary', '(no title)')}")
    return f"Events on {day}:\n" + "\n".join(lines)


CALENDAR_TOOLS = [schedule_meeting, list_events]
