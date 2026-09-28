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

    if start_dt < datetime.now(start_dt.tzinfo):
        return f"Error: {start_dt:%d.%m.%Y %H:%M} is in the past. Use the next upcoming date instead."

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
            f"- {start[11:16] or 'all day'}: {event.get('summary', '(no title)')}"
            f"[id: {event['id']}]")
    return f"Events on {day}:\n" + "\n".join(lines)


@tool(parse_docstring=True)
def update_event(
    event_id: str,
    title: str | None = None,
    start: str | None = None,
    duration_minutes: int | None = None,
    description: str | None = None,
) -> str:
    """Change an existing event in the user's Google Calendar.

    Call list_events first to find the event_id. Only pass the fields that should change.

    Args:
        event_id: The id of the event, as shown by list_events.
        title: New title, if it should change.
        start: New start time in ISO format without timezone, e.g. '2026-09-26T14:00'.
        duration_minutes: New length in minutes, if it should change.
        description: New notes, if they should change.
    """
    service = get_calendar_service()

    # 1. Load the current event: we need its old start/end to keep what isn't changing.
    try:
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
    except Exception as e:
        return f"Error: could not find event '{event_id}'. Call list_events first. ({e})"

    old_start = datetime.fromisoformat(event["start"]["dateTime"])
    old_end = datetime.fromisoformat(event["end"]["dateTime"])

    # 2. Work out the new times. Anything not given keeps its old value.
    if start:
        try:
            new_start = _parse_local(start)
        except ValueError:
            return f"Error: '{start}' is not a valid ISO date. Use e.g. '2026-09-26T14:00'."
        if new_start < datetime.now(new_start.tzinfo):
            return f"Error: {new_start:%d.%m.%Y %H:%M} is in the past."
    else:
        new_start = old_start

    duration = timedelta(
        minutes=duration_minutes) if duration_minutes else old_end - old_start
    new_end = new_start + duration

    # 3. Build a *partial* body: patch() only touches the fields we send.
    body = {}
    if title:
        body["summary"] = title
    if description is not None:
        body["description"] = description
    if start or duration_minutes:
        body["start"] = {"dateTime": new_start.isoformat(),
                         "timeZone": settings.TIMEZONE}
        body["end"] = {"dateTime": new_end.isoformat(),
                       "timeZone": settings.TIMEZONE}
    if not body:
        return "Nothing to change: pass at least one of title, start, duration_minutes, description."

    # 4. Ask the user, same as schedule_meeting.
    old_title = event.get("summary", "(ohne Titel)")
    answer = interrupt(
        {
            "question": f"Soll ich '{old_title}' ({old_start:%d.%m.%Y %H:%M}) ändern zu "
            f"'{title or old_title}' am {new_start:%d.%m.%Y} um {new_start:%H:%M} "
            f"({int(duration.total_seconds() // 60)} Min.)? (ja/nein)"
        }
    )
    if str(answer).strip().lower() not in ("ja", "j", "yes", "y", "ok", "okay", "klar", "passt"):
        return f"NICHT geändert. Der Nutzer hat geantwortet: '{answer}'."

    # 5. Send the change. sendUpdates notifies attendees if there are any.
    try:
        updated = service.events().patch(
            calendarId="primary",
            eventId=event_id,
            body=body,
            sendUpdates="all" if event.get("attendees") else "none",
        ).execute()
    except Exception as e:
        return f"Error updating the event: {e}"

    return f"Event updated: '{updated.get('summary')}' at {new_start:%d.%m.%Y %H:%M}. Link: {updated.get('htmlLink')}"


CALENDAR_TOOLS = [schedule_meeting, list_events, update_event]
