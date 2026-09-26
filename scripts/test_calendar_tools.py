import json

from ai_companion.modules.calendar.tools import list_events, schedule_meeting

# 1. What the LLM will "see" about the tool
print(schedule_meeting.name)
print(json.dumps(schedule_meeting.tool_call_schema.model_json_schema(), indent=2))

# 2. Create a meeting (no attendees, so no emails are sent)
print(schedule_meeting.invoke(
    {"title": "Bob Test", "start": "2026-09-28T15:00", "duration_minutes": 45}))

# 3. Read it back
print(list_events.invoke({"day": "2026-09-28"}))

# 4. Test the error handling
print(schedule_meeting.invoke({"title": "Broken", "start": "next monday"}))
