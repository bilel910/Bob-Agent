from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

creds = Credentials.from_authorized_user_file(
    "secrets/google_token.json", SCOPES)
service = build("calendar", "v3", credentials=creds)

now = datetime.now(timezone.utc).isoformat()
result = service.events().list(
    calendarId="primary",
    timeMin=now,
    maxResults=5,
    singleEvents=True,    # expand recurring events into single ones
    orderBy="startTime",
).execute()

for event in result.get("items", []):
    start = event["start"].get("dateTime", event["start"].get("date"))
    print(start, "-", event.get("summary", "(no title)"))

events = result.get("items", [])
if not events:
    print("No upcoming events found.")
