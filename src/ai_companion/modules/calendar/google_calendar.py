from functools import lru_cache

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from ai_companion.settings import settings

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


@lru_cache(maxsize=1)
def get_calendar_service():
    """Build (and reuse) the Google Calendar client.

    The access token expires after an hour, but the client refreshes it
    automatically using the refresh token stored in the file.
    """
    creds = Credentials.from_authorized_user_file(
        settings.GOOGLE_TOKEN_PATH, SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)
