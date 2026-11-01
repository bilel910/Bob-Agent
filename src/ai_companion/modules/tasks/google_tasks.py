from functools import lru_cache

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from ai_companion.settings import settings

SCOPES = ["https://www.googleapis.com/auth/tasks"]


@lru_cache(maxsize=1)
def get_tasks_service():
    """Build (and reuse) the Google Tasks client.

    Uses the same token file as the calendar. The token must have been created
    with the tasks scope (scripts/google_auth.py asks for both).
    """
    creds = Credentials.from_authorized_user_file(
        settings.GOOGLE_TOKEN_PATH, SCOPES)
    return build("tasks", "v1", credentials=creds, cache_discovery=False)
