from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
SECRETS = Path("secrets")

flow = InstalledAppFlow.from_client_secrets_file(
    SECRETS / "google_credentials.json", SCOPES)
creds = flow.run_local_server(port=0)  # opens your browser to log in

(SECRETS / "google_token.json").write_text(creds.to_json())
print("Token saved to secrets/google_token.json")
