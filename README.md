# Bob – your German-speaking AI companion

Bob is a multimodal AI companion built with [LangGraph](https://github.com/langchain-ai/langgraph). You can chat with him in the browser (Chainlit) or on WhatsApp. He replies in German with text, voice messages or generated images, remembers what you tell him, and can read and write your Google Calendar.

## Features

- **Conversation**: natural German replies, with a daily schedule that gives Bob a "current activity" for context.
- **Voice**: understands voice messages (Groq Whisper) and answers with speech (ElevenLabs).
- **Images**: generates images on request (Together FLUX, with a Pollinations fallback).
- **Memory**
  - Short-term: the conversation state is checkpointed in SQLite and summarized automatically once it gets long.
  - Long-term: facts about you are extracted and stored in Qdrant, then recalled when they are relevant.
- **Google Calendar**: lists your events for a day and schedules meetings. Bob asks for confirmation before he creates an event (human-in-the-loop via LangGraph `interrupt()`).

## How it works

Each message runs through a LangGraph workflow:

```
START ─┬─> memory_extraction_node ─┐
       └─> router_node ────────────┴─> context_injection_node ─> memory_injection_node
                                                                        │
             ┌──────────────┬──────────────┬────────────────────────────┤
             v              v              v                            v
     conversation_node  image_node    audio_node                  action_node <──┐
                                                                        │        │
                                                                        └─> tools_node
             └──────────── summarize_conversation_node (if needed) ─> END
```

- **router_node** picks one of `conversation`, `image`, `audio` or `action`.
- **action_node** is an LLM with the calendar tools (`schedule_meeting`, `list_events`) bound. It loops through `tools_node` until it produces a text answer.

## Tech stack

| Purpose             | Service / library                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| LLM, speech-to-text | [Groq](https://groq.com) (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `whisper-large-v3-turbo`)       |
| Text-to-speech      | [ElevenLabs](https://elevenlabs.io) (`eleven_flash_v2_5`)                                           |
| Image generation    | [Together AI](https://together.ai) (`FLUX.2-dev`), [Pollinations](https://pollinations.ai) fallback |
| Long-term memory    | [Qdrant](https://qdrant.tech) + `sentence-transformers`                                             |
| Short-term memory   | SQLite (LangGraph checkpointer)                                                                     |
| Calendar            | Google Calendar API                                                                                 |
| Interfaces          | [Chainlit](https://chainlit.io) (web), FastAPI webhook (WhatsApp Cloud API)                         |

## Project structure

```
src/ai_companion/
├── core/              # prompts, Bob's daily schedule, exceptions
├── graph/             # LangGraph state, nodes, edges and graph definition
├── interfaces/
│   ├── chainlit/      # web chat UI
│   └── whatsapp/      # WhatsApp webhook (FastAPI)
├── modules/
│   ├── calendar/      # Google Calendar client and tools
│   ├── image/         # text-to-image, image-to-text
│   ├── memory/        # long-term memory (Qdrant)
│   ├── schedules/     # current-activity context
│   └── speech/        # speech-to-text, text-to-speech
└── settings.py        # all configuration (pydantic-settings)
scripts/               # Google OAuth setup and manual test scripts
notebooks/             # router and character-card experiments
docs/                  # setup guides
```

## Getting started

### Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- API keys for Groq, ElevenLabs and Together AI
- Optional: a WhatsApp Business app (Meta) and a Google Cloud project with the Calendar API enabled

### 1. Install dependencies

```bash
uv venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

uv pip install -e .
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in the values in `.env`:

| Variable                                                              | Description                                                                                      |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `GROQ_API_KEY`                                                        | Groq API key                                                                                     |
| `ELEVENLABS_API_KEY`                                                  | ElevenLabs API key                                                                               |
| `ELEVENLABS_VOICE_ID`                                                 | Voice ID. Pick a multilingual or German voice so Bob does not sound English.                     |
| `TOGETHER_API_KEY`                                                    | Together AI API key. Image models require "third-party data sharing" to be enabled for your org. |
| `QDRANT_URL`, `QDRANT_API_KEY`                                        | Only needed for a hosted Qdrant. Docker Compose points to the local container.                   |
| `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_TOKEN`, `WHATSAPP_VERIFY_TOKEN` | Only needed for the WhatsApp interface                                                           |

Other options (models, timezone, memory limits) are in [settings.py](src/ai_companion/settings.py) and can be overridden in `.env`. The default timezone is `Europe/Berlin`.

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for screenshots showing where to find each API key.

### 3. Connect Google Calendar (optional)

1. In the Google Cloud Console, enable the **Google Calendar API** and create an **OAuth client ID** of type _Desktop app_.
2. Download the client JSON and save it as `secrets/google_credentials.json`.
3. Run the auth script. It opens your browser to log in and writes `secrets/google_token.json`:

   ```bash
   uv run python scripts/google_auth.py
   ```

The `secrets/` folder is git-ignored. Never commit it.

### 4. Run

**Windows (PowerShell):**

```powershell
.\run.ps1 ava-run      # build and start all containers
.\run.ps1 ava-stop     # stop them
.\run.ps1 ava-delete   # stop and delete memory data
```

**macOS / Linux:**

```bash
make ava-run
make ava-stop
make ava-delete
```

| Service           | URL                             |
| ----------------- | ------------------------------- |
| Chainlit web chat | http://localhost:8000           |
| WhatsApp webhook  | http://localhost:8081           |
| Qdrant dashboard  | http://localhost:6333/dashboard |

To use WhatsApp, expose port `8081` publicly (e.g. with ngrok) and register the URL as the webhook in your Meta app.

## Example prompts

```
Was steht am Montag in meinem Kalender?
Plane ein Meeting 'Team Sync' am Montag um 16 Uhr für eine Stunde
Schick mir eine Sprachnachricht!
Zeig mir ein Bild von deinem Arbeitsplatz
```

When Bob schedules a meeting, he asks first (_"Soll ich … in deinen Kalender eintragen? (ja/nein)"_). Answer `ja` to confirm, or say what you want to change.

## Development

```bash
.\run.ps1 format-fix   # or: make format-fix
.\run.ps1 lint-check   # or: make lint-check
```

Manual test scripts (they need a valid `.env` and a Google token):

```bash
uv run python scripts/test_router.py          # router decisions
uv run python scripts/test_calendar.py        # Google Calendar connection
uv run python scripts/test_calendar_tools.py  # calendar tools
uv run python scripts/test_action.py          # action node + tool loop end to end
```

You can also open the graph in LangGraph Studio using [langgraph.json](langgraph.json).

## Deployment

[cloudbuild.yaml](cloudbuild.yaml) deploys to Google Cloud Run. See [docs/gcp_setup.md](docs/gcp_setup.md).

## License

MIT. See [LICENSE](LICENSE).
