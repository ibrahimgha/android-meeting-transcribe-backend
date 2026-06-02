# Meeting Transcribe Backend

Open-source Django and Django REST Framework backend for authenticated meeting recording, transcription, AI-generated meeting notes, and meeting health analytics.

This backend is designed to pair with a mobile meeting recorder. The mobile client uploads speaker-labeled audio segments, while the server stores meetings, queues transcription jobs, generates derived meeting outputs, and exposes a web portal for reviewing results.

## Features

- Token-authenticated REST API for mobile clients.
- Meeting lifecycle endpoints for start, upload segments, end, and import existing recordings.
- Sequential background worker for imports, transcription, meeting output generation, and meeting-minutes jobs.
- OpenAI transcription support for uploaded audio segments.
- Authenticated web portal for meeting review.
- Browser playback for original audio segments.
- Generated display messages with merged transcripts, detailed summaries, short summaries, and meeting titles.
- Meeting-minutes extraction for requirements, follow-ups, draft delivery, and project-manager notes.
- Meeting health reports with a score out of 10 and dashboard statistics.
- Optional MCP server so agents can operate the tool through a controlled API.

## Architecture

```text
Mobile app / web import
        |
        v
Django REST API
        |
        v
Database-backed queue
        |
        v
Worker command: transcribe_segments
        |
        +-- import full recordings into audio segments
        +-- transcribe audio segments with OpenAI
        +-- build display messages, summaries, and titles
        +-- generate queued meeting minutes and health reports
```

The queue is intentionally simple: jobs are stored in the database and processed sequentially by a long-running Django management command.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set at least `DJANGO_SECRET_KEY` and `OPENAI_API_KEY` in `.env`, then run:

```powershell
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

Create a Django user through the admin, shell, or your own registration flow, then open:

```text
http://127.0.0.1:8000/meetings/
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `DJANGO_SECRET_KEY` | Django secret key. Required for production. |
| `DJANGO_DEBUG` | Use `false` in production. |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated allowed hosts. |
| `CORS_ALLOWED_ORIGINS` | Comma-separated mobile/web origins. |
| `OPENAI_API_KEY` | OpenAI API key used for transcription and analysis. |
| `OPENAI_TRANSCRIBE_MODEL` | Audio transcription model. |
| `OPENAI_MINUTES_MODEL` | Model used for meeting minutes and health reports. |
| `OPENAI_MEETING_ANALYSIS_MODEL` | Model used for message grouping, summaries, and titles. |
| `MAX_AUDIO_SEGMENT_BYTES` | Max single segment upload size. |
| `MAX_IMPORT_RECORDING_BYTES` | Max full recording import size. |
| `IMPORT_CHUNK_BYTES` | Browser chunk size for large uploads. |
| `FFMPEG_BINARY` | Optional path to ffmpeg for recording imports. |
| `MCP_AUTH_TOKEN` | Optional bearer token for MCP access. |
| `MCP_DEFAULT_USERNAME` | Django username used by MCP tools. |
| `MCP_PUBLIC_URL` | Public MCP endpoint URL. |
| `PROPOSAL_GENERATOR_URL` | Optional URL for a downstream proposal generator. |
| `PM_NOTES_PDF_AUTHOR` | Optional PDF metadata author for project-manager notes. |
| `PM_NOTES_PDF_FOOTER_TEXT` | Optional footer text for project-manager notes PDFs. |
| `DB_ENGINE`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | Database configuration. SQLite is used by default. |

## REST API

Use token authentication:

```text
Authorization: Token <token>
```

Main endpoints:

```text
POST /api/auth/register/
POST /api/auth/login/
POST /api/auth/logout/
GET  /api/auth/me/

POST /api/meetings/start/
GET  /api/meetings/
GET  /api/meetings/{meeting_id}/
POST /api/meetings/{meeting_id}/segments/
POST /api/meetings/{meeting_id}/end/
POST /api/meetings/import/
```

## Background Worker

Run one worker process:

```powershell
.\.venv\Scripts\python.exe manage.py transcribe_segments
```

For a one-shot local run:

```powershell
.\.venv\Scripts\python.exe manage.py transcribe_segments --once
```

Backfill meeting health reports for completed meetings:

```powershell
.\.venv\Scripts\python.exe manage.py queue_health_reports --process
```

## Web Portal

The authenticated web portal supports:

- importing previous recordings,
- viewing meetings and transcribed audio segments,
- playing source audio in the browser,
- extracting saved meeting outputs by type,
- downloading project-manager notes PDFs,
- reviewing meeting health reports and dashboard statistics.

Users can be granted access to all meetings through `UserWebSettings.can_view_all_meetings`.

## MCP

The optional MCP server exposes meeting operations to agents. Configure:

```text
MCP_AUTH_TOKEN=
MCP_DEFAULT_USERNAME=
MCP_PUBLIC_URL=
```

Then run the MCP service using your deployment process or the MCP server module.

## Security Notes

- Do not commit `.env`, `db.sqlite3`, uploaded media, static build output, or local virtual environments.
- Rotate any API key that was ever committed to git history.
- Use a strong `DJANGO_SECRET_KEY` in production.
- Set `DJANGO_DEBUG=false` in production.
- Restrict `DJANGO_ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS`, and MCP access for deployed environments.
- Uploaded recordings and transcripts can contain sensitive information; store and retain them according to your privacy requirements.

## Tests

```powershell
.\.venv\Scripts\python.exe manage.py test meetings
```
