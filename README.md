# Meeting Transcribe Backend

Django REST Framework backend for authenticated meeting recording sessions.

## What It Does

- Registers and authenticates users with DRF token auth.
- Creates a meeting when the mobile app starts recording.
- Ends the meeting when the mobile app stops recording.
- Receives sequential audio chunks with client-provided speaker metadata.
- Queues chunks in the database and transcribes them sequentially with OpenAI.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```powershell
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

## API

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
```

Use token auth:

```text
Authorization: Token <token>
```

## Transcription Worker

Run one worker process for sequential processing:

```powershell
.\.venv\Scripts\python.exe manage.py transcribe_segments
```

For a one-shot local run:

```powershell
.\.venv\Scripts\python.exe manage.py transcribe_segments --once
```

The default transcription model is `gpt-4o-mini-transcribe`. Override with `OPENAI_TRANSCRIBE_MODEL`.

## Tests

```powershell
.\.venv\Scripts\python.exe manage.py test meetings
```
