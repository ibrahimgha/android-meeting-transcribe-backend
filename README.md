# Meeting Transcribe Backend

Django REST Framework backend for authenticated meeting recording sessions.

## What It Does

- Registers and authenticates users with DRF token auth.
- Creates a meeting when the mobile app starts recording.
- Ends the meeting when the mobile app stops recording.
- Receives sequential audio chunks with client-provided speaker metadata.
- Queues chunks in the database and transcribes them sequentially with OpenAI.
- Provides an authenticated web page where a user can view meetings and extract meeting minutes.

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

## Web Meeting Minutes

Open `/meetings/` in a browser and log in with the same Django user used by the mobile app.
Open a meeting detail page to review processed messages. Each displayed message includes:

- The source audio segments with browser playback controls.
- The merged full transcript.
- A detailed English summary.
- A 12-word-or-fewer English summary.

On the meeting detail page, choose one of:

- Requirement gathering
- Followup meeting
- Draft delivery

Then click **Extract meeting minutes**. The backend sends the meeting transcript to OpenAI and stores
the generated Markdown minutes on the meeting. The default minutes model is `gpt-4o-mini`; override
with `OPENAI_MINUTES_MODEL`.

After a meeting finishes transcribing, the transcription worker also asks OpenAI to combine segments
into displayed messages, summarize each message, and generate a meeting title. The default model is
`gpt-4o-mini`; override with `OPENAI_MEETING_ANALYSIS_MODEL`.

To rebuild outputs manually:

```powershell
.\.venv\Scripts\python.exe manage.py process_meeting_outputs --latest --force
```

## Tests

```powershell
.\.venv\Scripts\python.exe manage.py test meetings
```
