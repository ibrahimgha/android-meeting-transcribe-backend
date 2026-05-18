from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.utils import timezone
from openai import OpenAI

from .models import Meeting


class MinutesConfigurationError(RuntimeError):
    pass


class MinutesInputError(ValueError):
    pass


@dataclass
class MinutesResult:
    text: str
    model: str
    raw_response: dict[str, Any]


class OpenAIMinutesClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MINUTES_MODEL
        if not self.api_key:
            raise MinutesConfigurationError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, meeting: Meeting) -> MinutesResult:
        transcript = build_transcript(meeting)
        if not transcript.strip():
            raise MinutesInputError("This meeting does not have completed transcriptions yet.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce crisp, useful meeting minutes from diarized transcripts. "
                        "Use only the transcript content. Do not invent facts. If something is unclear, "
                        "mark it as unclear. Preserve speaker labels when assigning owners."
                    ),
                },
                {
                    "role": "user",
                    "content": build_minutes_prompt(meeting, transcript),
                },
            ],
        )
        raw = serialize_response(response)
        text = response.choices[0].message.content or ""
        return MinutesResult(text=text.strip(), model=self.model, raw_response=raw)


def generate_minutes_for_meeting(
    meeting: Meeting,
    client: OpenAIMinutesClient | None = None,
) -> Meeting:
    client = client or OpenAIMinutesClient()
    try:
        result = client.generate(meeting)
    except Exception as exc:
        meeting.minutes_last_error = str(exc)
        meeting.save(update_fields=["minutes_last_error", "updated_at"])
        raise

    meeting.minutes_text = result.text
    meeting.minutes_model = result.model
    meeting.minutes_response = result.raw_response
    meeting.minutes_generated_at = timezone.now()
    meeting.minutes_last_error = ""
    meeting.save(
        update_fields=[
            "minutes_text",
            "minutes_model",
            "minutes_response",
            "minutes_generated_at",
            "minutes_last_error",
            "updated_at",
        ],
    )
    return meeting


def build_transcript(meeting: Meeting) -> str:
    messages = list(meeting.messages.order_by("sequence_number"))
    if messages:
        return "\n".join(
            f"[{format_timestamp(message.client_start_ms)}-{format_timestamp(message.client_end_ms)}] "
            f"{message.speaker_label}: {message.transcript_text.strip()}"
            for message in messages
            if message.transcript_text.strip()
        )

    lines = []
    for segment in meeting.segments.order_by("sequence_number"):
        text = (segment.transcription_text or "").strip()
        if not text:
            continue
        start = format_timestamp(segment.client_start_ms)
        end = format_timestamp(segment.client_end_ms)
        lines.append(f"[{start}-{end}] {segment.speaker_label}: {text}")
    return "\n".join(lines)


def build_minutes_prompt(meeting: Meeting, transcript: str) -> str:
    meeting_type = meeting.get_meeting_type_display() if meeting.meeting_type else "Unspecified"
    type_guidance = {
        "requirement_gathering": (
            "Focus on business goals, user needs, functional requirements, non-functional requirements, "
            "constraints, assumptions, open questions, risks, decisions, and next steps."
        ),
        "followup_meeting": (
            "Focus on progress since the previous discussion, blockers, decisions, changed scope, "
            "new action items, owners, due dates, and unresolved follow-ups."
        ),
        "draft_delivery": (
            "Focus on what was delivered, feedback received, requested revisions, accepted items, "
            "rejected items, open questions, risks, and next delivery actions."
        ),
    }.get(meeting.meeting_type, "Focus on decisions, action items, risks, and open questions.")

    return f"""Meeting type: {meeting_type}
Meeting title: {meeting.title or "Untitled meeting"}
Started at: {meeting.started_at.isoformat()}
Ended at: {meeting.ended_at.isoformat() if meeting.ended_at else "Not ended"}

Instructions:
- Write concise meeting minutes in Markdown.
- Include these sections: Summary, Key Discussion Points, Decisions, Action Items, Open Questions, Risks or Blockers.
- For action items, include Owner, Task, Due date, and Evidence. Use "Unassigned" or "Not stated" when missing.
- Tailor the output to this meeting type: {type_guidance}
- Keep the wording professional and practical.

Transcript:
{transcript}
"""


def format_timestamp(milliseconds: int) -> str:
    total_seconds = int(milliseconds / 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def serialize_response(response) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"text": str(response)}
