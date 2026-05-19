from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.utils import timezone
from openai import OpenAI

from .models import Meeting, MeetingType
from .openai_utils import chat_completion_options


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
            **chat_completion_options(self.model, temperature=0.2),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce faithful meeting outputs from diarized transcripts. "
                        "These transcripts came from recorded audio, so some words may be mistranscribed. "
                        "When a word or phrase makes no sense, infer the intended wording from context "
                        "only when the correction is reasonably clear. Use only the transcript content. "
                        "Do not invent facts. If something remains unclear, mark it as unclear. "
                        "Preserve speaker labels when assigning owners."
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
    if meeting.meeting_type == MeetingType.REQUIREMENT_GATHERING:
        return build_requirements_prompt(meeting, transcript, meeting_type)
    if meeting.meeting_type == MeetingType.PROJECT_MANAGER_NOTES:
        return build_project_manager_notes_prompt(meeting, transcript, meeting_type)

    type_guidance = {
        "requirement_gathering_minutes": (
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
- This transcript came from recorded audio and may contain transcription mistakes. If a word or phrase makes no sense, deduce the likely intended wording from context when reasonably clear; otherwise mark it as unclear.
- Include these sections: Summary, Key Discussion Points, Decisions, Action Items, Open Questions, Risks or Blockers.
- For action items, include Owner, Task, Due date, and Evidence. Use "Unassigned" or "Not stated" when missing.
- Tailor the output to this meeting type: {type_guidance}
- Keep the wording professional and practical.

Transcript:
{transcript}
"""


def build_project_manager_notes_prompt(meeting: Meeting, transcript: str, meeting_type: str) -> str:
    return f"""Meeting type: {meeting_type}
Meeting title: {meeting.title or "Untitled meeting"}
Started at: {meeting.started_at.isoformat()}
Ended at: {meeting.ended_at.isoformat() if meeting.ended_at else "Not ended"}

Instructions:
- You are generating meeting notes from a transcribed meeting.
- The transcript may contain transcription mistakes, wrong speaker labels, missing punctuation, repeated phrases, or misheard product and feature names. Deduce the intended meaning when something sounds wrong, but do not invent requirements or decisions that are not supported by the transcript.
- Write the notes in the same format and level of detail as professional project manager notes.
- Output only the meeting notes. Do not include an introduction, explanation, Markdown table, action-items section, or summary section unless the meeting itself explicitly had those items.
- Use plain text, not Markdown.

Use this exact structure:

Meeting Details:

Date: [Use the meeting date if known from metadata or transcript, otherwise "Not specified"]
Time: [Use the meeting time if known from metadata or transcript, otherwise "Not specified"]
Location/Platform: [Use the platform if known from the transcript, otherwise "Not specified"]
Attendees:

[List attendee names, one per line. If attendees are unclear, write "Not specified"]

Discussion Points:

[Group the discussion by product area, feature, flow, screen, role, or topic.]

For each discussion topic:
- Use short plain headings such as "Academy Admin Flow", "Player Flow", "Video Posts", or "Design & UX Notes".
- Capture all concrete requirements, decisions, UX notes, edge cases, and clarifications.
- Preserve hierarchy where needed:
  - Feature area
  - Screen or flow inside it
  - Specific requested changes
- Use concise lines and bullets.
- Avoid long paragraphs.
- Keep wording close to implementation-ready notes, not a conversational recap.
- Do not over-summarize. Important details must not be lost.
- Do not write "The team discussed..." unless needed for clarity.
- Keep product names, screen names, role names, and button labels exactly as intended.
- Do not include timestamps.
- Do not include speaker names unless the speaker identity matters to the requirement or decision.
- If the meeting discusses alternatives and later chooses one, include only the final chosen direction.
- If something is discussed and later removed, omit it completely.
- If two points contradict each other, keep the later one and omit the earlier one.

Transcript:
{transcript}
"""


def build_requirements_prompt(meeting: Meeting, transcript: str, meeting_type: str) -> str:
    return f"""Meeting type: {meeting_type}
Meeting title: {meeting.title or "Untitled meeting"}
Started at: {meeting.started_at.isoformat()}
Ended at: {meeting.ended_at.isoformat() if meeting.ended_at else "Not ended"}

Instructions:
- This is a transcribed requirements gathering meeting. The transcript may contain transcription mistakes. If a word or phrase makes no sense, deduce the likely intended wording from context when reasonably clear; otherwise mark it as unclear.
- Do not summarize the meeting.
- Output only the final gathered requirements.
- Return a concise Markdown bullet list, one requirement per bullet.
- Do not include headings, sections, meeting recap, action items, decisions, open questions, risks, participants, timestamps, or other fluff.
- If something was discussed and later removed, omit it completely from the requirements.
- If one requirement contradicts another, keep only the more recent requirement and omit the older conflicting requirement completely.
- Requirements must be written as clear product or business requirements, not as notes about who said what.
- Use only information supported by the transcript after applying the removal and recency rules above.

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
