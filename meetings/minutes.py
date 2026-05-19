from datetime import timezone as datetime_timezone
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

        if meeting.meeting_type == MeetingType.PROJECT_MANAGER_NOTES:
            return self.generate_project_manager_notes(meeting, transcript)

        response = self.client.chat.completions.create(
            **chat_completion_options(self.model, temperature=0.2),
            messages=[
                {
                    "role": "system",
                    "content": minutes_system_prompt(),
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

    def generate_project_manager_notes(self, meeting: Meeting, transcript: str) -> MinutesResult:
        chunks = chunk_transcript(transcript)
        chunk_notes = []
        chunk_responses = []
        for index, chunk in enumerate(chunks, start=1):
            response = self.client.chat.completions.create(
                **chat_completion_options(self.model, temperature=0.2),
                messages=[
                    {
                        "role": "system",
                        "content": minutes_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": build_project_manager_chunk_prompt(
                            meeting,
                            chunk,
                            chunk_index=index,
                            chunk_count=len(chunks),
                        ),
                    },
                ],
            )
            chunk_responses.append(serialize_response(response))
            chunk_notes.append((response.choices[0].message.content or "").strip())

        combined_notes = "\n\n".join(
            f"Chunk {index} notes:\n{notes}"
            for index, notes in enumerate(chunk_notes, start=1)
            if notes
        )
        response = self.client.chat.completions.create(
            **chat_completion_options(self.model, temperature=0.2),
            messages=[
                {
                    "role": "system",
                    "content": minutes_system_prompt(),
                },
                {
                    "role": "user",
                    "content": build_project_manager_final_prompt(meeting, combined_notes),
                },
            ],
        )
        final_raw = serialize_response(response)
        text = response.choices[0].message.content or ""
        return MinutesResult(
            text=text.strip(),
            model=self.model,
            raw_response={
                "chunk_count": len(chunks),
                "chunk_responses": chunk_responses,
                "final_response": final_raw,
            },
        )


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


def minutes_system_prompt() -> str:
    return (
        "You produce faithful meeting outputs from diarized transcripts. "
        "These transcripts came from recorded audio, so some words may be mistranscribed. "
        "When a word or phrase makes no sense, infer the intended wording from context "
        "only when the correction is reasonably clear. Use only the transcript content. "
        "Do not invent facts. If something remains unclear, mark it as unclear. "
        "Preserve speaker labels when assigning owners."
    )


def chunk_transcript(transcript: str, *, max_chars: int = 12_000) -> list[str]:
    lines = transcript.splitlines()
    chunks = []
    current = []
    current_size = 0
    for line in lines:
        line_size = len(line) + 1
        if current and current_size + line_size > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += line_size
    if current:
        chunks.append("\n".join(current))
    return chunks or [transcript]


def build_project_manager_chunk_prompt(
    meeting: Meeting,
    transcript_chunk: str,
    *,
    chunk_index: int,
    chunk_count: int,
) -> str:
    return f"""Meeting title: {meeting.title or "Untitled meeting"}
Transcript chunk: {chunk_index} of {chunk_count}

Task:
Extract exhaustive raw project-manager notes from this transcript chunk only.

Rules:
- This is an extraction pass, not a summary pass.
- Capture every concrete requested change, decision, edge case, role permission, screen/flow change, field, button, validation rule, and UX/design note.
- Keep small details as separate bullets so they cannot be lost later.
- Include uncertain or mistranscribed terms when the intended meaning is reasonably clear, and mark unclear wording as unclear.
- Preserve product names, screen names, role names, labels, and button names as closely as possible.
- Do not drop details because they seem minor, repeated, already implied, or similar to another point.
- If a point is later contradicted or removed inside this same chunk, keep only the later/final direction.
- Use plain topic headings and bullets.
- If the chunk has no usable meeting-note details, write "No concrete notes in this chunk."

Transcript chunk:
{transcript_chunk}
"""


def build_project_manager_final_prompt(meeting: Meeting, extracted_chunk_notes: str) -> str:
    meeting_date, meeting_time = meeting_datetime_for_prompt(meeting)
    return f"""Meeting title: {meeting.title or "Untitled meeting"}
Metadata date: {meeting_date}
Metadata time: {meeting_time}
Started at: {meeting.started_at.isoformat()}
Ended at: {meeting.ended_at.isoformat() if meeting.ended_at else "Not ended"}

You are given exhaustive notes extracted chunk-by-chunk from a full meeting transcript.

Task:
Consolidate the chunk notes into final project-manager meeting notes that match the style, shape, and approximate length of the format below.

Critical rules:
- This is not a transcript recap and not a conversational summary.
- The output can omit who said what, filler, repetitions, examples, and discussion wording.
- The output must not omit information: every distinct requirement, decision, constraint, edge case, filter, field, role permission, screen/flow change, button, validation rule, and UX/design note must appear at least once.
- Completeness means no unique product or project-management information is lost; it does not mean every utterance, unclear fragment, example, or rephrasing needs its own bullet.
- Target 800-1,300 words for the final notes. For a very information-dense meeting, you may go up to 1,600 words only if needed to avoid losing information.
- Keep the notes around the same length and density as the reference format below. Prefer compact, information-rich bullets over long paragraphs.
- Merge related wording into one bullet when no information is lost.
- Omit unclear transcript fragments unless they resolve to a concrete implementation note, decision, risk, or open point.
- Remove true duplicates only when the same point is repeated with no new detail.
- If something is discussed and later removed, omit it completely.
- If two points contradict each other, keep the later one and omit the earlier one.
- Use implementation-ready wording, not "the team discussed" phrasing.
- Output only the final meeting notes.
- Use plain text with hyphen bullets, not Markdown headings or tables.

Use this exact structure:

Meeting Details:

Date: {meeting_date}
Time: {meeting_time}
Location/Platform: [Use the platform if known from the notes, otherwise "Not specified"]

Attendees:

[List attendee names, one per line. If attendees are unclear, write "Not specified"]

Discussion Points:

[Group the discussion by product area, feature, flow, screen, role, or topic. Use concise headings and nested bullets.]

Reference style and length:

Meeting Details:

Date: 2026-05-19
Time: 09:52 UTC
Location/Platform: Not specified

Attendees:

Not specified

Discussion Points:

Team Player Assignment Flow

- Player assignment should be supported from both directions:
  - From Team: add/assign players to the team.
  - From Player: assign the player to a team or change the player’s team assignment.
- Same concept applies to coaches/players where assignment can be managed from either the team side or the individual profile side.
- The Player field is optional.
- Final direction is to keep the assignment capability available both in Teams and in Players, not only in one place.

Team Editing / Add Player Window

- In team editing, adding players should allow selection from existing teams and unassigned players.
- The add player window should support filtering by team, assigned/unassigned status, and position.
- Existing windows can be reused where possible, but a new or adjusted window may be needed if the current one cannot support team-indexed assigned players.

Extracted chunk notes:
{extracted_chunk_notes}
"""


def build_project_manager_notes_prompt(meeting: Meeting, transcript: str, meeting_type: str) -> str:
    meeting_date, meeting_time = meeting_datetime_for_prompt(meeting)
    return f"""Meeting type: {meeting_type}
Meeting title: {meeting.title or "Untitled meeting"}
Metadata date: {meeting_date}
Metadata time: {meeting_time}
Started at: {meeting.started_at.isoformat()}
Ended at: {meeting.ended_at.isoformat() if meeting.ended_at else "Not ended"}

Instructions:
- You are generating meeting notes from a transcribed meeting.
- The transcript may contain transcription mistakes, wrong speaker labels, missing punctuation, repeated phrases, or misheard product and feature names. Deduce the intended meaning when something sounds wrong, but do not invent requirements or decisions that are not supported by the transcript.
- Write the notes in the same format, density, and approximate length as compact professional project manager notes.
- Output only the meeting notes. Do not include an introduction, explanation, Markdown table, action-items section, or summary section unless the meeting itself explicitly had those items.
- Use plain text with hyphen bullets, not Markdown headings or tables.
- This is not a transcript recap and not a conversational summary.
- The output can omit who said what, filler, repetitions, examples, and discussion wording.
- The output must not omit information: every distinct requested change, decision, constraint, edge case, role permission, screen/flow change, field, button, validation rule, and UX/design note that appears in the transcript must appear at least once.
- Completeness means no unique product or project-management information is lost; it does not mean every utterance, unclear fragment, example, or rephrasing needs its own bullet.
- Target 800-1,300 words for the final notes. For a very information-dense meeting, you may go up to 1,600 words only if needed to avoid losing information.
- Keep the notes around the same length and density as the reference format below. Prefer compact, information-rich bullets over long paragraphs.
- When several small points belong under the same topic, use nested bullets. Merge related wording when no information is lost.
- Omit unclear transcript fragments unless they resolve to a concrete implementation note, decision, risk, or open point.
- Before finalizing, review the transcript again and add any concrete detail that was not already captured.

Use this exact structure:

Meeting Details:

Date: {meeting_date}
Time: {meeting_time}
Location/Platform: [Use the platform if known from the transcript, otherwise "Not specified"]

Attendees:

[List attendee names, one per line. If attendees are unclear, write "Not specified"]

Discussion Points:

[Group the discussion by product area, feature, flow, screen, role, or topic.]

For each discussion topic:
- Use short plain headings such as "Academy Admin Flow", "Player Flow", "Video Posts", or "Design & UX Notes".
- Capture all concrete requirements, decisions, UX notes, edge cases, and clarifications.
- Preserve exhaustive detail under each topic even when the topic already has a high-level bullet.
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

Reference style and length:

Meeting Details:

Date: 2026-05-19
Time: 09:52 UTC
Location/Platform: Not specified

Attendees:

Not specified

Discussion Points:

Team Player Assignment Flow

- Player assignment should be supported from both directions:
  - From Team: add/assign players to the team.
  - From Player: assign the player to a team or change the player’s team assignment.
- Same concept applies to coaches/players where assignment can be managed from either the team side or the individual profile side.
- The Player field is optional.
- Final direction is to keep the assignment capability available both in Teams and in Players, not only in one place.

Team Editing / Add Player Window

- In team editing, adding players should allow selection from existing teams and unassigned players.
- The add player window should support filtering by team, assigned/unassigned status, and position.
- Existing windows can be reused where possible, but a new or adjusted window may be needed if the current one cannot support team-indexed assigned players.

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


def meeting_datetime_for_prompt(meeting: Meeting) -> tuple[str, str]:
    started_at = meeting.started_at
    if not started_at:
        return "Not specified", "Not specified"
    started_at_utc = started_at.astimezone(datetime_timezone.utc)
    return started_at_utc.strftime("%Y-%m-%d"), started_at_utc.strftime("%H:%M UTC")


def serialize_response(response) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"text": str(response)}
