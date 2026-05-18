import json
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from openai import OpenAI

from .minutes import format_timestamp, serialize_response
from .models import (
    AudioSegment,
    Meeting,
    MeetingMessage,
    MeetingOutputStatus,
    MeetingStatus,
    SegmentStatus,
)


class MeetingOutputConfigurationError(RuntimeError):
    pass


class MeetingOutputInputError(ValueError):
    pass


@dataclass
class TextResult:
    text: str
    raw_response: dict[str, Any]


@dataclass
class MessageDraft:
    sequence_numbers: list[int]
    speaker_label: str
    transcript_text: str
    client_start_ms: int
    client_end_ms: int


class OpenAIMeetingOutputClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MEETING_ANALYSIS_MODEL
        if not self.api_key:
            raise MeetingOutputConfigurationError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=self.api_key)

    def compile_messages(
        self,
        meeting: Meeting,
        segments: list[AudioSegment],
    ) -> tuple[list[MessageDraft], dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You group diarized transcript segments into display messages. "
                        "Return valid JSON only. Preserve every source segment exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": build_compile_prompt(meeting, segments),
                },
            ],
        )
        raw = serialize_response(response)
        content = response.choices[0].message.content or "{}"
        groups = parse_message_groups(content)
        return normalize_message_groups(groups, segments), raw

    def summarize_message(self, draft: MessageDraft) -> TextResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write faithful English summaries. Include every concrete detail, "
                        "name, number, decision, constraint, and caveat from the transcript. "
                        "Do not add facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this displayed meeting message in English without missing details. "
                        "Use compact paragraphs or bullets as needed.\n\n"
                        f"Speaker: {draft.speaker_label}\n"
                        f"Transcript:\n{draft.transcript_text}"
                    ),
                },
            ],
        )
        raw = serialize_response(response)
        return TextResult(text=(response.choices[0].message.content or "").strip(), raw_response=raw)

    def summarize_message_short(self, draft: MessageDraft) -> TextResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "Write short English labels. Use 12 words or fewer. No extra commentary.",
                },
                {
                    "role": "user",
                    "content": (
                        "Create a 12-word-or-fewer English summary for this meeting message.\n\n"
                        f"Transcript:\n{draft.transcript_text}"
                    ),
                },
            ],
        )
        raw = serialize_response(response)
        text = response.choices[0].message.content or ""
        if len(text.split()) > 12:
            rewrite = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite labels in fluent English using 12 words or fewer.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous label was too long or awkward. Rewrite it in 12 words or fewer, "
                            "without dangling words.\n\n"
                            f"Previous label: {text}\n\n"
                            f"Transcript:\n{draft.transcript_text}"
                        ),
                    },
                ],
            )
            raw = serialize_response(rewrite)
            text = rewrite.choices[0].message.content or ""
        return TextResult(text=cap_words(text, 12), raw_response=raw)

    def generate_title(self, meeting: Meeting, drafts: list[MessageDraft]) -> TextResult:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create concise professional meeting titles in English. "
                        "Use 3 to 9 words. Do not wrap the title in quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": build_title_prompt(meeting, drafts),
                },
            ],
        )
        raw = serialize_response(response)
        return TextResult(text=clean_title(response.choices[0].message.content or ""), raw_response=raw)


def process_meeting_outputs(
    meeting: Meeting,
    *,
    client: OpenAIMeetingOutputClient | None = None,
    force: bool = False,
) -> Meeting:
    meeting.refresh_from_db()
    if (
        not force
        and meeting.output_status == MeetingOutputStatus.COMPLETE
        and meeting.messages.exists()
    ):
        return meeting

    segments = completed_segments(meeting)
    if not segments:
        raise MeetingOutputInputError("This meeting does not have completed transcriptions yet.")

    meeting.output_status = MeetingOutputStatus.PROCESSING
    meeting.output_last_error = ""
    meeting.save(update_fields=["output_status", "output_last_error", "updated_at"])

    client = client or OpenAIMeetingOutputClient()
    try:
        drafts, compile_response = client.compile_messages(meeting, segments)
        title_result = client.generate_title(meeting, drafts)
        summarized = []
        for draft in drafts:
            detailed = client.summarize_message(draft)
            short = client.summarize_message_short(draft)
            summarized.append((draft, detailed, short))

        with transaction.atomic():
            meeting.messages.all().delete()
            segment_by_sequence = {segment.sequence_number: segment for segment in segments}
            for index, (draft, detailed, short) in enumerate(summarized, start=1):
                message = MeetingMessage.objects.create(
                    meeting=meeting,
                    user=meeting.user,
                    sequence_number=index,
                    speaker_label=draft.speaker_label,
                    client_start_ms=draft.client_start_ms,
                    client_end_ms=draft.client_end_ms,
                    transcript_text=draft.transcript_text,
                    detailed_summary=detailed.text,
                    short_summary=short.text,
                    detailed_summary_response=detailed.raw_response,
                    short_summary_response=short.raw_response,
                    summary_model=client.model,
                )
                message.segments.set(
                    segment_by_sequence[number]
                    for number in draft.sequence_numbers
                    if number in segment_by_sequence
                )

            meeting.title = title_result.text or meeting.title
            meeting.output_status = MeetingOutputStatus.COMPLETE
            meeting.output_model = client.model
            meeting.output_response = compile_response
            meeting.output_generated_at = timezone.now()
            meeting.output_last_error = ""
            meeting.title_model = client.model
            meeting.title_response = title_result.raw_response
            meeting.title_generated_at = timezone.now()
            meeting.save(
                update_fields=[
                    "title",
                    "output_status",
                    "output_model",
                    "output_response",
                    "output_generated_at",
                    "output_last_error",
                    "title_model",
                    "title_response",
                    "title_generated_at",
                    "updated_at",
                ],
            )
    except Exception as exc:
        meeting.output_status = MeetingOutputStatus.FAILED
        meeting.output_last_error = str(exc)
        meeting.save(update_fields=["output_status", "output_last_error", "updated_at"])
        raise

    return meeting


def maybe_process_completed_meeting(meeting: Meeting) -> None:
    meeting.refresh_from_db()
    if meeting.status != MeetingStatus.COMPLETE or not settings.OPENAI_API_KEY:
        return
    if meeting.output_status == MeetingOutputStatus.PROCESSING:
        return
    try:
        process_meeting_outputs(meeting)
    except Exception:
        return


def completed_segments(meeting: Meeting) -> list[AudioSegment]:
    return list(
        meeting.segments.filter(
            transcription_status=SegmentStatus.COMPLETE,
        )
        .exclude(transcription_text="")
        .order_by("sequence_number")
    )


def build_compile_prompt(meeting: Meeting, segments: list[AudioSegment]) -> str:
    segment_lines = []
    for segment in segments:
        segment_lines.append(
            json.dumps(
                {
                    "sequence_number": segment.sequence_number,
                    "speaker_label": segment.speaker_label,
                    "start": format_timestamp(segment.client_start_ms),
                    "end": format_timestamp(segment.client_end_ms),
                    "transcript": segment.transcription_text,
                },
                ensure_ascii=False,
            )
        )

    return f"""Meeting title: {meeting.title or "Untitled meeting"}

Group these consecutive transcript segments into displayed messages.

Rules:
- Return JSON with exactly this shape: {{"messages": [{{"sequence_numbers": [1, 2]}}]}}.
- Every input sequence_number must appear exactly once.
- Keep sequence_numbers in chronological order.
- Merge adjacent segments when they are the same speaker continuing one thought, a sentence was split, or a brief interruption clearly belongs with the same turn.
- Do not merge across a clear new topic or a substantial speaker turn change.
- Do not summarize or rewrite text in the JSON.

Segments:
{chr(10).join(segment_lines)}
"""


def parse_message_groups(content: str) -> list[list[int]]:
    parsed = json.loads(content)
    groups = []
    for item in parsed.get("messages", []):
        numbers = item.get("sequence_numbers", [])
        if isinstance(numbers, list):
            cleaned = []
            for number in numbers:
                try:
                    cleaned.append(int(number))
                except (TypeError, ValueError):
                    continue
            if cleaned:
                groups.append(cleaned)
    return groups


def normalize_message_groups(
    groups: list[list[int]],
    segments: list[AudioSegment],
) -> list[MessageDraft]:
    segment_by_sequence = {segment.sequence_number: segment for segment in segments}
    expected = [segment.sequence_number for segment in segments]
    used = set()
    normalized = []

    for group in groups:
        sequence_numbers = []
        for number in group:
            if number in segment_by_sequence and number not in used:
                sequence_numbers.append(number)
                used.add(number)
        if sequence_numbers:
            normalized.append(sequence_numbers)

    for number in expected:
        if number not in used:
            normalized.append([number])

    normalized.sort(key=lambda item: expected.index(item[0]))
    return [draft_from_segments([segment_by_sequence[number] for number in group]) for group in normalized]


def draft_from_segments(segments: list[AudioSegment]) -> MessageDraft:
    speaker_labels = []
    for segment in segments:
        if segment.speaker_label not in speaker_labels:
            speaker_labels.append(segment.speaker_label)

    transcript = "\n".join(
        f"{segment.speaker_label}: {segment.transcription_text.strip()}"
        for segment in segments
        if segment.transcription_text.strip()
    )
    return MessageDraft(
        sequence_numbers=[segment.sequence_number for segment in segments],
        speaker_label=" / ".join(speaker_labels),
        transcript_text=transcript,
        client_start_ms=min(segment.client_start_ms for segment in segments),
        client_end_ms=max(segment.client_end_ms for segment in segments),
    )


def build_title_prompt(meeting: Meeting, drafts: list[MessageDraft]) -> str:
    transcript = "\n\n".join(
        f"Message {index}: {draft.transcript_text}"
        for index, draft in enumerate(drafts, start=1)
    )
    return f"""Create a concise English title for this meeting.

Existing title: {meeting.title or "Untitled meeting"}
Started at: {meeting.started_at.isoformat()}

Transcript messages:
{transcript}
"""


def clean_title(value: str) -> str:
    cleaned = " ".join(value.strip().strip('"').strip("'").split())
    return cleaned[:160]


def cap_words(value: str, limit: int) -> str:
    words = value.strip().split()
    return " ".join(words[:limit])
