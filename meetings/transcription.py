import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.files.storage import default_storage
from django.db import transaction
from django.db.models import Max, Min
from django.utils import timezone
from openai import OpenAI

from .models import AudioSegment, Meeting, MeetingStatus, SegmentStatus


logger = logging.getLogger(__name__)


class TranscriptionConfigurationError(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    text: str
    model: str
    raw_response: dict[str, Any]


class OpenAITranscriptionClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_TRANSCRIBE_MODEL
        if not self.api_key:
            raise TranscriptionConfigurationError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, segment: AudioSegment) -> TranscriptionResult:
        with default_storage.open(segment.audio_file.name, "rb") as audio_file:
            file_payload = (
                Path(segment.audio_file.name).name,
                audio_file.read(),
                segment.audio_content_type or "audio/wav",
            )
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=file_payload,
                response_format="json",
            )

        raw = self._serialize_response(response)
        text = raw.get("text") or getattr(response, "text", "")
        return TranscriptionResult(
            text=text,
            model=self.model,
            raw_response=raw,
        )

    @staticmethod
    def _serialize_response(response) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, dict):
            return response
        text = getattr(response, "text", "")
        return {"text": text}


def claim_next_pending_segment() -> AudioSegment | None:
    requeue_stale_segments()
    with transaction.atomic():
        queryset = AudioSegment.objects.select_related("meeting").filter(
            transcription_status=SegmentStatus.PENDING,
        )
        connection_features = transaction.get_connection().features
        if connection_features.has_select_for_update:
            if connection_features.has_select_for_update_skip_locked:
                queryset = queryset.select_for_update(skip_locked=True)
            else:
                queryset = queryset.select_for_update()

        segment = queryset.order_by(
            "meeting__started_at",
            "meeting_id",
            "sequence_number",
            "created_at",
        ).first()
        if segment is None:
            return None

        segment.transcription_status = SegmentStatus.PROCESSING
        segment.transcription_attempts += 1
        segment.transcription_started_at = timezone.now()
        segment.last_error = ""
        segment.save(
            update_fields=[
                "transcription_status",
                "transcription_attempts",
                "transcription_started_at",
                "last_error",
                "updated_at",
            ],
        )
        return segment


def requeue_stale_segments() -> int:
    stale_after = getattr(settings, "QUEUE_STALE_AFTER_SECONDS", 30 * 60)
    if stale_after <= 0:
        return 0

    cutoff = timezone.now() - timedelta(seconds=stale_after)
    return AudioSegment.objects.filter(
        transcription_status=SegmentStatus.PROCESSING,
        transcribed_at__isnull=True,
        transcription_started_at__lt=cutoff,
    ).update(
        transcription_status=SegmentStatus.PENDING,
        transcription_started_at=None,
        last_error="",
        updated_at=timezone.now(),
    )


def process_next_pending_segment(
    client: OpenAITranscriptionClient | None = None,
) -> AudioSegment | None:
    segment = claim_next_pending_segment()
    if segment is None:
        return None

    started_monotonic = time.monotonic()
    logger.info(
        "meeting_segment_transcription_started meeting_id=%s segment_id=%s sequence=%s "
        "attempt=%s audio_size_bytes=%s audio_duration_seconds=%.3f",
        segment.meeting_id,
        segment.id,
        segment.sequence_number,
        segment.transcription_attempts,
        segment.audio_size_bytes,
        segment_audio_duration_seconds(segment),
    )

    try:
        if client is None:
            client = OpenAITranscriptionClient()
        result = client.transcribe(segment)
    except Exception as exc:
        elapsed_seconds = time.monotonic() - started_monotonic
        segment.transcription_status = SegmentStatus.FAILED
        segment.last_error = str(exc)
        segment.save(
            update_fields=[
                "transcription_status",
                "last_error",
                "updated_at",
            ],
        )
        logger.exception(
            "meeting_segment_transcription_failed meeting_id=%s segment_id=%s sequence=%s "
            "attempt=%s elapsed_seconds=%.3f error=%s",
            segment.meeting_id,
            segment.id,
            segment.sequence_number,
            segment.transcription_attempts,
            elapsed_seconds,
            str(exc),
        )
    else:
        elapsed_seconds = time.monotonic() - started_monotonic
        segment.transcription_status = SegmentStatus.COMPLETE
        segment.transcription_text = result.text
        segment.transcription_model = result.model
        segment.transcription_response = result.raw_response
        segment.transcribed_at = timezone.now()
        segment.last_error = ""
        segment.save(
            update_fields=[
                "transcription_status",
                "transcription_text",
                "transcription_model",
                "transcription_response",
                "transcribed_at",
                "last_error",
                "updated_at",
            ],
        )
        logger.info(
            "meeting_segment_transcription_completed meeting_id=%s segment_id=%s sequence=%s "
            "attempt=%s model=%s elapsed_seconds=%.3f transcript_characters=%s",
            segment.meeting_id,
            segment.id,
            segment.sequence_number,
            segment.transcription_attempts,
            result.model,
            elapsed_seconds,
            len(result.text),
        )

    meeting = segment.meeting
    previous_meeting_status = meeting.status
    meeting.refresh_completion_status()
    meeting.refresh_from_db(fields=["status", "updated_at"])
    if (
        previous_meeting_status not in {MeetingStatus.COMPLETE, MeetingStatus.FAILED}
        and meeting.status in {MeetingStatus.COMPLETE, MeetingStatus.FAILED}
    ):
        log_meeting_transcription_finished(meeting)

    from .postprocessing import maybe_process_completed_meeting

    maybe_process_completed_meeting(meeting)
    return segment


def segment_audio_duration_seconds(segment: AudioSegment) -> float:
    duration_ms = max(0, int(segment.client_end_ms or 0) - int(segment.client_start_ms or 0))
    return duration_ms / 1000


def log_meeting_transcription_finished(meeting: Meeting) -> None:
    segments = list(
        meeting.segments.order_by("sequence_number").values(
            "transcription_status",
            "transcription_started_at",
            "transcribed_at",
            "client_start_ms",
            "client_end_ms",
        )
    )
    if not segments:
        return

    timestamps = meeting.segments.aggregate(
        first_started_at=Min("transcription_started_at"),
        last_transcribed_at=Max("transcribed_at"),
    )
    first_started_at = timestamps["first_started_at"]
    last_transcribed_at = timestamps["last_transcribed_at"]
    transcription_elapsed_seconds = None
    if first_started_at is not None and last_transcribed_at is not None:
        transcription_elapsed_seconds = (last_transcribed_at - first_started_at).total_seconds()

    complete_count = sum(1 for item in segments if item["transcription_status"] == SegmentStatus.COMPLETE)
    failed_count = sum(1 for item in segments if item["transcription_status"] == SegmentStatus.FAILED)
    audio_duration_ms = sum(
        max(0, int(item["client_end_ms"] or 0) - int(item["client_start_ms"] or 0))
        for item in segments
    )
    wall_since_meeting_end_seconds = None
    if meeting.ended_at is not None:
        wall_since_meeting_end_seconds = (timezone.now() - meeting.ended_at).total_seconds()

    logger.info(
        "meeting_transcription_finished meeting_id=%s status=%s segments_total=%s "
        "segments_complete=%s segments_failed=%s transcription_elapsed_seconds=%s "
        "wall_since_meeting_end_seconds=%s audio_duration_seconds=%.3f",
        meeting.id,
        meeting.status,
        len(segments),
        complete_count,
        failed_count,
        format_optional_seconds(transcription_elapsed_seconds),
        format_optional_seconds(wall_since_meeting_end_seconds),
        audio_duration_ms / 1000,
    )


def format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"


def run_transcription_loop(
    *,
    once: bool = False,
    limit: int | None = None,
    sleep_seconds: float = 2.0,
) -> int:
    processed = 0

    while True:
        from .import_processing import process_next_pending_import

        import_job = process_next_pending_import()
        if import_job is not None:
            processed += 1
            if limit is not None and processed >= limit:
                return processed
            continue

        segment = process_next_pending_segment()
        if segment is not None:
            processed += 1
            if limit is not None and processed >= limit:
                return processed
            continue

        from .minutes import process_next_pending_minutes

        minutes_job = process_next_pending_minutes()
        if minutes_job is not None:
            processed += 1
            if limit is not None and processed >= limit:
                return processed
            continue

        if once:
            return processed
        time.sleep(sleep_seconds)
