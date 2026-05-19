import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.db.models import Count, Q
from django.utils import timezone

from .import_processing import process_next_pending_import
from .minutes import generate_minutes_for_meeting
from .models import Meeting, MeetingImport, MeetingStatus, MeetingType
from .postprocessing import process_meeting_outputs


class McpToolError(ValueError):
    pass


def default_user():
    username = settings.MCP_DEFAULT_USERNAME
    if not username:
        raise McpToolError("MCP_DEFAULT_USERNAME is not configured.")

    User = get_user_model()
    try:
        return User.objects.get(username=username)
    except User.DoesNotExist as exc:
        raise McpToolError(f"Configured MCP user '{username}' does not exist.") from exc


def list_meetings(*, limit: int = 20, status: str = "") -> dict:
    user = default_user()
    queryset = (
        Meeting.objects.filter(user=user)
        .annotate(
            segment_count=Count("segments"),
            completed_transcription_count=Count(
                "segments",
                filter=Q(segments__transcription_text__gt=""),
            ),
        )
        .prefetch_related("imports")
        .order_by("-started_at")
    )
    if status:
        if status not in MeetingStatus.values:
            raise McpToolError(f"Unknown meeting status '{status}'.")
        queryset = queryset.filter(status=status)

    limit = max(1, min(int(limit), 100))
    return {
        "user": user.username,
        "meetings": [serialize_meeting_summary(meeting) for meeting in queryset[:limit]],
    }


def get_meeting(meeting_id: str) -> dict:
    meeting = get_user_meeting(meeting_id)
    return serialize_meeting_detail(meeting)


def import_recording_from_url(
    *,
    recording_url: str,
    title: str = "",
    original_filename: str = "",
    process_now: bool = False,
) -> dict:
    user = default_user()
    parsed = urlparse(recording_url)
    if parsed.scheme not in {"http", "https"}:
        raise McpToolError("recording_url must be an http or https URL.")

    filename = original_filename.strip() or Path(parsed.path).name or "imported-recording.wav"
    if Path(filename).suffix.lower() != ".wav":
        raise McpToolError("Only WAV recordings are supported for imports.")

    meeting_title = title.strip() or Path(filename).stem[:160] or "Imported meeting"
    max_bytes = settings.MAX_IMPORT_RECORDING_BYTES
    request = Request(recording_url, headers={"User-Agent": "meeting-transcribe-mcp/1.0"})

    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "audio/wav")
        declared_length = response.headers.get("Content-Length")
        if declared_length and int(declared_length) > max_bytes:
            raise McpToolError("Recording is larger than the configured import limit.")

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise McpToolError("Recording is larger than the configured import limit.")
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.seek(0)

            meeting = Meeting.objects.create(
                user=user,
                title=meeting_title,
                status=MeetingStatus.ENDED,
                ended_at=timezone.now(),
            )
            import_job = MeetingImport.objects.create(
                meeting=meeting,
                user=user,
                source_file=File(temp_file, name=filename),
                original_filename=filename,
                content_type=content_type,
                size_bytes=total,
            )

    if process_now:
        process_next_pending_import()
        import_job.refresh_from_db()
        meeting.refresh_from_db()

    return {
        "meeting": serialize_meeting_summary(meeting),
        "import": serialize_import(import_job),
        "message": "Recording queued for background segmentation and transcription.",
    }


def extract_meeting_minutes(meeting_id: str, meeting_type: str) -> dict:
    if meeting_type not in MeetingType.values:
        allowed = ", ".join(MeetingType.values)
        raise McpToolError(f"Unknown meeting_type '{meeting_type}'. Use one of: {allowed}.")

    meeting = get_user_meeting(meeting_id)
    meeting.meeting_type = meeting_type
    meeting.save(update_fields=["meeting_type", "updated_at"])
    generate_minutes_for_meeting(meeting)
    meeting.refresh_from_db()
    return {
        "meeting": serialize_meeting_summary(meeting),
        "meeting_type": meeting.meeting_type,
        "minutes_text": meeting.minutes_text,
        "minutes_model": meeting.minutes_model,
        "minutes_generated_at": meeting.minutes_generated_at.isoformat()
        if meeting.minutes_generated_at
        else None,
    }


def rebuild_messages_and_summaries(meeting_id: str) -> dict:
    meeting = get_user_meeting(meeting_id)
    process_meeting_outputs(meeting, force=True)
    meeting.refresh_from_db()
    return serialize_meeting_detail(meeting)


def process_one_import_job() -> dict:
    import_job = process_next_pending_import()
    if import_job is None:
        return {"processed": False, "message": "No pending meeting imports."}
    import_job.refresh_from_db()
    return {
        "processed": True,
        "import": serialize_import(import_job),
        "meeting": serialize_meeting_summary(import_job.meeting),
    }


def get_user_meeting(meeting_id: str) -> Meeting:
    user = default_user()
    try:
        return (
            Meeting.objects.filter(user=user)
            .prefetch_related("imports", "segments", "messages__segments")
            .get(id=meeting_id)
        )
    except Meeting.DoesNotExist as exc:
        raise McpToolError("Meeting was not found for the configured MCP user.") from exc


def serialize_meeting_summary(meeting: Meeting) -> dict:
    import_items = [serialize_import(item) for item in meeting.imports.all()]
    segment_count = getattr(meeting, "segment_count", None)
    if segment_count is None:
        segment_count = meeting.segments.count()
    completed_count = getattr(meeting, "completed_transcription_count", None)
    if completed_count is None:
        completed_count = meeting.segments.exclude(transcription_text="").count()

    return {
        "id": str(meeting.id),
        "title": meeting.title or "Mobile meeting",
        "status": meeting.status,
        "started_at": meeting.started_at.isoformat(),
        "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
        "segment_count": segment_count,
        "completed_transcription_count": completed_count,
        "message_processing_status": meeting.output_status,
        "meeting_type": meeting.meeting_type,
        "minutes_generated_at": meeting.minutes_generated_at.isoformat()
        if meeting.minutes_generated_at
        else None,
        "imports": import_items,
    }


def serialize_meeting_detail(meeting: Meeting) -> dict:
    summary = serialize_meeting_summary(meeting)
    summary["segments"] = [
        {
            "id": str(segment.id),
            "sequence_number": segment.sequence_number,
            "speaker_label": segment.speaker_label,
            "speaker_confidence": segment.speaker_confidence,
            "client_start_ms": segment.client_start_ms,
            "client_end_ms": segment.client_end_ms,
            "transcription_status": segment.transcription_status,
            "transcription_text": segment.transcription_text,
            "audio_url": absolute_media_url(segment.audio_file.url) if segment.audio_file else "",
        }
        for segment in meeting.segments.all().order_by("sequence_number")
    ]
    summary["messages"] = [
        {
            "id": str(message.id),
            "sequence_number": message.sequence_number,
            "speaker_label": message.speaker_label,
            "client_start_ms": message.client_start_ms,
            "client_end_ms": message.client_end_ms,
            "transcript_text": message.transcript_text,
            "detailed_summary": message.detailed_summary,
            "short_summary": message.short_summary,
            "segment_sequence_numbers": [
                segment.sequence_number for segment in message.segments.all().order_by("sequence_number")
            ],
        }
        for message in meeting.messages.all().order_by("sequence_number")
    ]
    if meeting.minutes_text:
        summary["minutes"] = {
            "meeting_type": meeting.meeting_type,
            "text": meeting.minutes_text,
            "model": meeting.minutes_model,
            "generated_at": meeting.minutes_generated_at.isoformat()
            if meeting.minutes_generated_at
            else None,
        }
    else:
        summary["minutes"] = None
    return summary


def serialize_import(import_job: MeetingImport) -> dict:
    return {
        "id": str(import_job.id),
        "original_filename": import_job.original_filename,
        "status": import_job.status,
        "created_segments": import_job.created_segments,
        "started_at": import_job.started_at.isoformat() if import_job.started_at else None,
        "processed_at": import_job.processed_at.isoformat() if import_job.processed_at else None,
        "last_error": import_job.last_error,
    }


def absolute_media_url(path: str) -> str:
    public = settings.MCP_PUBLIC_URL.rsplit("/mcp", 1)[0].rstrip("/")
    return f"{public}{path}"
