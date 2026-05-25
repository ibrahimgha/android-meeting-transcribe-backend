import base64
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.db.models import Count, Q
from django.utils.dateparse import parse_datetime
from django.utils import timezone

from .import_formats import SUPPORTED_IMPORT_AUDIO_EXTENSIONS, supported_import_audio_message
from .import_processing import process_next_pending_import
from .minutes import PM_NOTES_TYPES, generate_minutes_for_meeting, queue_minutes_for_meeting, sync_meeting_minutes_fields
from .models import AudioSegment, Meeting, MeetingImport, MeetingMinutesOutput, MeetingMinutesStatus, MeetingStatus, MeetingType
from .pdf import build_pm_notes_pdf
from .postprocessing import process_meeting_outputs
from .transcription import process_next_pending_segment
from .web_views import build_meeting_progress


class McpToolError(ValueError):
    pass


SUPPORTED_SEGMENT_AUDIO_EXTENSIONS = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}


def default_user():
    username = settings.MCP_DEFAULT_USERNAME
    if not username:
        raise McpToolError("MCP_DEFAULT_USERNAME is not configured.")

    User = get_user_model()
    try:
        return User.objects.get(username=username)
    except User.DoesNotExist as exc:
        raise McpToolError(f"Configured MCP user '{username}' does not exist.") from exc


def get_current_user() -> dict:
    user = default_user()
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
    }


def list_meeting_types() -> dict:
    return {
        "meeting_types": [
            {
                "value": value,
                "label": label,
                "supports_pdf": value in PM_NOTES_TYPES,
            }
            for value, label in MeetingType.choices
            if value != MeetingType.LUJY_PM_NOTES
        ]
    }


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
        .prefetch_related("imports", "minutes_outputs")
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


def start_meeting(*, title: str = "") -> dict:
    meeting = Meeting.objects.create(
        user=default_user(),
        title=title.strip()[:160],
    )
    return serialize_meeting_detail(meeting)


def end_meeting(meeting_id: str, *, ended_at: str = "", rebuild_outputs: bool = True) -> dict:
    meeting = get_user_meeting(meeting_id)
    if meeting.status != MeetingStatus.RECORDING:
        raise McpToolError("Meeting is not currently recording.")

    parsed_ended_at = None
    if ended_at:
        parsed_ended_at = parse_datetime(ended_at)
        if parsed_ended_at is None:
            raise McpToolError("ended_at must be an ISO-8601 datetime.")
        if timezone.is_naive(parsed_ended_at):
            parsed_ended_at = timezone.make_aware(parsed_ended_at)

    meeting.ended_at = parsed_ended_at or timezone.now()
    meeting.status = MeetingStatus.ENDED
    meeting.save(update_fields=["ended_at", "status", "updated_at"])
    meeting.refresh_completion_status()
    if rebuild_outputs:
        from .postprocessing import maybe_process_completed_meeting

        maybe_process_completed_meeting(meeting)
    meeting.refresh_from_db()
    return serialize_meeting_detail(meeting)


def get_meeting(meeting_id: str) -> dict:
    meeting = get_user_meeting(meeting_id)
    return serialize_meeting_detail(meeting)


def get_meeting_progress(meeting_id: str) -> dict:
    meeting = get_user_meeting(meeting_id)
    return build_meeting_progress(meeting)


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
    extension = Path(filename).suffix.lower()
    if extension.lstrip(".") not in SUPPORTED_IMPORT_AUDIO_EXTENSIONS:
        raise McpToolError(
            f"Unsupported recording format. Use one of: {supported_import_audio_message()}."
        )

    meeting_title = title.strip() or Path(filename).stem[:160] or "Imported meeting"
    max_bytes = settings.MAX_IMPORT_RECORDING_BYTES
    request = Request(recording_url, headers={"User-Agent": "meeting-transcribe-mcp/1.0"})

    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "audio/wav")
        declared_length = response.headers.get("Content-Length")
        if declared_length and int(declared_length) > max_bytes:
            raise McpToolError("Recording is larger than the configured import limit.")

        with tempfile.NamedTemporaryFile(suffix=extension) as temp_file:
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


def import_recording_from_base64(
    *,
    filename: str,
    content_base64: str,
    title: str = "",
    content_type: str = "",
) -> dict:
    user = default_user()
    filename = filename.strip()
    if not filename:
        raise McpToolError("filename is required.")
    extension = validate_extension(filename, SUPPORTED_IMPORT_AUDIO_EXTENSIONS, supported_import_audio_message())

    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise McpToolError("content_base64 must be valid base64.") from exc
    if not content:
        raise McpToolError("content_base64 is empty.")
    if len(content) > settings.MAX_IMPORT_RECORDING_BYTES:
        raise McpToolError("Recording is larger than the configured import limit.")

    meeting_title = title.strip() or Path(filename).stem[:160] or "Imported meeting"
    meeting = Meeting.objects.create(
        user=user,
        title=meeting_title,
        status=MeetingStatus.ENDED,
        ended_at=timezone.now(),
    )
    import_job = MeetingImport.objects.create(
        meeting=meeting,
        user=user,
        source_file=ContentFile(content, name=filename),
        original_filename=filename,
        content_type=content_type or content_type_for_extension(extension),
        size_bytes=len(content),
    )
    return {
        "meeting": serialize_meeting_summary(meeting),
        "import": serialize_import(import_job),
        "message": "Recording queued for background segmentation and transcription.",
    }


def upload_audio_segment_from_url(
    *,
    meeting_id: str,
    audio_url: str,
    sequence_number: int,
    speaker_label: str,
    client_start_ms: int,
    client_end_ms: int,
    client_segment_id: str = "",
    speaker_confidence: float | None = None,
    codec: str = "",
    sample_rate: int | None = None,
    original_filename: str = "",
    process_now: bool = False,
) -> dict:
    meeting = get_user_meeting(meeting_id)
    ensure_meeting_accepts_segments(meeting)

    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise McpToolError("audio_url must be an http or https URL.")

    filename = original_filename.strip() or Path(parsed.path).name or "segment.wav"
    extension = validate_extension(filename, SUPPORTED_SEGMENT_AUDIO_EXTENSIONS, ", ".join(sorted(SUPPORTED_SEGMENT_AUDIO_EXTENSIONS)))
    request = Request(audio_url, headers={"User-Agent": "meeting-transcribe-mcp/1.0"})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", content_type_for_extension(extension))
        declared_length = response.headers.get("Content-Length")
        if declared_length and int(declared_length) > settings.MAX_AUDIO_SEGMENT_BYTES:
            raise McpToolError("Audio segment is larger than the configured segment limit.")

        with tempfile.NamedTemporaryFile(suffix=extension) as temp_file:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.MAX_AUDIO_SEGMENT_BYTES:
                    raise McpToolError("Audio segment is larger than the configured segment limit.")
                temp_file.write(chunk)
            temp_file.flush()
            temp_file.seek(0)
            segment = create_audio_segment(
                meeting=meeting,
                audio_file=File(temp_file, name=filename),
                audio_size_bytes=total,
                audio_content_type=content_type,
                sequence_number=sequence_number,
                speaker_label=speaker_label,
                client_start_ms=client_start_ms,
                client_end_ms=client_end_ms,
                client_segment_id=client_segment_id,
                speaker_confidence=speaker_confidence,
                codec=codec,
                sample_rate=sample_rate,
            )

    if process_now:
        process_next_pending_segment()
        segment.refresh_from_db()

    return serialize_segment(segment)


def upload_audio_segment_from_base64(
    *,
    meeting_id: str,
    filename: str,
    content_base64: str,
    sequence_number: int,
    speaker_label: str,
    client_start_ms: int,
    client_end_ms: int,
    client_segment_id: str = "",
    speaker_confidence: float | None = None,
    codec: str = "",
    sample_rate: int | None = None,
    content_type: str = "",
    process_now: bool = False,
) -> dict:
    meeting = get_user_meeting(meeting_id)
    ensure_meeting_accepts_segments(meeting)
    filename = filename.strip()
    if not filename:
        raise McpToolError("filename is required.")
    extension = validate_extension(filename, SUPPORTED_SEGMENT_AUDIO_EXTENSIONS, ", ".join(sorted(SUPPORTED_SEGMENT_AUDIO_EXTENSIONS)))
    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise McpToolError("content_base64 must be valid base64.") from exc
    if not content:
        raise McpToolError("content_base64 is empty.")
    if len(content) > settings.MAX_AUDIO_SEGMENT_BYTES:
        raise McpToolError("Audio segment is larger than the configured segment limit.")

    segment = create_audio_segment(
        meeting=meeting,
        audio_file=ContentFile(content, name=filename),
        audio_size_bytes=len(content),
        audio_content_type=content_type or content_type_for_extension(extension),
        sequence_number=sequence_number,
        speaker_label=speaker_label,
        client_start_ms=client_start_ms,
        client_end_ms=client_end_ms,
        client_segment_id=client_segment_id,
        speaker_confidence=speaker_confidence,
        codec=codec,
        sample_rate=sample_rate,
    )
    if process_now:
        process_next_pending_segment()
        segment.refresh_from_db()

    return serialize_segment(segment)


def extract_meeting_minutes(meeting_id: str, meeting_type: str, *, wait: bool = False) -> dict:
    if meeting_type not in MeetingType.values:
        allowed = ", ".join(MeetingType.values)
        raise McpToolError(f"Unknown meeting_type '{meeting_type}'. Use one of: {allowed}.")

    meeting = get_user_meeting(meeting_id)
    meeting.meeting_type = meeting_type
    meeting.save(update_fields=["meeting_type", "updated_at"])
    output = MeetingMinutesOutput.objects.filter(meeting=meeting, meeting_type=meeting_type).first()
    if output and output.status == MeetingMinutesStatus.COMPLETE and output.text.strip() and not wait:
        sync_meeting_minutes_fields(meeting, output)
    elif output and output.status in {MeetingMinutesStatus.PENDING, MeetingMinutesStatus.PROCESSING} and not wait:
        sync_meeting_minutes_fields(meeting, output)
    elif wait:
        output = MeetingMinutesOutput.objects.filter(meeting=meeting, meeting_type=meeting_type).first()
        generate_minutes_for_meeting(meeting, output=output)
    else:
        queue_minutes_for_meeting(meeting)
    meeting.refresh_from_db()
    return {
        "meeting": serialize_meeting_summary(meeting),
        "meeting_type": meeting.meeting_type,
        "minutes_status": meeting.minutes_status,
        "minutes_text": meeting.minutes_text,
        "minutes_model": meeting.minutes_model,
        "minutes_generated_at": meeting.minutes_generated_at.isoformat()
        if meeting.minutes_generated_at
        else None,
    }


def get_project_manager_notes_pdf(meeting_id: str, meeting_type: str = "") -> dict:
    meeting = get_user_meeting(meeting_id)
    requested_type = meeting_type or meeting.meeting_type
    if requested_type not in PM_NOTES_TYPES:
        raise McpToolError("Project manager notes PDF is only available for PM notes meeting outputs.")
    output = MeetingMinutesOutput.objects.filter(
        meeting=meeting,
        meeting_type=requested_type,
        status=MeetingMinutesStatus.COMPLETE,
    ).first()
    if output is None or not output.text.strip():
        raise McpToolError("Project manager notes PDF is only available for generated PM notes outputs.")

    pdf_bytes = build_pm_notes_pdf(meeting, minutes_text=output.text)
    filename = f"{safe_filename(meeting.title or 'meeting-notes')}-pm-notes.pdf"
    return {
        "meeting_id": str(meeting.id),
        "filename": filename,
        "content_type": "application/pdf",
        "content_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "size_bytes": len(pdf_bytes),
    }


def rebuild_messages_and_summaries(meeting_id: str) -> dict:
    meeting = get_user_meeting(meeting_id)
    process_meeting_outputs(meeting, force=True)
    meeting.refresh_from_db()
    return serialize_meeting_detail(meeting)


def process_one_transcription_job() -> dict:
    segment = process_next_pending_segment()
    if segment is None:
        return {"processed": False, "message": "No pending audio segments."}
    segment.refresh_from_db()
    return {
        "processed": True,
        "segment": serialize_segment(segment),
        "meeting": serialize_meeting_summary(segment.meeting),
    }


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
            .prefetch_related("imports", "segments", "minutes_outputs", "messages__segments")
            .get(id=meeting_id)
        )
    except Meeting.DoesNotExist as exc:
        raise McpToolError("Meeting was not found for the configured MCP user.") from exc


def ensure_meeting_accepts_segments(meeting: Meeting) -> None:
    if meeting.status not in [MeetingStatus.RECORDING, MeetingStatus.ENDED]:
        raise McpToolError("Meeting is no longer accepting audio segments.")


def create_audio_segment(
    *,
    meeting: Meeting,
    audio_file,
    audio_size_bytes: int,
    audio_content_type: str,
    sequence_number: int,
    speaker_label: str,
    client_start_ms: int,
    client_end_ms: int,
    client_segment_id: str = "",
    speaker_confidence: float | None = None,
    codec: str = "",
    sample_rate: int | None = None,
) -> AudioSegment:
    if int(client_end_ms) <= int(client_start_ms):
        raise McpToolError("client_end_ms must be greater than client_start_ms.")
    if not speaker_label.strip():
        raise McpToolError("speaker_label is required.")
    try:
        return AudioSegment.objects.create(
            meeting=meeting,
            user=meeting.user,
            client_segment_id=client_segment_id.strip(),
            sequence_number=int(sequence_number),
            speaker_label=speaker_label.strip(),
            speaker_confidence=speaker_confidence,
            client_start_ms=int(client_start_ms),
            client_end_ms=int(client_end_ms),
            codec=codec.strip(),
            sample_rate=sample_rate,
            audio_file=audio_file,
            audio_content_type=audio_content_type,
            audio_size_bytes=audio_size_bytes,
        )
    except IntegrityError as exc:
        raise McpToolError("sequence_number already exists for this meeting.") from exc


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
        "minutes_status": meeting.minutes_status,
        "minutes_requested_at": meeting.minutes_requested_at.isoformat()
        if meeting.minutes_requested_at
        else None,
        "minutes_started_at": meeting.minutes_started_at.isoformat()
        if meeting.minutes_started_at
        else None,
        "minutes_generated_at": meeting.minutes_generated_at.isoformat()
        if meeting.minutes_generated_at
        else None,
        "minutes_last_error": meeting.minutes_last_error,
        "available_minutes_outputs": [
            serialize_minutes_output(output)
            for output in meeting.minutes_outputs.all().order_by("meeting_type")
        ],
        "imports": import_items,
    }


def serialize_meeting_detail(meeting: Meeting) -> dict:
    summary = serialize_meeting_summary(meeting)
    summary["segments"] = [
        serialize_segment(segment)
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
    outputs = [
        serialize_minutes_output(output)
        for output in meeting.minutes_outputs.all().order_by("meeting_type")
    ]
    summary["minutes_outputs"] = outputs
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


def serialize_minutes_output(output: MeetingMinutesOutput) -> dict:
    return {
        "id": str(output.id),
        "meeting_type": output.meeting_type,
        "meeting_type_label": output.get_meeting_type_display(),
        "status": output.status,
        "text": output.text,
        "model": output.model,
        "requested_at": output.requested_at.isoformat() if output.requested_at else None,
        "started_at": output.started_at.isoformat() if output.started_at else None,
        "generated_at": output.generated_at.isoformat() if output.generated_at else None,
        "last_error": output.last_error,
        "supports_pdf": output.meeting_type in PM_NOTES_TYPES,
    }


def serialize_segment(segment: AudioSegment) -> dict:
    return {
        "id": str(segment.id),
        "meeting_id": str(segment.meeting_id),
        "sequence_number": segment.sequence_number,
        "client_segment_id": segment.client_segment_id,
        "speaker_label": segment.speaker_label,
        "speaker_confidence": segment.speaker_confidence,
        "client_start_ms": segment.client_start_ms,
        "client_end_ms": segment.client_end_ms,
        "codec": segment.codec,
        "sample_rate": segment.sample_rate,
        "audio_content_type": segment.audio_content_type,
        "audio_size_bytes": segment.audio_size_bytes,
        "transcription_status": segment.transcription_status,
        "transcription_text": segment.transcription_text,
        "transcription_model": segment.transcription_model,
        "last_error": segment.last_error,
        "audio_url": absolute_media_url(segment.audio_file.url) if segment.audio_file else "",
    }


def serialize_import(import_job: MeetingImport) -> dict:
    return {
        "id": str(import_job.id),
        "meeting_id": str(import_job.meeting_id),
        "original_filename": import_job.original_filename,
        "content_type": import_job.content_type,
        "size_bytes": import_job.size_bytes,
        "status": import_job.status,
        "progress_percent": import_job.progress_percent,
        "progress_message": import_job.progress_message,
        "created_segments": import_job.created_segments,
        "started_at": import_job.started_at.isoformat() if import_job.started_at else None,
        "processed_at": import_job.processed_at.isoformat() if import_job.processed_at else None,
        "last_error": import_job.last_error,
    }


def absolute_media_url(path: str) -> str:
    public = settings.MCP_PUBLIC_URL.rsplit("/mcp", 1)[0].rstrip("/")
    return f"{public}{path}"


def validate_extension(filename: str, allowed: set[str], allowed_message: str) -> str:
    extension = Path(filename).suffix.lower()
    if not extension:
        raise McpToolError(f"Filename must include an extension. Use one of: {allowed_message}.")
    if extension.lstrip(".") not in allowed:
        raise McpToolError(f"Unsupported format '{extension.lstrip('.')}'. Use one of: {allowed_message}.")
    return extension


def content_type_for_extension(extension: str) -> str:
    return {
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
    }.get(extension.lower(), "application/octet-stream")


def safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "meeting-notes"
