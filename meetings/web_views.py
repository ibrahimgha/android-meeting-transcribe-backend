import json
import shutil
from pathlib import Path
from uuid import uuid4

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.conf import settings
from django.core.files import File
from django.http import JsonResponse
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, ListView

from .forms import MeetingImportForm, MeetingMinutesForm
from .import_formats import SUPPORTED_IMPORT_AUDIO_EXTENSIONS, supported_import_audio_message
from .minutes import generate_minutes_for_meeting
from .postprocessing import process_meeting_outputs
from .models import Meeting, MeetingImport, MeetingStatus


class MeetingListView(LoginRequiredMixin, ListView):
    template_name = "meetings/meeting_list.html"
    context_object_name = "meetings"

    def get_queryset(self):
        return (
            Meeting.objects.filter(user=self.request.user)
            .annotate(
                segment_count=Count("segments"),
                completed_transcription_count=Count(
                    "segments",
                    filter=Q(segments__transcription_text__gt=""),
                ),
            )
            .prefetch_related("segments")
            .order_by("-started_at")
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["import_form"] = MeetingImportForm()
        context["import_chunk_bytes"] = settings.IMPORT_CHUNK_BYTES
        return context


class MeetingDetailView(LoginRequiredMixin, DetailView):
    template_name = "meetings/meeting_detail.html"
    context_object_name = "meeting"

    def get_queryset(self):
        return (
            Meeting.objects.filter(user=self.request.user)
            .annotate(
                segment_count=Count("segments"),
                completed_transcription_count=Count(
                    "segments",
                    filter=Q(segments__transcription_text__gt=""),
                ),
            )
            .prefetch_related(
                "segments",
                "imports",
                "messages__segments",
            )
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["minutes_form"] = MeetingMinutesForm(instance=self.object)
        return context


class ImportMeetingRecordingView(LoginRequiredMixin, View):
    def post(self, request):
        form = MeetingImportForm(request.POST, request.FILES)
        if not form.is_valid():
            first_error = next(iter(form.errors.values()))[0]
            messages.error(request, f"Could not import recording: {first_error}")
            return redirect("web-meetings")

        recording_file = form.cleaned_data["recording_file"]
        title = form.cleaned_data.get("title", "").strip()
        if not title:
            title = recording_file.name.rsplit(".", 1)[0][:160] or "Imported meeting"

        meeting = Meeting.objects.create(
            user=request.user,
            title=title,
            status=MeetingStatus.ENDED,
            ended_at=timezone.now(),
        )
        MeetingImport.objects.create(
            meeting=meeting,
            user=request.user,
            source_file=recording_file,
            original_filename=recording_file.name,
            content_type=getattr(recording_file, "content_type", "") or "",
            size_bytes=recording_file.size,
        )
        messages.success(request, "Recording uploaded. It will be segmented and transcribed in the background.")
        return redirect("web-meeting-detail", pk=meeting.pk)


class StartChunkedImportView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse({"detail": "Invalid JSON payload."}, status=400)

        filename = str(payload.get("filename", "")).strip()
        title = str(payload.get("title", "")).strip()
        content_type = str(payload.get("content_type", "")).strip()
        try:
            total_size = int(payload.get("total_size", 0))
            total_chunks = int(payload.get("total_chunks", 0))
            chunk_size = int(payload.get("chunk_size", settings.IMPORT_CHUNK_BYTES))
        except (TypeError, ValueError):
            return JsonResponse({"detail": "Invalid chunk metadata."}, status=400)

        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in SUPPORTED_IMPORT_AUDIO_EXTENSIONS:
            return JsonResponse(
                {"detail": f"Unsupported recording format. Use one of: {supported_import_audio_message()}."},
                status=400,
            )
        if total_size <= 0 or total_size > settings.MAX_IMPORT_RECORDING_BYTES:
            return JsonResponse({"detail": "Recording is larger than the configured import limit."}, status=400)
        if total_chunks <= 0 or chunk_size <= 0 or chunk_size > settings.IMPORT_CHUNK_BYTES:
            return JsonResponse({"detail": "Invalid chunk size."}, status=400)

        upload_id = uuid4().hex
        upload_dir(upload_id).mkdir(parents=True, exist_ok=True)
        uploads = request.session.get("meeting_import_uploads", {})
        uploads[upload_id] = {
            "filename": filename,
            "title": title,
            "content_type": content_type,
            "total_size": total_size,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "received": [],
        }
        request.session["meeting_import_uploads"] = uploads
        request.session.modified = True
        return JsonResponse(
            {
                "upload_id": upload_id,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
            },
            status=201,
        )


class UploadImportChunkView(LoginRequiredMixin, View):
    def post(self, request, upload_id):
        metadata = import_upload_metadata(request, upload_id)
        if metadata is None:
            return JsonResponse({"detail": "Upload session was not found."}, status=404)

        try:
            index = int(request.POST.get("index", ""))
        except ValueError:
            return JsonResponse({"detail": "Invalid chunk index."}, status=400)

        total_chunks = metadata["total_chunks"]
        if index < 0 or index >= total_chunks:
            return JsonResponse({"detail": "Chunk index is out of range."}, status=400)

        chunk = request.FILES.get("chunk")
        if chunk is None:
            return JsonResponse({"detail": "Missing chunk file."}, status=400)
        if chunk.size > settings.IMPORT_CHUNK_BYTES:
            return JsonResponse({"detail": "Chunk is larger than the configured chunk limit."}, status=400)

        path = upload_dir(upload_id) / f"{index:06d}.part"
        with path.open("wb") as destination:
            for piece in chunk.chunks():
                destination.write(piece)

        received = set(metadata.get("received", []))
        received.add(index)
        metadata["received"] = sorted(received)
        save_import_upload_metadata(request, upload_id, metadata)
        return JsonResponse(
            {
                "received_chunks": len(metadata["received"]),
                "total_chunks": total_chunks,
            }
        )


class FinishChunkedImportView(LoginRequiredMixin, View):
    def post(self, request, upload_id):
        metadata = import_upload_metadata(request, upload_id)
        if metadata is None:
            return JsonResponse({"detail": "Upload session was not found."}, status=404)

        total_chunks = metadata["total_chunks"]
        missing = [
            index
            for index in range(total_chunks)
            if not (upload_dir(upload_id) / f"{index:06d}.part").exists()
        ]
        if missing:
            return JsonResponse({"detail": f"Missing chunks: {missing[:10]}"}, status=400)

        source_path = upload_dir(upload_id) / f"assembled{Path(metadata['filename']).suffix.lower()}"
        total_size = 0
        with source_path.open("wb") as assembled:
            for index in range(total_chunks):
                chunk_path = upload_dir(upload_id) / f"{index:06d}.part"
                total_size += chunk_path.stat().st_size
                with chunk_path.open("rb") as source:
                    shutil.copyfileobj(source, assembled)

        expected_size = metadata["total_size"]
        if total_size != expected_size:
            return JsonResponse(
                {"detail": f"Assembled file size mismatch. Expected {expected_size}, got {total_size}."},
                status=400,
            )

        filename = metadata["filename"]
        title = metadata.get("title") or Path(filename).stem[:160] or "Imported meeting"
        meeting = Meeting.objects.create(
            user=request.user,
            title=title,
            status=MeetingStatus.ENDED,
            ended_at=timezone.now(),
        )
        with source_path.open("rb") as source:
            MeetingImport.objects.create(
                meeting=meeting,
                user=request.user,
                source_file=File(source, name=filename),
                original_filename=filename,
                content_type=metadata.get("content_type", "") or "audio/wav",
                size_bytes=total_size,
            )

        clear_import_upload_metadata(request, upload_id)
        shutil.rmtree(upload_dir(upload_id), ignore_errors=True)
        return JsonResponse(
            {
                "meeting_id": str(meeting.id),
                "meeting_url": reverse("web-meeting-detail", kwargs={"pk": meeting.pk}),
            },
            status=201,
        )


class GenerateMeetingMinutesView(LoginRequiredMixin, View):
    def post(self, request, pk):
        meeting = get_object_or_404(Meeting, pk=pk, user=request.user)
        form = MeetingMinutesForm(request.POST, instance=meeting)
        if not form.is_valid():
            messages.error(request, "Choose a meeting type before extracting minutes.")
            return redirect("web-meeting-detail", pk=meeting.pk)

        form.save()
        try:
            generate_minutes_for_meeting(meeting)
        except Exception as exc:
            messages.error(request, f"Could not extract minutes: {exc}")
        else:
            messages.success(request, "Meeting minutes extracted.")

        return redirect("web-meeting-detail", pk=meeting.pk)


def upload_dir(upload_id: str) -> Path:
    return Path(settings.MEDIA_ROOT) / "meeting_import_chunks" / upload_id


def import_upload_metadata(request, upload_id: str) -> dict | None:
    uploads = request.session.get("meeting_import_uploads", {})
    return uploads.get(upload_id)


def save_import_upload_metadata(request, upload_id: str, metadata: dict) -> None:
    uploads = request.session.get("meeting_import_uploads", {})
    uploads[upload_id] = metadata
    request.session["meeting_import_uploads"] = uploads
    request.session.modified = True


def clear_import_upload_metadata(request, upload_id: str) -> None:
    uploads = request.session.get("meeting_import_uploads", {})
    uploads.pop(upload_id, None)
    request.session["meeting_import_uploads"] = uploads
    request.session.modified = True


class GenerateMeetingOutputsView(LoginRequiredMixin, View):
    def post(self, request, pk):
        meeting = get_object_or_404(Meeting, pk=pk, user=request.user)
        try:
            process_meeting_outputs(meeting, force=True)
        except Exception as exc:
            messages.error(request, f"Could not rebuild messages and summaries: {exc}")
        else:
            messages.success(request, "Messages, summaries, and title rebuilt.")
        return redirect("web-meeting-detail", pk=meeting.pk)
