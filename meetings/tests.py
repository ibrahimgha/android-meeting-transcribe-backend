import base64
import io
import json
import math
import wave
from array import array
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.test import Client
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from .minutes import (
    MinutesResult,
    build_minutes_prompt,
    build_project_manager_compaction_prompt,
    build_project_manager_final_prompt,
    chunk_transcript,
    generate_minutes_for_meeting,
)
from .import_processing import claim_next_pending_import, process_next_pending_import
from . import mcp_api
from .models import (
    AudioSegment,
    Meeting,
    MeetingImport,
    MeetingImportStatus,
    MeetingMessage,
    MeetingOutputStatus,
    MeetingStatus,
    MeetingType,
    SegmentStatus,
)
from .openai_utils import chat_completion_options
from .postprocessing import MessageDraft, TextResult, process_meeting_outputs
from .transcription import TranscriptionResult, claim_next_pending_segment, process_next_pending_segment

User = get_user_model()


def wav_bytes() -> bytes:
    return b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def voiced_wav_bytes(sample_rate=16_000) -> bytes:
    output = io.BytesIO()
    parts = [
        (1.25, 440.0, 0.24),
        (0.35, None, 0.0),
        (1.25, 660.0, 0.22),
    ]
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for seconds, frequency, amplitude in parts:
            sample_count = int(seconds * sample_rate)
            for index in range(sample_count):
                value = 0.0
                if frequency is not None:
                    value = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
                frames.extend(int(value * 32767).to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(frames))
    return output.getvalue()


def voiced_samples(sample_rate=16_000) -> array:
    samples = array("f")
    for index in range(int(1.5 * sample_rate)):
        value = 0.24 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate)
        samples.append(value)
    return samples


class MeetingApiTests(APITestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.temp_dir.name)
        self.settings_override.enable()
        self.user = User.objects.create_user(
            username="mobile-user",
            email="mobile@example.com",
            password="strong-password-123",
        )
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {self.token.key}")

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    def test_register_returns_token(self):
        self.client.credentials()
        response = self.client.post(
            "/api/auth/register/",
            {
                "username": "new-user",
                "email": "new@example.com",
                "password": "strong-password-456",
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", response.data)
        self.assertEqual(response.data["user"]["username"], "new-user")

    def test_start_upload_and_end_meeting(self):
        start_response = self.client.post("/api/meetings/start/", {"title": "Weekly sync"})

        self.assertEqual(start_response.status_code, status.HTTP_201_CREATED)
        meeting_id = start_response.data["id"]
        meeting = Meeting.objects.get(id=meeting_id)
        self.assertEqual(meeting.user, self.user)
        self.assertEqual(meeting.status, MeetingStatus.RECORDING)

        upload_response = self.client.post(
            f"/api/meetings/{meeting_id}/segments/",
            {
                "sequence_number": 1,
                "speaker_label": "person_1",
                "speaker_confidence": "0.91",
                "client_start_ms": 100,
                "client_end_ms": 1200,
                "codec": "wav_pcm16",
                "sample_rate": 16000,
                "audio_file": ContentFile(wav_bytes(), name="seg_000001.wav"),
            },
            format="multipart",
        )

        self.assertEqual(upload_response.status_code, status.HTTP_201_CREATED)
        segment = AudioSegment.objects.get(meeting=meeting)
        self.assertEqual(segment.user, self.user)
        self.assertEqual(segment.transcription_status, SegmentStatus.PENDING)
        self.assertEqual(segment.speaker_label, "person_1")

        duplicate_response = self.client.post(
            f"/api/meetings/{meeting_id}/segments/",
            {
                "sequence_number": 1,
                "speaker_label": "person_1",
                "client_start_ms": 1400,
                "client_end_ms": 2400,
                "audio_file": ContentFile(wav_bytes(), name="seg_000001_again.wav"),
            },
            format="multipart",
        )
        self.assertEqual(duplicate_response.status_code, status.HTTP_400_BAD_REQUEST)

        end_response = self.client.post(f"/api/meetings/{meeting_id}/end/", {})

        self.assertEqual(end_response.status_code, status.HTTP_200_OK)
        meeting.refresh_from_db()
        self.assertEqual(meeting.status, MeetingStatus.ENDED)
        self.assertIsNotNone(meeting.ended_at)

    def test_user_cannot_access_another_users_meeting(self):
        other_user = User.objects.create_user(
            username="other",
            email="other@example.com",
            password="strong-password-123",
        )
        meeting = Meeting.objects.create(user=other_user)

        response = self.client.post(f"/api/meetings/{meeting.id}/end/", {})

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_import_recording_creates_pending_background_job(self):
        response = self.client.post(
            "/api/meetings/import/",
            {
                "title": "Imported workshop",
                "recording_file": ContentFile(voiced_wav_bytes(), name="workshop.wav"),
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        meeting = Meeting.objects.get(id=response.data["meeting"]["id"])
        import_job = MeetingImport.objects.get(meeting=meeting)
        self.assertEqual(meeting.user, self.user)
        self.assertEqual(meeting.status, MeetingStatus.ENDED)
        self.assertEqual(meeting.title, "Imported workshop")
        self.assertEqual(import_job.status, MeetingImportStatus.PENDING)
        self.assertEqual(meeting.segments.count(), 0)

    def test_import_recording_accepts_mp3_m4a_and_mp4(self):
        for extension in ["mp3", "m4a", "mp4"]:
            response = self.client.post(
                "/api/meetings/import/",
                {
                    "title": f"Imported {extension}",
                    "recording_file": ContentFile(b"fake compressed audio", name=f"workshop.{extension}"),
                },
                format="multipart",
            )

            self.assertEqual(response.status_code, status.HTTP_201_CREATED)
            meeting = Meeting.objects.get(id=response.data["meeting"]["id"])
            import_job = MeetingImport.objects.get(meeting=meeting)
            self.assertEqual(import_job.original_filename, f"workshop.{extension}")
            self.assertEqual(import_job.status, MeetingImportStatus.PENDING)


class TranscriptionQueueTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.temp_dir.name)
        self.settings_override.enable()
        self.user = User.objects.create_user(username="queue-user")
        self.meeting = Meeting.objects.create(user=self.user, status=MeetingStatus.ENDED)

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    def make_segment(self, sequence_number: int) -> AudioSegment:
        return AudioSegment.objects.create(
            meeting=self.meeting,
            user=self.user,
            sequence_number=sequence_number,
            speaker_label=f"person_{sequence_number}",
            client_start_ms=sequence_number * 1000,
            client_end_ms=(sequence_number * 1000) + 500,
            audio_file=ContentFile(wav_bytes(), name=f"seg_{sequence_number}.wav"),
            audio_size_bytes=len(wav_bytes()),
            audio_content_type="audio/wav",
        )

    def test_processes_segments_in_sequence_order(self):
        self.make_segment(2)
        self.make_segment(1)
        fake_client = FakeTranscriptionClient()

        first = process_next_pending_segment(client=fake_client)
        second = process_next_pending_segment(client=fake_client)

        self.assertEqual([first.sequence_number, second.sequence_number], [1, 2])
        self.assertEqual(fake_client.sequences, [1, 2])
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.status, MeetingStatus.COMPLETE)

    def test_returns_none_when_queue_empty_without_openai_key(self):
        self.assertIsNone(process_next_pending_segment())

    @override_settings(QUEUE_STALE_AFTER_SECONDS=1)
    def test_requeues_stale_processing_segment(self):
        segment = self.make_segment(1)
        segment.transcription_status = SegmentStatus.PROCESSING
        segment.transcription_started_at = timezone.now() - timedelta(minutes=5)
        segment.save(update_fields=["transcription_status", "transcription_started_at", "updated_at"])

        claimed = claim_next_pending_segment()

        self.assertEqual(claimed.id, segment.id)
        claimed.refresh_from_db()
        self.assertEqual(claimed.transcription_status, SegmentStatus.PROCESSING)
        self.assertEqual(claimed.transcription_attempts, 1)


class MeetingImportQueueTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.temp_dir.name)
        self.settings_override.enable()
        self.user = User.objects.create_user(username="import-user")
        self.meeting = Meeting.objects.create(
            user=self.user,
            title="Imported audio",
            status=MeetingStatus.ENDED,
        )

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    def test_process_next_pending_import_creates_pending_segments(self):
        import_job = MeetingImport.objects.create(
            meeting=self.meeting,
            user=self.user,
            source_file=ContentFile(voiced_wav_bytes(), name="source.wav"),
            original_filename="source.wav",
            content_type="audio/wav",
            size_bytes=len(voiced_wav_bytes()),
        )

        processed = process_next_pending_import()

        self.assertEqual(processed.id, import_job.id)
        import_job.refresh_from_db()
        self.meeting.refresh_from_db()
        self.assertEqual(import_job.status, MeetingImportStatus.COMPLETE)
        self.assertGreaterEqual(import_job.created_segments, 1)
        self.assertEqual(self.meeting.status, MeetingStatus.ENDED)
        self.assertIsNotNone(self.meeting.ended_at)
        segment = self.meeting.segments.order_by("sequence_number").first()
        self.assertIsNotNone(segment)
        self.assertEqual(segment.transcription_status, SegmentStatus.PENDING)
        self.assertEqual(segment.codec, "wav_pcm16")
        self.assertTrue(segment.audio_file.name.endswith(".wav"))

    @patch("meetings.import_processing.decode_with_ffmpeg")
    def test_process_next_pending_import_decodes_compressed_audio(self, decoder):
        decoder.return_value = (voiced_samples(), 16000)
        import_job = MeetingImport.objects.create(
            meeting=self.meeting,
            user=self.user,
            source_file=ContentFile(b"fake mp3", name="source.mp3"),
            original_filename="source.mp3",
            content_type="audio/mpeg",
            size_bytes=8,
        )

        processed = process_next_pending_import()

        self.assertEqual(processed.id, import_job.id)
        decoder.assert_called_once()
        import_job.refresh_from_db()
        self.assertEqual(import_job.status, MeetingImportStatus.COMPLETE)
        self.assertGreaterEqual(import_job.created_segments, 1)
        self.assertEqual(self.meeting.segments.count(), import_job.created_segments)

    @override_settings(QUEUE_STALE_AFTER_SECONDS=1)
    def test_requeues_stale_processing_import(self):
        import_job = MeetingImport.objects.create(
            meeting=self.meeting,
            user=self.user,
            source_file=ContentFile(b"fake mp3", name="source.mp3"),
            original_filename="source.mp3",
            content_type="audio/mpeg",
            size_bytes=8,
            status=MeetingImportStatus.PROCESSING,
            started_at=timezone.now() - timedelta(minutes=5),
        )

        claimed = claim_next_pending_import()

        self.assertEqual(claimed.id, import_job.id)
        claimed.refresh_from_db()
        self.assertEqual(claimed.status, MeetingImportStatus.PROCESSING)
        self.assertIsNotNone(claimed.started_at)

    @patch("meetings.import_processing.decode_with_ffmpeg")
    def test_interrupted_import_segments_are_replaced_on_retry(self, decoder):
        decoder.return_value = (voiced_samples(), 16000)
        import_job = MeetingImport.objects.create(
            meeting=self.meeting,
            user=self.user,
            source_file=ContentFile(b"fake mp3", name="source.mp3"),
            original_filename="source.mp3",
            content_type="audio/mpeg",
            size_bytes=8,
        )
        self.meeting.minutes_text = "Partial notes"
        self.meeting.output_status = MeetingOutputStatus.COMPLETE
        self.meeting.save(update_fields=["minutes_text", "output_status", "updated_at"])
        partial_segment = AudioSegment.objects.create(
            meeting=self.meeting,
            user=self.user,
            client_segment_id=f"import_{import_job.id}_000001",
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=1000,
            audio_file=ContentFile(wav_bytes(), name="seg_1.wav"),
            audio_size_bytes=len(wav_bytes()),
            audio_content_type="audio/wav",
        )
        MeetingMessage.objects.create(
            meeting=self.meeting,
            user=self.user,
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=1000,
            transcript_text="Partial transcript.",
        )
        partial_audio_name = partial_segment.audio_file.name

        processed = process_next_pending_import()

        self.assertEqual(processed.id, import_job.id)
        decoder.assert_called_once()
        import_job.refresh_from_db()
        self.meeting.refresh_from_db()
        self.assertEqual(import_job.status, MeetingImportStatus.COMPLETE)
        self.assertEqual(import_job.created_segments, 1)
        self.assertEqual(self.meeting.output_status, MeetingOutputStatus.PENDING)
        self.assertEqual(self.meeting.minutes_text, "")
        self.assertEqual(self.meeting.messages.count(), 0)
        self.assertFalse(default_storage.exists(partial_audio_name))
        segment = self.meeting.segments.get()
        self.assertNotEqual(segment.id, partial_segment.id)
        self.assertTrue(segment.client_segment_id.startswith(f"import_{import_job.id}_"))


class MeetingMcpApiTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=self.temp_dir.name,
            MCP_DEFAULT_USERNAME="mcp-user",
            MCP_PUBLIC_URL="https://example.test/mcp",
        )
        self.settings_override.enable()
        self.user = User.objects.create_user(username="mcp-user")
        self.other_user = User.objects.create_user(username="other-mcp-user")

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    def make_meeting(self, user=None, title="MCP meeting") -> Meeting:
        meeting = Meeting.objects.create(
            user=user or self.user,
            title=title,
            status=MeetingStatus.COMPLETE,
        )
        AudioSegment.objects.create(
            meeting=meeting,
            user=user or self.user,
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=1000,
            transcription_status=SegmentStatus.COMPLETE,
            transcription_text="Hello from the MCP test.",
            audio_file=ContentFile(wav_bytes(), name="mcp-segment.wav"),
            audio_size_bytes=len(wav_bytes()),
            audio_content_type="audio/wav",
        )
        return meeting

    def test_mcp_list_meetings_uses_configured_user(self):
        own_meeting = self.make_meeting(title="Own MCP meeting")
        self.make_meeting(user=self.other_user, title="Other MCP meeting")

        payload = mcp_api.list_meetings()

        self.assertEqual(payload["user"], "mcp-user")
        self.assertEqual(len(payload["meetings"]), 1)
        self.assertEqual(payload["meetings"][0]["id"], str(own_meeting.id))
        self.assertEqual(payload["meetings"][0]["title"], "Own MCP meeting")

    def test_mcp_get_meeting_returns_segments_and_absolute_audio_urls(self):
        meeting = self.make_meeting()

        payload = mcp_api.get_meeting(str(meeting.id))

        self.assertEqual(payload["id"], str(meeting.id))
        self.assertEqual(payload["segments"][0]["transcription_text"], "Hello from the MCP test.")
        self.assertTrue(payload["segments"][0]["audio_url"].startswith("https://example.test/media/"))

    def test_mcp_exposes_user_and_meeting_types(self):
        user_payload = mcp_api.get_current_user()
        type_payload = mcp_api.list_meeting_types()

        self.assertEqual(user_payload["username"], "mcp-user")
        self.assertIn(
            {
                "value": MeetingType.PROJECT_MANAGER_NOTES,
                "label": "Project manager notes",
                "supports_pdf": True,
            },
            type_payload["meeting_types"],
        )

    def test_mcp_start_end_and_progress_meeting(self):
        started = mcp_api.start_meeting(title="Agent-created meeting")

        self.assertEqual(started["title"], "Agent-created meeting")
        self.assertEqual(started["status"], MeetingStatus.RECORDING)

        progress = mcp_api.get_meeting_progress(started["id"])
        self.assertEqual(progress["percent"], 100)
        self.assertEqual(progress["message"], "Complete")

        ended = mcp_api.end_meeting(started["id"], rebuild_outputs=False)
        self.assertEqual(ended["status"], MeetingStatus.COMPLETE)
        self.assertIsNotNone(ended["ended_at"])

    def test_mcp_upload_audio_segment_from_base64(self):
        meeting = Meeting.objects.create(user=self.user, title="Recording", status=MeetingStatus.RECORDING)

        segment = mcp_api.upload_audio_segment_from_base64(
            meeting_id=str(meeting.id),
            filename="segment.wav",
            content_base64=base64.b64encode(wav_bytes()).decode("ascii"),
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=1000,
            content_type="audio/wav",
        )

        self.assertEqual(segment["sequence_number"], 1)
        self.assertEqual(segment["transcription_status"], SegmentStatus.PENDING)
        self.assertTrue(segment["audio_url"].startswith("https://example.test/media/"))

    def test_mcp_import_recording_from_base64(self):
        payload = mcp_api.import_recording_from_base64(
            filename="old-meeting.mp3",
            content_base64=base64.b64encode(wav_bytes()).decode("ascii"),
            title="Old meeting",
            content_type="audio/mpeg",
        )

        self.assertEqual(payload["meeting"]["title"], "Old meeting")
        self.assertEqual(payload["import"]["status"], MeetingImportStatus.PENDING)
        self.assertEqual(payload["import"]["original_filename"], "old-meeting.mp3")

    def test_mcp_project_manager_pdf_returns_base64_pdf(self):
        meeting = self.make_meeting(title="PM notes")
        meeting.meeting_type = MeetingType.PROJECT_MANAGER_NOTES
        meeting.minutes_text = (
            "Meeting Details:\n\nDate: 2026-05-19\nTime: 09:52 UTC\n"
            "Location/Platform: Not specified\n\nAttendees:\n\nNot specified\n\n"
            "Discussion Points:\n\nTopic\n\n- Keep all details."
        )
        meeting.save(update_fields=["meeting_type", "minutes_text"])

        payload = mcp_api.get_project_manager_notes_pdf(str(meeting.id))
        pdf_bytes = base64.b64decode(payload["content_base64"])

        self.assertEqual(payload["content_type"], "application/pdf")
        self.assertTrue(payload["filename"].endswith("-pm-notes.pdf"))
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


class MeetingMinutesTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=self.temp_dir.name,
            IMPORT_CHUNK_BYTES=1024,
        )
        self.settings_override.enable()
        self.user = User.objects.create_user(
            username="web-user",
            password="strong-password-123",
        )
        self.other_user = User.objects.create_user(username="other-web-user")
        self.client = Client()

    def tearDown(self):
        self.settings_override.disable()
        self.temp_dir.cleanup()

    def make_meeting(self, user=None, title="Discovery call") -> Meeting:
        meeting = Meeting.objects.create(
            user=user or self.user,
            title=title,
            status=MeetingStatus.COMPLETE,
            meeting_type=MeetingType.REQUIREMENT_GATHERING,
        )
        AudioSegment.objects.create(
            meeting=meeting,
            user=user or self.user,
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=5000,
            transcription_status=SegmentStatus.COMPLETE,
            transcription_text="We need a customer portal with approvals.",
            audio_file=ContentFile(wav_bytes(), name="seg_1.wav"),
            audio_size_bytes=len(wav_bytes()),
            audio_content_type="audio/wav",
        )
        return meeting

    def test_web_meetings_requires_login(self):
        response = self.client.get("/meetings/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_web_meetings_show_only_current_users_meetings(self):
        own_meeting = self.make_meeting(title="Own meeting")
        self.make_meeting(user=self.other_user, title="Other meeting")
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.get("/meetings/")

        self.assertContains(response, own_meeting.title)
        self.assertNotContains(response, "Other meeting")
        self.assertContains(response, "Import a previous recording")

    def test_web_import_upload_redirects_to_detail_and_queues_job(self):
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.post(
            "/meetings/import/",
            {
                "title": "Old call",
                "recording_file": ContentFile(voiced_wav_bytes(), name="old-call.wav"),
            },
        )

        meeting = Meeting.objects.get(title="Old call")
        import_job = MeetingImport.objects.get(meeting=meeting)
        self.assertEqual(response.status_code, 302)
        self.assertIn(str(meeting.id), response["Location"])
        self.assertEqual(meeting.user, self.user)
        self.assertEqual(import_job.status, MeetingImportStatus.PENDING)

    def test_web_import_accepts_m4a_upload(self):
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.post(
            "/meetings/import/",
            {
                "title": "Old m4a call",
                "recording_file": ContentFile(b"fake m4a", name="old-call.m4a"),
            },
        )

        meeting = Meeting.objects.get(title="Old m4a call")
        import_job = MeetingImport.objects.get(meeting=meeting)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(import_job.original_filename, "old-call.m4a")
        self.assertEqual(import_job.status, MeetingImportStatus.PENDING)

    def test_meeting_progress_endpoint_reports_import_step(self):
        meeting = Meeting.objects.create(
            user=self.user,
            title="Long import",
            status=MeetingStatus.ENDED,
        )
        MeetingImport.objects.create(
            meeting=meeting,
            user=self.user,
            source_file=ContentFile(b"fake mp4", name="source.mp4"),
            original_filename="source.mp4",
            content_type="video/mp4",
            size_bytes=8,
            status=MeetingImportStatus.PROCESSING,
            progress_percent=40,
            progress_message="Detecting speech ranges",
        )
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.get(f"/meetings/{meeting.id}/progress/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["percent"], 40)
        self.assertEqual(payload["message"], "Detecting speech ranges")
        self.assertTrue(payload["should_poll"])

    def test_chunked_web_import_assembles_file_and_queues_job(self):
        self.client.login(username=self.user.username, password="strong-password-123")
        payload = voiced_wav_bytes()
        chunk_size = 1024
        total_chunks = math.ceil(len(payload) / chunk_size)

        start_response = self.client.post(
            "/meetings/import/chunked/start/",
            data=json.dumps(
                {
                    "title": "Chunked old call",
                    "filename": "chunked-old-call.mp4",
                    "content_type": "video/mp4",
                    "total_size": len(payload),
                    "chunk_size": chunk_size,
                    "total_chunks": total_chunks,
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(start_response.status_code, 201)
        upload_id = start_response.json()["upload_id"]

        for index in range(total_chunks):
            start = index * chunk_size
            end = min(start + chunk_size, len(payload))
            chunk_response = self.client.post(
                f"/meetings/import/chunked/{upload_id}/chunk/",
                {
                    "index": str(index),
                    "chunk": ContentFile(payload[start:end], name=f"{index}.part"),
                },
            )
            self.assertEqual(chunk_response.status_code, 200)

        finish_response = self.client.post(f"/meetings/import/chunked/{upload_id}/finish/")

        self.assertEqual(finish_response.status_code, 201)
        meeting = Meeting.objects.get(title="Chunked old call")
        import_job = MeetingImport.objects.get(meeting=meeting)
        self.assertEqual(meeting.user, self.user)
        self.assertEqual(import_job.status, MeetingImportStatus.PENDING)
        self.assertEqual(import_job.size_bytes, len(payload))
        self.assertEqual(import_job.original_filename, "chunked-old-call.mp4")

    def test_generate_minutes_saves_type_and_calls_extractor(self):
        meeting = self.make_meeting()
        self.client.login(username=self.user.username, password="strong-password-123")

        with patch("meetings.web_views.generate_minutes_for_meeting") as extractor:
            response = self.client.post(
                f"/meetings/{meeting.id}/minutes/",
                {"meeting_type": MeetingType.FOLLOWUP_MEETING},
            )

        self.assertEqual(response.status_code, 302)
        meeting.refresh_from_db()
        self.assertEqual(meeting.meeting_type, MeetingType.FOLLOWUP_MEETING)
        extractor.assert_called_once()

    def test_generate_minutes_for_meeting_stores_openai_output(self):
        meeting = self.make_meeting()
        fake_client = FakeMinutesClient()

        generate_minutes_for_meeting(meeting, client=fake_client)

        meeting.refresh_from_db()
        self.assertEqual(meeting.minutes_text, "## Summary\n- Clear next step.")
        self.assertEqual(meeting.minutes_model, "fake-minutes")
        self.assertEqual(meeting.minutes_response["id"], "fake")
        self.assertEqual(meeting.minutes_last_error, "")
        self.assertIsNotNone(meeting.minutes_generated_at)
        self.assertIn("customer portal", fake_client.transcripts[0])

    def test_requirement_gathering_prompt_outputs_only_final_requirements(self):
        meeting = self.make_meeting()
        meeting.meeting_type = MeetingType.REQUIREMENT_GATHERING

        prompt = build_minutes_prompt(
            meeting,
            "person_1: Add chat. person_2: Remove chat and use tickets instead.",
        )

        self.assertIn("Output only the final gathered requirements", prompt)
        self.assertIn("If something was discussed and later removed, omit it completely", prompt)
        self.assertIn("keep only the more recent requirement", prompt)
        self.assertIn("Do not include headings, sections", prompt)
        self.assertNotIn("Action Items", prompt)

    def test_requirement_gathering_minutes_keeps_minutes_sections(self):
        meeting = self.make_meeting()
        meeting.meeting_type = MeetingType.REQUIREMENT_GATHERING_MINUTES

        prompt = build_minutes_prompt(meeting, "person_1: We need approvals.")

        self.assertIn("Summary, Key Discussion Points, Decisions", prompt)
        self.assertIn("Focus on business goals, user needs", prompt)
        self.assertIn("may contain transcription mistakes", prompt)

    def test_project_manager_notes_prompt_matches_requested_format(self):
        meeting = self.make_meeting(title="ScoutX planning")
        meeting.meeting_type = MeetingType.PROJECT_MANAGER_NOTES

        prompt = build_minutes_prompt(
            meeting,
            "person_1: Add a filter by team. person_2: Remove the save option from videos.",
        )

        self.assertIn("Use this exact structure:", prompt)
        self.assertIn("Meeting Details:", prompt)
        self.assertIn("Date: 2026-05-19", prompt)
        self.assertIn("Time:", prompt)
        self.assertIn("Attendees:", prompt)
        self.assertIn("Discussion Points:", prompt)
        self.assertIn("compact professional project manager notes", prompt)
        self.assertIn("Use plain text with hyphen bullets", prompt)
        self.assertIn("This is not a transcript recap", prompt)
        self.assertIn("must not omit information", prompt)
        self.assertIn("Target 800-1,300 words", prompt)
        self.assertIn("Completeness means no unique product", prompt)
        self.assertIn("Keep the notes around the same length and density", prompt)
        self.assertIn("Omit unclear transcript fragments", prompt)
        self.assertIn("Reference style and length", prompt)
        self.assertIn("Team Player Assignment Flow", prompt)
        self.assertIn("Before finalizing, review the transcript again", prompt)
        self.assertIn("If something is discussed and later removed, omit it completely", prompt)

    def test_project_manager_notes_use_chunked_extraction_helpers(self):
        meeting = self.make_meeting(title="Long PM meeting")
        transcript = "\n".join(f"[00:{index:02d}] person_1: Detail {index}" for index in range(20))

        chunks = chunk_transcript(transcript, max_chars=120)
        final_prompt = build_project_manager_final_prompt(meeting, "Chunk 1 notes:\n- Add chat buttons.")

        self.assertGreater(len(chunks), 1)
        self.assertIn("This is not a transcript recap", final_prompt)
        self.assertIn("must not omit information", final_prompt)
        self.assertIn("Target 800-1,300 words", final_prompt)
        self.assertIn("Completeness means no unique product", final_prompt)
        self.assertIn("Keep the notes around the same length and density", final_prompt)
        self.assertIn("Reference style and length", final_prompt)

    def test_project_manager_compaction_prompt_preserves_information_with_word_cap(self):
        meeting = self.make_meeting(title="Long PM meeting")

        prompt = build_project_manager_compaction_prompt(
            meeting,
            "Meeting Details:\n\nDate: 2026-05-19\n\nDiscussion Points:\n\n- Add filters.",
        )

        self.assertIn("Hard maximum: 1,400 words", prompt)
        self.assertIn("Do not omit any unique requirement", prompt)
        self.assertIn("Merge sibling bullets aggressively", prompt)
        self.assertIn("Keep the exact same top-level structure", prompt)

    def test_gpt_55_minutes_options_omit_temperature(self):
        self.assertEqual(chat_completion_options("gpt-5.5", temperature=0.2), {"model": "gpt-5.5"})
        self.assertEqual(
            chat_completion_options("gpt-4o-mini", temperature=0.2),
            {"model": "gpt-4o-mini", "temperature": 0.2},
        )

    def test_detail_page_lists_processed_messages_and_audio(self):
        meeting = self.make_meeting(title="Processed meeting")
        meeting.minutes_text = "- The system must support approvals."
        meeting.minutes_model = "fake-minutes"
        meeting.save(update_fields=["minutes_text", "minutes_model"])
        message = MeetingMessage.objects.create(
            meeting=meeting,
            user=self.user,
            sequence_number=1,
            speaker_label="person_1",
            client_start_ms=0,
            client_end_ms=5000,
            transcript_text="person_1: We need a customer portal with approvals.",
            detailed_summary="The customer portal needs approvals.",
            short_summary="Customer portal needs approvals",
        )
        message.segments.set(meeting.segments.all())
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.get(f"/meetings/{meeting.id}/")

        self.assertContains(response, "Processed meeting")
        self.assertContains(response, "Processing progress")
        self.assertContains(response, "Extract meeting minutes")
        self.assertContains(response, "Paste in proposal generator")
        self.assertContains(response, "proposal-engine-vm8.bit68-infra.com/requirements-to-pdf/form/")
        self.assertContains(response, 'window.open(destination, "_blank"', html=False)
        self.assertNotContains(response, "window.location.href = destination")
        self.assertContains(response, "Customer portal needs approvals")
        self.assertContains(response, "<audio", html=False)

    def test_project_manager_notes_pdf_download(self):
        meeting = self.make_meeting(title="PM Notes")
        meeting.meeting_type = MeetingType.PROJECT_MANAGER_NOTES
        meeting.minutes_text = "Meeting Details:\n\nDate: 2026-05-19\nTime: 10:00\nLocation/Platform: Google Meet\nAttendees:\n\nLujain\n\nDiscussion Points:\n\nAdmin Profile\n- Add an Admin Profile flow from the sidebar."
        meeting.save(update_fields=["meeting_type", "minutes_text"])
        self.client.login(username=self.user.username, password="strong-password-123")

        detail_response = self.client.get(f"/meetings/{meeting.id}/")
        pdf_response = self.client.get(f"/meetings/{meeting.id}/minutes/pdf/")

        self.assertContains(detail_response, "Download PM notes PDF")
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response["Content-Type"], "application/pdf")
        self.assertTrue(bytes(pdf_response.content).startswith(b"%PDF"))
        self.assertIn("attachment;", pdf_response["Content-Disposition"])

    def test_proposal_generator_button_only_for_gathered_requirements(self):
        meeting = self.make_meeting(title="Followup")
        meeting.meeting_type = MeetingType.FOLLOWUP_MEETING
        meeting.minutes_text = "## Summary\n- Followup notes."
        meeting.save(update_fields=["meeting_type", "minutes_text"])
        self.client.login(username=self.user.username, password="strong-password-123")

        response = self.client.get(f"/meetings/{meeting.id}/")

        self.assertNotContains(response, "Paste in proposal generator")
        self.assertNotContains(response, "proposal-engine-vm8.bit68-infra.com/requirements-to-pdf/form/")

    def test_process_meeting_outputs_creates_messages_summaries_and_title(self):
        meeting = self.make_meeting(title="")
        AudioSegment.objects.create(
            meeting=meeting,
            user=self.user,
            sequence_number=2,
            speaker_label="person_1",
            client_start_ms=5000,
            client_end_ms=9000,
            transcription_status=SegmentStatus.COMPLETE,
            transcription_text="It should also export reports.",
            audio_file=ContentFile(wav_bytes(), name="seg_2.wav"),
            audio_size_bytes=len(wav_bytes()),
            audio_content_type="audio/wav",
        )
        fake_client = FakeMeetingOutputClient()

        process_meeting_outputs(meeting, client=fake_client, force=True)

        meeting.refresh_from_db()
        self.assertEqual(meeting.output_status, MeetingOutputStatus.COMPLETE)
        self.assertEqual(meeting.title, "Customer Portal Planning")
        self.assertEqual(meeting.messages.count(), 1)
        message = meeting.messages.get()
        self.assertEqual(list(message.segments.order_by("sequence_number").values_list("sequence_number", flat=True)), [1, 2])
        self.assertIn("export reports", message.transcript_text)
        self.assertEqual(message.detailed_summary, "Detailed summary without missing details.")
        self.assertEqual(message.short_summary, "Short summary")


class FakeTranscriptionClient:
    def __init__(self):
        self.sequences = []

    def transcribe(self, segment: AudioSegment) -> TranscriptionResult:
        self.sequences.append(segment.sequence_number)
        return TranscriptionResult(
            text=f"Transcript {segment.sequence_number}",
            model="fake-transcribe",
            raw_response={"text": f"Transcript {segment.sequence_number}"},
        )


class FakeMinutesClient:
    def __init__(self):
        self.transcripts = []

    def generate(self, meeting: Meeting) -> MinutesResult:
        transcript = "\n".join(
            segment.transcription_text
            for segment in meeting.segments.order_by("sequence_number")
        )
        self.transcripts.append(transcript)
        return MinutesResult(
            text="## Summary\n- Clear next step.",
            model="fake-minutes",
            raw_response={"id": "fake"},
        )


class FakeMeetingOutputClient:
    model = "fake-output"

    def compile_messages(self, meeting: Meeting, segments: list[AudioSegment]):
        draft = MessageDraft(
            sequence_numbers=[segment.sequence_number for segment in segments],
            speaker_label="person_1",
            transcript_text="\n".join(segment.transcription_text for segment in segments),
            client_start_ms=min(segment.client_start_ms for segment in segments),
            client_end_ms=max(segment.client_end_ms for segment in segments),
        )
        return [draft], {"id": "compile"}

    def summarize_message(self, draft: MessageDraft) -> TextResult:
        return TextResult(
            text="Detailed summary without missing details.",
            raw_response={"id": "detailed"},
        )

    def summarize_message_short(self, draft: MessageDraft) -> TextResult:
        return TextResult(text="Short summary", raw_response={"id": "short"})

    def generate_title(self, meeting: Meeting, drafts: list[MessageDraft]) -> TextResult:
        return TextResult(
            text="Customer Portal Planning",
            raw_response={"id": "title"},
        )
