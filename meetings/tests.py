from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from .models import AudioSegment, Meeting, MeetingStatus, SegmentStatus
from .transcription import TranscriptionResult, process_next_pending_segment

User = get_user_model()


def wav_bytes() -> bytes:
    return b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


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
