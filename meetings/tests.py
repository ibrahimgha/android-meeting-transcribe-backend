from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import Client
from django.test import TestCase, override_settings
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from .minutes import MinutesResult, build_minutes_prompt, generate_minutes_for_meeting
from .models import (
    AudioSegment,
    Meeting,
    MeetingMessage,
    MeetingOutputStatus,
    MeetingStatus,
    MeetingType,
    SegmentStatus,
)
from .openai_utils import chat_completion_options
from .postprocessing import MessageDraft, TextResult, process_meeting_outputs
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


class MeetingMinutesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="web-user",
            password="strong-password-123",
        )
        self.other_user = User.objects.create_user(username="other-web-user")
        self.client = Client()

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

    def test_gpt_55_minutes_options_omit_temperature(self):
        self.assertEqual(chat_completion_options("gpt-5.5", temperature=0.2), {"model": "gpt-5.5"})
        self.assertEqual(
            chat_completion_options("gpt-4o-mini", temperature=0.2),
            {"model": "gpt-4o-mini", "temperature": 0.2},
        )

    def test_detail_page_lists_processed_messages_and_audio(self):
        meeting = self.make_meeting(title="Processed meeting")
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
        self.assertContains(response, "Extract meeting minutes")
        self.assertContains(response, "Customer portal needs approvals")
        self.assertContains(response, "<audio", html=False)

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
