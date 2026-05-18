from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.db import models
from django.utils import timezone


class MeetingStatus(models.TextChoices):
    RECORDING = "recording", "Recording"
    ENDED = "ended", "Ended"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class SegmentStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class MeetingType(models.TextChoices):
    REQUIREMENT_GATHERING = "requirement_gathering", "Requirement gathering"
    FOLLOWUP_MEETING = "followup_meeting", "Followup meeting"
    DRAFT_DELIVERY = "draft_delivery", "Draft delivery"


class MeetingOutputStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


def segment_upload_path(instance: "AudioSegment", filename: str) -> str:
    extension = Path(filename).suffix.lower() or ".wav"
    return (
        f"meetings/{instance.meeting_id}/segments/"
        f"{instance.sequence_number:06d}-{uuid4().hex}{extension}"
    )


class Meeting(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meetings",
    )
    title = models.CharField(max_length=160, blank=True)
    status = models.CharField(
        max_length=16,
        choices=MeetingStatus.choices,
        default=MeetingStatus.RECORDING,
    )
    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)
    meeting_type = models.CharField(
        max_length=32,
        choices=MeetingType.choices,
        blank=True,
    )
    minutes_text = models.TextField(blank=True)
    minutes_model = models.CharField(max_length=80, blank=True)
    minutes_response = models.JSONField(default=dict, blank=True)
    minutes_generated_at = models.DateTimeField(null=True, blank=True)
    minutes_last_error = models.TextField(blank=True)
    output_status = models.CharField(
        max_length=16,
        choices=MeetingOutputStatus.choices,
        default=MeetingOutputStatus.PENDING,
        db_index=True,
    )
    output_model = models.CharField(max_length=80, blank=True)
    output_response = models.JSONField(default=dict, blank=True)
    output_generated_at = models.DateTimeField(null=True, blank=True)
    output_last_error = models.TextField(blank=True)
    title_model = models.CharField(max_length=80, blank=True)
    title_response = models.JSONField(default=dict, blank=True)
    title_generated_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return self.title or f"Meeting {self.id}"

    def refresh_completion_status(self) -> None:
        if self.status == MeetingStatus.RECORDING:
            return

        segments = self.segments.all()
        if segments.filter(
            transcription_status__in=[SegmentStatus.PENDING, SegmentStatus.PROCESSING],
        ).exists():
            if self.status != MeetingStatus.ENDED:
                self.status = MeetingStatus.ENDED
                self.save(update_fields=["status", "updated_at"])
            return

        next_status = (
            MeetingStatus.FAILED
            if segments.filter(transcription_status=SegmentStatus.FAILED).exists()
            else MeetingStatus.COMPLETE
        )
        if self.status != next_status:
            self.status = next_status
            self.save(update_fields=["status", "updated_at"])


class AudioSegment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="segments",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="audio_segments",
    )
    client_segment_id = models.CharField(max_length=80, blank=True)
    sequence_number = models.PositiveIntegerField()
    speaker_label = models.CharField(max_length=64)
    speaker_confidence = models.FloatField(null=True, blank=True)
    client_start_ms = models.PositiveBigIntegerField()
    client_end_ms = models.PositiveBigIntegerField()
    codec = models.CharField(max_length=32, blank=True)
    sample_rate = models.PositiveIntegerField(null=True, blank=True)
    audio_file = models.FileField(upload_to=segment_upload_path)
    audio_content_type = models.CharField(max_length=120, blank=True)
    audio_size_bytes = models.PositiveIntegerField(default=0)

    transcription_status = models.CharField(
        max_length=16,
        choices=SegmentStatus.choices,
        default=SegmentStatus.PENDING,
        db_index=True,
    )
    transcription_text = models.TextField(blank=True)
    transcription_model = models.CharField(max_length=80, blank=True)
    transcription_response = models.JSONField(default=dict, blank=True)
    transcription_attempts = models.PositiveIntegerField(default=0)
    transcription_started_at = models.DateTimeField(null=True, blank=True)
    transcribed_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["meeting__started_at", "meeting_id", "sequence_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["meeting", "sequence_number"],
                name="unique_segment_sequence_per_meeting",
            ),
        ]
        indexes = [
            models.Index(fields=["transcription_status", "created_at"]),
            models.Index(fields=["meeting", "sequence_number"]),
        ]

    def __str__(self) -> str:
        return f"{self.meeting_id} #{self.sequence_number} {self.speaker_label}"


class MeetingMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meeting_messages",
    )
    segments = models.ManyToManyField(
        AudioSegment,
        related_name="display_messages",
        blank=True,
    )
    sequence_number = models.PositiveIntegerField()
    speaker_label = models.CharField(max_length=64, blank=True)
    client_start_ms = models.PositiveBigIntegerField(default=0)
    client_end_ms = models.PositiveBigIntegerField(default=0)
    transcript_text = models.TextField()
    detailed_summary = models.TextField(blank=True)
    short_summary = models.CharField(max_length=180, blank=True)
    detailed_summary_response = models.JSONField(default=dict, blank=True)
    short_summary_response = models.JSONField(default=dict, blank=True)
    summary_model = models.CharField(max_length=80, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["meeting__started_at", "meeting_id", "sequence_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["meeting", "sequence_number"],
                name="unique_message_sequence_per_meeting",
            ),
        ]
        indexes = [
            models.Index(fields=["meeting", "sequence_number"]),
        ]

    def __str__(self) -> str:
        return f"{self.meeting_id} message #{self.sequence_number}"
