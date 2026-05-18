from django.contrib import admin

from .models import AudioSegment, Meeting, MeetingMessage


class AudioSegmentInline(admin.TabularInline):
    model = AudioSegment
    extra = 0
    readonly_fields = [
        "id",
        "sequence_number",
        "speaker_label",
        "transcription_status",
        "created_at",
    ]


class MeetingMessageInline(admin.TabularInline):
    model = MeetingMessage
    extra = 0
    readonly_fields = [
        "id",
        "sequence_number",
        "speaker_label",
        "short_summary",
        "created_at",
    ]
    fields = [
        "id",
        "sequence_number",
        "speaker_label",
        "short_summary",
        "created_at",
    ]


@admin.register(Meeting)
class MeetingAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "user",
        "title",
        "status",
        "meeting_type",
        "output_status",
        "minutes_generated_at",
        "started_at",
        "ended_at",
    ]
    list_filter = ["status", "meeting_type", "output_status", "started_at"]
    search_fields = ["id", "title", "user__username", "user__email"]
    readonly_fields = [
        "minutes_generated_at",
        "minutes_model",
        "minutes_last_error",
        "output_generated_at",
        "output_model",
        "output_last_error",
        "title_generated_at",
        "title_model",
    ]
    inlines = [MeetingMessageInline, AudioSegmentInline]


@admin.register(AudioSegment)
class AudioSegmentAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "meeting",
        "sequence_number",
        "speaker_label",
        "transcription_status",
        "transcription_attempts",
        "created_at",
    ]
    list_filter = ["transcription_status", "speaker_label", "created_at"]
    search_fields = ["id", "meeting__id", "speaker_label", "transcription_text"]


@admin.register(MeetingMessage)
class MeetingMessageAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "meeting",
        "sequence_number",
        "speaker_label",
        "short_summary",
        "created_at",
    ]
    list_filter = ["speaker_label", "created_at"]
    search_fields = [
        "id",
        "meeting__id",
        "speaker_label",
        "transcript_text",
        "detailed_summary",
        "short_summary",
    ]
