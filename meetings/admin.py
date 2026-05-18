from django.contrib import admin

from .models import AudioSegment, Meeting


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


@admin.register(Meeting)
class MeetingAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "user",
        "title",
        "status",
        "meeting_type",
        "minutes_generated_at",
        "started_at",
        "ended_at",
    ]
    list_filter = ["status", "meeting_type", "started_at"]
    search_fields = ["id", "title", "user__username", "user__email"]
    readonly_fields = ["minutes_generated_at", "minutes_model", "minutes_last_error"]
    inlines = [AudioSegmentInline]


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
