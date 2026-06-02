from django.contrib import admin

from .models import AudioSegment, Meeting, MeetingImport, MeetingMessage, MeetingMinutesOutput, UserWebSettings


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


class MeetingImportInline(admin.TabularInline):
    model = MeetingImport
    extra = 0
    readonly_fields = [
        "id",
        "original_filename",
        "status",
        "created_segments",
        "last_error",
        "created_at",
    ]
    fields = [
        "id",
        "original_filename",
        "status",
        "created_segments",
        "last_error",
        "created_at",
    ]


class MeetingMinutesOutputInline(admin.TabularInline):
    model = MeetingMinutesOutput
    extra = 0
    readonly_fields = [
        "id",
        "meeting_type",
        "status",
        "generated_at",
        "model",
        "last_error",
        "created_at",
    ]
    fields = [
        "id",
        "meeting_type",
        "status",
        "generated_at",
        "model",
        "last_error",
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
        "minutes_status",
        "output_status",
        "minutes_generated_at",
        "started_at",
        "ended_at",
    ]
    list_filter = ["status", "meeting_type", "minutes_status", "output_status", "started_at"]
    search_fields = ["id", "title", "user__username", "user__email"]
    readonly_fields = [
        "minutes_status",
        "minutes_requested_at",
        "minutes_started_at",
        "minutes_generated_at",
        "minutes_model",
        "minutes_last_error",
        "output_generated_at",
        "output_model",
        "output_last_error",
        "title_generated_at",
        "title_model",
    ]
    inlines = [MeetingImportInline, MeetingMinutesOutputInline, MeetingMessageInline, AudioSegmentInline]


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


@admin.register(MeetingImport)
class MeetingImportAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "meeting",
        "user",
        "original_filename",
        "status",
        "created_segments",
        "created_at",
    ]
    list_filter = ["status", "created_at"]
    search_fields = ["id", "meeting__id", "original_filename", "user__username"]


@admin.register(MeetingMinutesOutput)
class MeetingMinutesOutputAdmin(admin.ModelAdmin):
    list_display = ["id", "meeting", "meeting_type", "status", "model", "generated_at", "updated_at"]
    list_filter = ["meeting_type", "status", "generated_at"]
    search_fields = ["id", "meeting__id", "meeting__title", "text", "last_error"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(UserWebSettings)
class UserWebSettingsAdmin(admin.ModelAdmin):
    list_display = ["user", "force_password_change", "can_view_all_meetings", "password_changed_at", "updated_at"]
    list_filter = ["force_password_change", "can_view_all_meetings", "password_changed_at"]
    search_fields = ["user__username", "user__email"]
