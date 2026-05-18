from pathlib import Path

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .models import AudioSegment, Meeting, MeetingStatus

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email"]


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]

    def validate_password(self, value: str) -> str:
        validate_password(value)
        return value

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        user = authenticate(
            request=self.context.get("request"),
            username=attrs["username"],
            password=attrs["password"],
        )
        if not user:
            raise serializers.ValidationError("Invalid username or password.")
        attrs["user"] = user
        return attrs


class AuthTokenSerializer(serializers.Serializer):
    token = serializers.CharField()
    user = UserSerializer()


class AudioSegmentSerializer(serializers.ModelSerializer):
    audio_url = serializers.SerializerMethodField()

    class Meta:
        model = AudioSegment
        fields = [
            "id",
            "client_segment_id",
            "sequence_number",
            "speaker_label",
            "speaker_confidence",
            "client_start_ms",
            "client_end_ms",
            "codec",
            "sample_rate",
            "audio_url",
            "audio_content_type",
            "audio_size_bytes",
            "transcription_status",
            "transcription_text",
            "transcription_model",
            "transcription_attempts",
            "last_error",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_audio_url(self, obj: AudioSegment) -> str:
        if not obj.audio_file:
            return ""
        request = self.context.get("request")
        url = obj.audio_file.url
        return request.build_absolute_uri(url) if request else url


class MeetingSerializer(serializers.ModelSerializer):
    segments = AudioSegmentSerializer(many=True, read_only=True)
    segment_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Meeting
        fields = [
            "id",
            "title",
            "status",
            "started_at",
            "ended_at",
            "segment_count",
            "segments",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "started_at",
            "ended_at",
            "segment_count",
            "segments",
            "created_at",
            "updated_at",
        ]


class MeetingStartSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=160, required=False, allow_blank=True)


class SegmentUploadSerializer(serializers.ModelSerializer):
    audio_file = serializers.FileField(write_only=True)

    class Meta:
        model = AudioSegment
        fields = [
            "id",
            "client_segment_id",
            "sequence_number",
            "speaker_label",
            "speaker_confidence",
            "client_start_ms",
            "client_end_ms",
            "codec",
            "sample_rate",
            "audio_file",
        ]
        read_only_fields = ["id"]

    def validate_audio_file(self, value):
        if value.size > settings.MAX_AUDIO_SEGMENT_BYTES:
            raise serializers.ValidationError("Audio segment is larger than the configured limit.")

        extension = Path(value.name).suffix.lower().lstrip(".")
        allowed = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
        if extension not in allowed:
            raise serializers.ValidationError(
                f"Unsupported audio format '{extension}'. Use one of: {', '.join(sorted(allowed))}."
            )
        return value

    def validate(self, attrs):
        if attrs["client_end_ms"] <= attrs["client_start_ms"]:
            raise serializers.ValidationError("client_end_ms must be greater than client_start_ms.")

        meeting = self.context["meeting"]
        if AudioSegment.objects.filter(
            meeting=meeting,
            sequence_number=attrs["sequence_number"],
        ).exists():
            raise serializers.ValidationError(
                {"sequence_number": "A segment with this sequence number already exists."}
            )
        return attrs

    def create(self, validated_data):
        request = self.context["request"]
        meeting = self.context["meeting"]
        audio_file = validated_data["audio_file"]
        return AudioSegment.objects.create(
            meeting=meeting,
            user=request.user,
            audio_content_type=getattr(audio_file, "content_type", "") or "",
            audio_size_bytes=audio_file.size,
            **validated_data,
        )


class MeetingEndSerializer(serializers.Serializer):
    ended_at = serializers.DateTimeField(required=False)

    def validate(self, attrs):
        meeting: Meeting = self.context["meeting"]
        if meeting.status != MeetingStatus.RECORDING:
            raise serializers.ValidationError("Meeting is not currently recording.")
        return attrs
