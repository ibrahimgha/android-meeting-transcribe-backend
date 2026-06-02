from datetime import datetime, time, timedelta

from django.db import IntegrityError
from django.db.models import Count
from django.utils.dateparse import parse_date, parse_datetime
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Meeting, MeetingStatus, UserWebSettings
from .postprocessing import maybe_process_completed_meeting
from .serializers import (
    AuthTokenSerializer,
    LoginSerializer,
    MeetingEndSerializer,
    MeetingSerializer,
    MeetingStartSerializer,
    RegisterSerializer,
    SegmentUploadSerializer,
    UserSerializer,
    AudioSegmentSerializer,
    MeetingImportCreateSerializer,
    MeetingImportSerializer,
)


def can_view_all_meetings(user) -> bool:
    if not user.is_authenticated:
        return False
    if user.is_staff or user.is_superuser:
        return True
    return UserWebSettings.objects.filter(
        user=user,
        can_view_all_meetings=True,
    ).exists()


def parse_range_bound(value: str, *, is_end: bool):
    if "T" not in value and " " not in value:
        parsed_date = parse_date(value)
        if parsed_date is not None:
            bound = datetime.combine(parsed_date, time.min)
            bound = timezone.make_aware(bound, timezone.get_current_timezone())
            if is_end:
                bound += timedelta(days=1)
            return bound

    parsed_datetime = parse_datetime(value)
    if parsed_datetime is not None:
        if timezone.is_naive(parsed_datetime):
            parsed_datetime = timezone.make_aware(parsed_datetime, timezone.get_current_timezone())
        return parsed_datetime

    parsed_date = parse_date(value)
    if parsed_date is None:
        return None

    bound = datetime.combine(parsed_date, time.min)
    bound = timezone.make_aware(bound, timezone.get_current_timezone())
    if is_end:
        bound += timedelta(days=1)
    return bound


def meeting_duration_seconds(meeting: Meeting) -> int | None:
    if meeting.started_at is None or meeting.ended_at is None:
        return None
    return max(0, int((meeting.ended_at - meeting.started_at).total_seconds()))


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response(
            AuthTokenSerializer({"token": token.key, "user": user}).data,
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        token, _ = Token.objects.get_or_create(user=user)
        return Response(AuthTokenSerializer({"token": token.key, "user": user}).data)


class LogoutView(APIView):
    def post(self, request):
        Token.objects.filter(user=request.user).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeView(APIView):
    def get(self, request):
        return Response(UserSerializer(request.user).data)


class MeetingsByUserReportView(APIView):
    def get(self, request):
        if not can_view_all_meetings(request.user):
            return Response(
                {"detail": "You do not have permission to view system-wide meeting reports."},
                status=status.HTTP_403_FORBIDDEN,
            )

        start_value = request.query_params.get("start_date", "")
        end_value = request.query_params.get("end_date", "")
        if not start_value or not end_value:
            return Response(
                {"detail": "start_date and end_date query parameters are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        start_at = parse_range_bound(start_value, is_end=False)
        end_at = parse_range_bound(end_value, is_end=True)
        if start_at is None or end_at is None:
            return Response(
                {"detail": "Use ISO dates or datetimes for start_date and end_date."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if start_at >= end_at:
            return Response(
                {"detail": "start_date must be before end_date."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        meetings = (
            Meeting.objects.filter(started_at__gte=start_at, started_at__lt=end_at)
            .select_related("user")
            .order_by("user__username", "started_at")
        )
        users = {}
        total_duration = 0
        total_meetings = 0
        for meeting in meetings:
            user_payload = users.setdefault(
                meeting.user_id,
                {
                    "user": UserSerializer(meeting.user).data,
                    "meeting_count": 0,
                    "total_duration_seconds": 0,
                    "meetings": [],
                },
            )
            duration_seconds = meeting_duration_seconds(meeting)
            if duration_seconds is not None:
                user_payload["total_duration_seconds"] += duration_seconds
                total_duration += duration_seconds
            total_meetings += 1
            user_payload["meeting_count"] += 1
            user_payload["meetings"].append(
                {
                    "id": str(meeting.id),
                    "title": meeting.title or "Mobile meeting",
                    "status": meeting.status,
                    "started_at": meeting.started_at.isoformat() if meeting.started_at else None,
                    "ended_at": meeting.ended_at.isoformat() if meeting.ended_at else None,
                    "duration_seconds": duration_seconds,
                }
            )

        return Response(
            {
                "start_date": start_value,
                "end_date": end_value,
                "range_start": start_at.isoformat(),
                "range_end_exclusive": end_at.isoformat(),
                "total_users": len(users),
                "total_meetings": total_meetings,
                "total_duration_seconds": total_duration,
                "users": list(users.values()),
            }
        )


class MeetingViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = MeetingSerializer

    def get_queryset(self):
        return (
            Meeting.objects.filter(user=self.request.user)
            .annotate(segment_count=Count("segments"))
            .prefetch_related("segments")
        )

    @action(detail=False, methods=["post"], url_path="start")
    def start(self, request):
        serializer = MeetingStartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        meeting = Meeting.objects.create(
            user=request.user,
            title=serializer.validated_data.get("title", ""),
        )
        response = self.get_serializer(meeting)
        return Response(response.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["post"], url_path="import")
    def import_recording(self, request):
        serializer = MeetingImportCreateSerializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)
        import_job = serializer.save()
        meeting_data = self.get_serializer(import_job.meeting).data
        import_data = MeetingImportSerializer(import_job).data
        return Response(
            {"meeting": meeting_data, "import": import_data},
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["post"], url_path="end")
    def end(self, request, pk=None):
        meeting = self.get_object()
        serializer = MeetingEndSerializer(data=request.data, context={"meeting": meeting})
        serializer.is_valid(raise_exception=True)

        meeting.ended_at = serializer.validated_data.get("ended_at") or timezone.now()
        meeting.status = MeetingStatus.ENDED
        meeting.save(update_fields=["ended_at", "status", "updated_at"])
        meeting.refresh_completion_status()
        maybe_process_completed_meeting(meeting)

        return Response(self.get_serializer(meeting).data)

    @action(detail=True, methods=["post"], url_path="segments")
    def upload_segment(self, request, pk=None):
        meeting = self.get_object()
        if meeting.status not in [MeetingStatus.RECORDING, MeetingStatus.ENDED]:
            return Response(
                {"detail": "Meeting is no longer accepting audio segments."},
                status=status.HTTP_409_CONFLICT,
            )

        serializer = SegmentUploadSerializer(
            data=request.data,
            context={"request": request, "meeting": meeting},
        )
        serializer.is_valid(raise_exception=True)
        try:
            segment = serializer.save()
        except IntegrityError:
            return Response(
                {"detail": "sequence_number already exists for this meeting."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        response = AudioSegmentSerializer(segment, context={"request": request})
        return Response(response.data, status=status.HTTP_201_CREATED)
