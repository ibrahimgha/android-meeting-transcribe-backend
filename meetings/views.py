from django.db import IntegrityError
from django.db.models import Count
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Meeting, MeetingStatus
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
)


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
