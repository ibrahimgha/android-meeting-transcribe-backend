from django.urls import path

from .web_views import (
    GenerateMeetingMinutesView,
    GenerateMeetingOutputsView,
    FinishChunkedImportView,
    ImportMeetingRecordingView,
    MeetingDetailView,
    MeetingListView,
    MeetingProgressView,
    StartChunkedImportView,
    UploadImportChunkView,
)

urlpatterns = [
    path("", MeetingListView.as_view(), name="web-meetings"),
    path("import/", ImportMeetingRecordingView.as_view(), name="web-import-meeting"),
    path("import/chunked/start/", StartChunkedImportView.as_view(), name="web-import-chunked-start"),
    path(
        "import/chunked/<str:upload_id>/chunk/",
        UploadImportChunkView.as_view(),
        name="web-import-chunked-chunk",
    ),
    path(
        "import/chunked/<str:upload_id>/finish/",
        FinishChunkedImportView.as_view(),
        name="web-import-chunked-finish",
    ),
    path("<uuid:pk>/", MeetingDetailView.as_view(), name="web-meeting-detail"),
    path("<uuid:pk>/progress/", MeetingProgressView.as_view(), name="web-meeting-progress"),
    path("<uuid:pk>/minutes/", GenerateMeetingMinutesView.as_view(), name="web-generate-minutes"),
    path("<uuid:pk>/outputs/", GenerateMeetingOutputsView.as_view(), name="web-generate-outputs"),
]
