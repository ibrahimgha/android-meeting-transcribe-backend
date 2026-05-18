from django.urls import path

from .web_views import (
    GenerateMeetingMinutesView,
    GenerateMeetingOutputsView,
    MeetingDetailView,
    MeetingListView,
)

urlpatterns = [
    path("", MeetingListView.as_view(), name="web-meetings"),
    path("<uuid:pk>/", MeetingDetailView.as_view(), name="web-meeting-detail"),
    path("<uuid:pk>/minutes/", GenerateMeetingMinutesView.as_view(), name="web-generate-minutes"),
    path("<uuid:pk>/outputs/", GenerateMeetingOutputsView.as_view(), name="web-generate-outputs"),
]
