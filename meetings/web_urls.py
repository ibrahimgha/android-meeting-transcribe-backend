from django.urls import path

from .web_views import GenerateMeetingMinutesView, MeetingListView

urlpatterns = [
    path("", MeetingListView.as_view(), name="web-meetings"),
    path("<uuid:pk>/minutes/", GenerateMeetingMinutesView.as_view(), name="web-generate-minutes"),
]
