from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect
from django.views import View
from django.views.generic import ListView

from .forms import MeetingMinutesForm
from .minutes import generate_minutes_for_meeting
from .models import Meeting


class MeetingListView(LoginRequiredMixin, ListView):
    template_name = "meetings/meeting_list.html"
    context_object_name = "meetings"

    def get_queryset(self):
        return (
            Meeting.objects.filter(user=self.request.user)
            .annotate(
                segment_count=Count("segments"),
                completed_transcription_count=Count(
                    "segments",
                    filter=Q(segments__transcription_text__gt=""),
                ),
            )
            .prefetch_related("segments")
            .order_by("-started_at")
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        for meeting in context["meetings"]:
            meeting.minutes_form = MeetingMinutesForm(instance=meeting)
        return context


class GenerateMeetingMinutesView(LoginRequiredMixin, View):
    def post(self, request, pk):
        meeting = get_object_or_404(Meeting, pk=pk, user=request.user)
        form = MeetingMinutesForm(request.POST, instance=meeting)
        if not form.is_valid():
            messages.error(request, "Choose a meeting type before extracting minutes.")
            return redirect("web-meetings")

        form.save()
        try:
            generate_minutes_for_meeting(meeting)
        except Exception as exc:
            messages.error(request, f"Could not extract minutes: {exc}")
        else:
            messages.success(request, "Meeting minutes extracted.")

        return redirect("web-meetings")
