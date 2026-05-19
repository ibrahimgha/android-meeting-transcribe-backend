from pathlib import Path

from django.conf import settings
from django import forms

from .models import Meeting, MeetingType


class MeetingMinutesForm(forms.ModelForm):
    meeting_type = forms.ChoiceField(
        choices=MeetingType.choices,
        label="Meeting type",
        widget=forms.Select(attrs={"class": "meeting-type-select"}),
    )

    class Meta:
        model = Meeting
        fields = ["meeting_type"]


class MeetingImportForm(forms.Form):
    title = forms.CharField(max_length=160, required=False, label="Meeting title")
    recording_file = forms.FileField(label="Full meeting recording")

    def clean_recording_file(self):
        recording_file = self.cleaned_data["recording_file"]
        if recording_file.size > settings.MAX_IMPORT_RECORDING_BYTES:
            raise forms.ValidationError("Recording is larger than the configured import limit.")

        extension = Path(recording_file.name).suffix.lower().lstrip(".")
        if extension != "wav":
            raise forms.ValidationError(
                "Only WAV recordings are supported for server-side import right now."
            )
        return recording_file
