from pathlib import Path

from django.conf import settings
from django import forms

from .import_formats import SUPPORTED_IMPORT_AUDIO_EXTENSIONS, supported_import_audio_message
from .models import Meeting, MeetingType


class MeetingMinutesForm(forms.ModelForm):
    meeting_type = forms.ChoiceField(
        choices=[
            choice
            for choice in MeetingType.choices
            if choice[0] != MeetingType.COMPACT_PM_NOTES
        ],
        label="Meeting type",
        widget=forms.Select(attrs={"class": "meeting-type-select"}),
    )

    class Meta:
        model = Meeting
        fields = ["meeting_type"]


class MeetingImportForm(forms.Form):
    title = forms.CharField(max_length=160, required=False, label="Meeting title")
    recording_file = forms.FileField(
        label="Full meeting recording",
        widget=forms.FileInput(attrs={"accept": ".wav,.mp3,.m4a,.mp4,audio/*,video/mp4"}),
    )

    def clean_recording_file(self):
        recording_file = self.cleaned_data["recording_file"]
        if recording_file.size > settings.MAX_IMPORT_RECORDING_BYTES:
            raise forms.ValidationError("Recording is larger than the configured import limit.")

        extension = Path(recording_file.name).suffix.lower().lstrip(".")
        if extension not in SUPPORTED_IMPORT_AUDIO_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported recording format. Use one of: {supported_import_audio_message()}."
            )
        return recording_file
