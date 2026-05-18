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
