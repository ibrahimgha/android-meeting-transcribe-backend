from django.db import migrations


LEGACY_VALUE = "\x6c\x75\x6a\x79_pm_notes"
COMPACT_VALUE = "compact_pm_notes"


def rename_forward(apps, schema_editor):
    Meeting = apps.get_model("meetings", "Meeting")
    MeetingMinutesOutput = apps.get_model("meetings", "MeetingMinutesOutput")
    Meeting.objects.filter(meeting_type=LEGACY_VALUE).update(meeting_type=COMPACT_VALUE)
    MeetingMinutesOutput.objects.filter(meeting_type=LEGACY_VALUE).update(meeting_type=COMPACT_VALUE)


def rename_backward(apps, schema_editor):
    Meeting = apps.get_model("meetings", "Meeting")
    MeetingMinutesOutput = apps.get_model("meetings", "MeetingMinutesOutput")
    Meeting.objects.filter(meeting_type=COMPACT_VALUE).update(meeting_type=LEGACY_VALUE)
    MeetingMinutesOutput.objects.filter(meeting_type=COMPACT_VALUE).update(meeting_type=LEGACY_VALUE)


class Migration(migrations.Migration):
    dependencies = [
        ("meetings", "0011_meeting_health_dashboard"),
    ]

    operations = [
        migrations.RunPython(rename_forward, rename_backward),
    ]
