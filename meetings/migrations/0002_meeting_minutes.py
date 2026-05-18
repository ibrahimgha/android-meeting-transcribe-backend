from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("meetings", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="meeting",
            name="meeting_type",
            field=models.CharField(
                blank=True,
                choices=[
                    ("requirement_gathering", "Requirement gathering"),
                    ("followup_meeting", "Followup meeting"),
                    ("draft_delivery", "Draft delivery"),
                ],
                max_length=32,
            ),
        ),
        migrations.AddField(
            model_name="meeting",
            name="minutes_text",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="meeting",
            name="minutes_model",
            field=models.CharField(blank=True, max_length=80),
        ),
        migrations.AddField(
            model_name="meeting",
            name="minutes_response",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="meeting",
            name="minutes_generated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="meeting",
            name="minutes_last_error",
            field=models.TextField(blank=True),
        ),
    ]
