from django.core.management.base import BaseCommand

from meetings.minutes import (
    OpenAIMinutesClient,
    generate_minutes_for_meeting,
    queue_health_report_for_meeting,
)
from meetings.models import Meeting, MeetingMinutesOutput, MeetingMinutesStatus, MeetingStatus, MeetingType


class Command(BaseCommand):
    help = "Queue meeting health reports for completed meetings."

    def add_arguments(self, parser):
        parser.add_argument(
            "--process",
            action="store_true",
            help="Process queued health reports immediately after queueing them.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate existing completed health reports.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of health reports to process when --process is used.",
        )

    def handle(self, *args, **options):
        force = options["force"]
        queued = 0
        skipped = 0

        meetings = Meeting.objects.filter(status=MeetingStatus.COMPLETE).order_by("started_at")
        for meeting in meetings.iterator():
            output = queue_health_report_for_meeting(meeting, force=force)
            if output is None:
                skipped += 1
            elif output.status == MeetingMinutesStatus.PENDING:
                queued += 1
            else:
                skipped += 1

        processed = 0
        failed = 0
        if options["process"]:
            client = OpenAIMinutesClient()
            pending_outputs = MeetingMinutesOutput.objects.select_related("meeting").filter(
                meeting__status=MeetingStatus.COMPLETE,
                meeting_type=MeetingType.MEETING_HEALTH_REPORT,
                status=MeetingMinutesStatus.PENDING,
            ).order_by("requested_at", "updated_at")
            limit = options["limit"]
            if limit is not None:
                pending_outputs = pending_outputs[:limit]

            for output in pending_outputs:
                sync_parent = output.meeting.meeting_type == output.meeting_type
                try:
                    generate_minutes_for_meeting(
                        output.meeting,
                        client=client,
                        output=output,
                        sync_parent=sync_parent,
                    )
                except Exception as exc:
                    failed += 1
                    self.stderr.write(f"Failed {output.meeting_id}: {exc}")
                else:
                    processed += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Health reports queued={queued}, skipped={skipped}, processed={processed}, failed={failed}."
            )
        )
