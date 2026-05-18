from django.core.management.base import BaseCommand, CommandError

from meetings.models import Meeting
from meetings.postprocessing import process_meeting_outputs


class Command(BaseCommand):
    help = "Compile displayed messages, summaries, and a title for a meeting."

    def add_arguments(self, parser):
        parser.add_argument("--meeting-id", help="Meeting UUID to process.")
        parser.add_argument(
            "--latest",
            action="store_true",
            help="Process the latest meeting.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate existing output.",
        )

    def handle(self, *args, **options):
        meeting_id = options.get("meeting_id")
        latest = options.get("latest")
        if bool(meeting_id) == bool(latest):
            raise CommandError("Use exactly one of --meeting-id or --latest.")

        if latest:
            meeting = Meeting.objects.order_by("-started_at").first()
            if meeting is None:
                raise CommandError("No meetings found.")
        else:
            meeting = Meeting.objects.get(pk=meeting_id)

        process_meeting_outputs(meeting, force=options["force"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Processed meeting {meeting.id} into {meeting.messages.count()} messages."
            )
        )
