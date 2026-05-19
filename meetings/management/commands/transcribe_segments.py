from django.core.management.base import BaseCommand

from meetings.transcription import run_transcription_loop


class Command(BaseCommand):
    help = "Process pending meeting imports and audio segments sequentially."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Exit after the queue is empty instead of waiting for more work.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of segments to process before exiting.",
        )
        parser.add_argument(
            "--sleep",
            type=float,
            default=2.0,
            help="Seconds to wait before polling again when the queue is empty.",
        )

    def handle(self, *args, **options):
        processed = run_transcription_loop(
            once=options["once"],
            limit=options["limit"],
            sleep_seconds=options["sleep"],
        )
        self.stdout.write(self.style.SUCCESS(f"Processed {processed} queue item(s)."))
