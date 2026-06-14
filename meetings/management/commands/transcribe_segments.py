from django.core.management.base import BaseCommand
from django.conf import settings

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
        parser.add_argument(
            "--segment-concurrency",
            type=int,
            default=getattr(settings, "TRANSCRIPTION_CONCURRENCY", 20),
            help="Maximum number of audio segment transcription jobs to run in parallel.",
        )

    def handle(self, *args, **options):
        processed = run_transcription_loop(
            once=options["once"],
            limit=options["limit"],
            sleep_seconds=options["sleep"],
            segment_concurrency=options["segment_concurrency"],
        )
        self.stdout.write(self.style.SUCCESS(f"Processed {processed} queue item(s)."))
