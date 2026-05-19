import io
import math
import shutil
import subprocess
import tempfile
import wave
from array import array
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from sys import byteorder

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from .models import (
    AudioSegment,
    MeetingImport,
    MeetingImportStatus,
    MeetingStatus,
)
from .import_formats import COMPRESSED_IMPORT_AUDIO_EXTENSIONS, SUPPORTED_IMPORT_AUDIO_EXTENSIONS


class MeetingImportProcessingError(ValueError):
    pass


@dataclass(frozen=True)
class ImportAudioConfig:
    sample_rate: int = 16_000
    frame_ms: int = 50
    end_silence_ms: int = 180
    min_speech_ms: int = 1_000
    max_processed_segment_ms: int = 12_000
    min_output_segment_ms: int = 5_000
    turn_merge_gap_ms: int = 8_000
    short_segment_absorb_gap_ms: int = 8_000
    max_output_turn_ms: int = 300_000
    utterance_lead_ms: int = 90
    utterance_trail_ms: int = 140

    @property
    def frame_samples(self) -> int:
        return max(1, self.sample_rate * self.frame_ms // 1000)


@dataclass(frozen=True)
class SpeechRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class SpeechSegment:
    samples: list[float]
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpeakerLabel:
    label: str
    confidence: float


@dataclass(frozen=True)
class LabeledSpeechSegment:
    segment: SpeechSegment
    label: SpeakerLabel


@dataclass
class SpeakerCluster:
    label: str
    centroid: list[float]
    count: int = 1


def claim_next_pending_import() -> MeetingImport | None:
    with transaction.atomic():
        queryset = MeetingImport.objects.select_related("meeting", "user").filter(
            status=MeetingImportStatus.PENDING,
        )
        connection_features = transaction.get_connection().features
        if connection_features.has_select_for_update:
            if connection_features.has_select_for_update_skip_locked:
                queryset = queryset.select_for_update(skip_locked=True)
            else:
                queryset = queryset.select_for_update()

        import_job = queryset.order_by("created_at").first()
        if import_job is None:
            return None

        import_job.status = MeetingImportStatus.PROCESSING
        import_job.started_at = timezone.now()
        import_job.last_error = ""
        import_job.save(
            update_fields=[
                "status",
                "started_at",
                "last_error",
                "updated_at",
            ],
        )
        return import_job


def process_next_pending_import() -> MeetingImport | None:
    import_job = claim_next_pending_import()
    if import_job is None:
        return None

    try:
        created_count, duration_ms = process_import_recording(import_job)
        if created_count <= 0:
            raise MeetingImportProcessingError("No speech segments were detected in this recording.")
    except Exception as exc:
        import_job.status = MeetingImportStatus.FAILED
        import_job.last_error = str(exc)
        import_job.processed_at = timezone.now()
        import_job.save(
            update_fields=[
                "status",
                "last_error",
                "processed_at",
                "updated_at",
            ],
        )
        if not import_job.meeting.segments.exists():
            import_job.meeting.status = MeetingStatus.FAILED
            import_job.meeting.save(update_fields=["status", "updated_at"])
    else:
        import_job.status = MeetingImportStatus.COMPLETE
        import_job.created_segments = created_count
        import_job.processed_at = timezone.now()
        import_job.last_error = ""
        import_job.save(
            update_fields=[
                "status",
                "created_segments",
                "processed_at",
                "last_error",
                "updated_at",
            ],
        )
        meeting = import_job.meeting
        meeting.status = MeetingStatus.ENDED
        meeting.ended_at = meeting.started_at + timedelta(milliseconds=duration_ms)
        meeting.save(update_fields=["status", "ended_at", "updated_at"])

    return import_job


def process_import_recording(
    import_job: MeetingImport,
    *,
    config: ImportAudioConfig | None = None,
) -> tuple[int, int]:
    config = config or ImportAudioConfig()
    samples, sample_rate = read_import_samples(import_job, config)

    if not samples:
        raise MeetingImportProcessingError("The recording is empty.")

    if sample_rate != config.sample_rate:
        samples = resample_linear(samples, sample_rate, config.sample_rate)

    duration_ms = len(samples) * 1000 // config.sample_rate
    ranges = segment_samples(samples, config)
    merged_segments = label_and_merge_segments(ranges, samples, config)
    create_audio_segments(import_job, merged_segments, config.sample_rate)
    return len(merged_segments), duration_ms


def read_import_samples(
    import_job: MeetingImport,
    config: ImportAudioConfig,
) -> tuple[list[float], int]:
    extension = Path(import_job.source_file.name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_IMPORT_AUDIO_EXTENSIONS:
        raise MeetingImportProcessingError(f"Unsupported recording format: {extension or 'unknown'}.")

    if extension in COMPRESSED_IMPORT_AUDIO_EXTENSIONS:
        return decode_with_ffmpeg(import_job.source_file.name, config)

    with default_storage.open(import_job.source_file.name, "rb") as source_file:
        return read_wav_samples(source_file)


def decode_with_ffmpeg(
    storage_name: str,
    config: ImportAudioConfig,
) -> tuple[list[float], int]:
    if shutil.which("ffmpeg") is None:
        raise MeetingImportProcessingError(
            "ffmpeg is required to import MP3, M4A, and MP4 recordings."
        )

    source_suffix = Path(storage_name).suffix.lower() or ".audio"
    with tempfile.TemporaryDirectory(prefix="meeting-import-decode-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_path = temp_dir / f"source{source_suffix}"
        decoded_path = temp_dir / "decoded.wav"

        with default_storage.open(storage_name, "rb") as source, source_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(config.sample_rate),
            "-f",
            "wav",
            str(decoded_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.IMPORT_DECODE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeetingImportProcessingError("Timed out while decoding the recording.") from exc

        if result.returncode != 0:
            error = (result.stderr or "Unknown ffmpeg error.").strip()
            raise MeetingImportProcessingError(f"Could not decode recording with ffmpeg: {error}")

        with decoded_path.open("rb") as decoded:
            return read_wav_samples(decoded)


def read_wav_samples(source_file) -> tuple[list[float], int]:
    try:
        with wave.open(source_file, "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise MeetingImportProcessingError("Only uncompressed WAV recordings are supported.")

            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except wave.Error as exc:
        raise MeetingImportProcessingError(
            "Could not read recording as a WAV file. Upload a PCM WAV recording."
        ) from exc

    if channels < 1:
        raise MeetingImportProcessingError("The WAV recording has no audio channels.")
    if sample_width not in {1, 2, 4}:
        raise MeetingImportProcessingError(
            "Only 8-bit, 16-bit, or 32-bit PCM WAV recordings are supported."
        )

    return pcm_to_mono_float(raw, channels, sample_width), sample_rate


def pcm_to_mono_float(raw: bytes, channels: int, sample_width: int) -> list[float]:
    if sample_width == 1:
        values = array("f", ((byte - 128) / 128.0 for byte in raw))
    elif sample_width == 2:
        pcm = array("h")
        pcm.frombytes(raw)
        if byteorder != "little":
            pcm.byteswap()
        values = array("f", (value / 32768.0 for value in pcm))
    else:
        pcm = array("i")
        pcm.frombytes(raw)
        if byteorder != "little":
            pcm.byteswap()
        values = array("f", (value / 2147483648.0 for value in pcm))

    if channels == 1:
        return values

    frame_count = len(values) // channels
    mono = array("f")
    for frame_index in range(frame_count):
        start = frame_index * channels
        mono.append(sum(values[start : start + channels]) / channels)
    return mono


def resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate <= 0:
        raise MeetingImportProcessingError("The WAV recording has an invalid sample rate.")
    if source_rate == target_rate or not samples:
        return samples

    output_size = max(1, round(len(samples) * target_rate / source_rate))
    output = []
    for index in range(output_size):
        source_position = index * source_rate / target_rate
        left = int(math.floor(source_position))
        right = min(left + 1, len(samples) - 1)
        fraction = source_position - left
        output.append((samples[left] * (1.0 - fraction)) + (samples[right] * fraction))
    return output


def segment_samples(samples: list[float], config: ImportAudioConfig) -> list[SpeechRange]:
    frame_values = frame_db(samples, config.frame_samples)
    speech_threshold_db = adaptive_speech_threshold(frame_values)
    lead_samples = config.sample_rate * config.utterance_lead_ms // 1000
    trail_samples = config.sample_rate * config.utterance_trail_ms // 1000
    end_silence_frames = max(1, config.end_silence_ms // config.frame_ms)
    min_segment_samples = config.sample_rate * config.min_speech_ms // 1000

    ranges = []
    in_speech = False
    segment_start = 0
    last_speech_frame = 0
    current_silence_frames = 0

    for frame_index, value_db in enumerate(frame_values):
        frame_start = frame_index * config.frame_samples
        is_speech = value_db >= speech_threshold_db
        if is_speech:
            if not in_speech:
                segment_start = max(0, frame_start - lead_samples)
                in_speech = True
            last_speech_frame = frame_index
            current_silence_frames = 0
        elif in_speech:
            current_silence_frames += 1
            if current_silence_frames >= end_silence_frames:
                speech_end = min((last_speech_frame + 1) * config.frame_samples, len(samples))
                segment_end = min(speech_end + trail_samples, len(samples))
                if segment_end - segment_start >= min_segment_samples:
                    ranges.extend(
                        split_long_range(
                            SpeechRange(segment_start, segment_end),
                            frame_values,
                            config,
                        )
                    )
                in_speech = False
                current_silence_frames = 0

    if in_speech:
        speech_end = min((last_speech_frame + 1) * config.frame_samples, len(samples))
        segment_end = min(speech_end + trail_samples, len(samples))
        if segment_end - segment_start >= min_segment_samples:
            ranges.extend(
                split_long_range(
                    SpeechRange(segment_start, segment_end),
                    frame_values,
                    config,
                )
            )

    return trim_overlaps(merge_short_ranges(ranges, min_segment_samples), min_segment_samples)


def frame_db(samples: list[float], frame_samples: int) -> list[float]:
    values = []
    offset = 0
    while offset < len(samples):
        end = min(offset + frame_samples, len(samples))
        values.append(rms_db(samples, offset, end))
        offset += frame_samples
    return values


def adaptive_speech_threshold(frame_values: list[float]) -> float:
    if not frame_values:
        return -44.0
    sorted_values = sorted(frame_values)
    low_speech = percentile(sorted_values, 0.22)
    return min(-31.0, max(-42.0, low_speech + 4.0))


def split_long_range(
    speech_range: SpeechRange,
    frame_values: list[float],
    config: ImportAudioConfig,
) -> list[SpeechRange]:
    max_samples = config.sample_rate * config.max_processed_segment_ms // 1000
    if speech_range.size <= max_samples:
        return [speech_range]

    search_start = speech_range.start + (max_samples * 45 // 100)
    search_end = min(speech_range.start + max_samples, speech_range.end - config.sample_rate)
    if search_end <= search_start:
        return [speech_range]

    best_frame = search_start // config.frame_samples
    best_db = float("inf")
    for frame in range(search_start // config.frame_samples, (search_end // config.frame_samples) + 1):
        if frame >= len(frame_values):
            continue
        if frame_values[frame] < best_db:
            best_db = frame_values[frame]
            best_frame = frame

    split = best_frame * config.frame_samples
    split = min(
        max(split, speech_range.start + config.frame_samples),
        speech_range.end - config.frame_samples,
    )
    if split <= speech_range.start or split >= speech_range.end:
        return [speech_range]

    return (
        split_long_range(SpeechRange(speech_range.start, split), frame_values, config)
        + split_long_range(SpeechRange(split, speech_range.end), frame_values, config)
    )


def merge_short_ranges(ranges: list[SpeechRange], min_segment_samples: int) -> list[SpeechRange]:
    if not ranges:
        return []

    merged = []
    for speech_range in ranges:
        previous = merged[-1] if merged else None
        previous_samples = previous.size if previous else 0
        gap_samples = speech_range.start - previous.end if previous else 2**31
        should_merge = (
            previous is not None
            and (previous_samples < min_segment_samples or speech_range.size < min_segment_samples)
            and gap_samples < min_segment_samples
        )
        if should_merge:
            merged[-1] = SpeechRange(previous.start, speech_range.end)
        else:
            merged.append(speech_range)
    return merged


def trim_overlaps(ranges: list[SpeechRange], min_segment_samples: int) -> list[SpeechRange]:
    if not ranges:
        return []

    trimmed = []
    previous_end = 0
    for speech_range in ranges:
        start = max(speech_range.start, previous_end)
        if speech_range.end - start >= min_segment_samples:
            trimmed.append(SpeechRange(start, speech_range.end))
            previous_end = speech_range.end
    return trimmed


def label_and_merge_segments(
    ranges: list[SpeechRange],
    source_samples: list[float],
    config: ImportAudioConfig,
) -> list[LabeledSpeechSegment]:
    speech_segments = [
        SpeechSegment(
            samples=source_samples[speech_range.start : speech_range.end],
            start_ms=speech_range.start * 1000 // config.sample_rate,
            end_ms=speech_range.end * 1000 // config.sample_rate,
        )
        for speech_range in ranges
    ]
    labeler = BasicSpeakerLabeler()
    labels = labeler.label_batch(speech_segments, config.sample_rate)
    labeled = [
        LabeledSpeechSegment(segment=segment, label=label)
        for segment, label in zip(speech_segments, labels)
    ]
    return merge_speaker_turns(labeled, source_samples, config)


def merge_speaker_turns(
    labeled_segments: list[LabeledSpeechSegment],
    source_samples: list[float],
    config: ImportAudioConfig,
) -> list[LabeledSpeechSegment]:
    if not labeled_segments:
        return []

    smoothed = smooth_short_labels(labeled_segments, config)
    merged = []
    current_label = smoothed[0].label
    current_start_ms = smoothed[0].segment.start_ms
    current_end_ms = smoothed[0].segment.end_ms

    def samples_for_range(start_ms: int, end_ms: int) -> list[float]:
        start = max(0, min(len(source_samples), start_ms * config.sample_rate // 1000))
        end = max(start, min(len(source_samples), end_ms * config.sample_rate // 1000))
        return source_samples[start:end]

    for item in smoothed[1:]:
        gap_ms = item.segment.start_ms - current_end_ms
        combined_duration_ms = item.segment.end_ms - current_start_ms
        should_merge = (
            item.label.label == current_label.label
            and gap_ms <= config.turn_merge_gap_ms
            and combined_duration_ms <= config.max_output_turn_ms
        )
        if should_merge:
            current_end_ms = max(current_end_ms, item.segment.end_ms)
        else:
            merged.append(
                LabeledSpeechSegment(
                    segment=SpeechSegment(
                        samples_for_range(current_start_ms, current_end_ms),
                        current_start_ms,
                        current_end_ms,
                    ),
                    label=current_label,
                )
            )
            current_label = item.label
            current_start_ms = item.segment.start_ms
            current_end_ms = item.segment.end_ms

    merged.append(
        LabeledSpeechSegment(
            segment=SpeechSegment(
                samples_for_range(current_start_ms, current_end_ms),
                current_start_ms,
                current_end_ms,
            ),
            label=current_label,
        )
    )
    return merged


def smooth_short_labels(
    labeled_segments: list[LabeledSpeechSegment],
    config: ImportAudioConfig,
) -> list[LabeledSpeechSegment]:
    smoothed = list(labeled_segments)
    for index, item in enumerate(smoothed):
        duration_ms = item.segment.end_ms - item.segment.start_ms
        if duration_ms >= config.min_output_segment_ms:
            continue

        previous = smoothed[index - 1] if index > 0 else None
        next_item = smoothed[index + 1] if index + 1 < len(smoothed) else None
        previous_gap = item.segment.start_ms - previous.segment.end_ms if previous else None
        next_gap = next_item.segment.start_ms - item.segment.end_ms if next_item else None
        replacement_label = None

        if (
            previous
            and next_item
            and previous.label.label == next_item.label.label
            and previous_gap is not None
            and next_gap is not None
            and previous_gap <= config.turn_merge_gap_ms
            and next_gap <= config.turn_merge_gap_ms
        ):
            replacement_label = previous.label
        elif (
            previous
            and previous.label.label == item.label.label
            and previous_gap is not None
            and previous_gap <= config.turn_merge_gap_ms
        ):
            replacement_label = previous.label
        elif (
            next_item
            and next_item.label.label == item.label.label
            and next_gap is not None
            and next_gap <= config.turn_merge_gap_ms
        ):
            replacement_label = next_item.label
        elif previous and previous_gap is not None and previous_gap <= config.short_segment_absorb_gap_ms:
            replacement_label = previous.label
        elif next_item and next_gap is not None and next_gap <= config.short_segment_absorb_gap_ms:
            replacement_label = next_item.label

        if replacement_label is not None:
            smoothed[index] = LabeledSpeechSegment(segment=item.segment, label=replacement_label)

    return smoothed


def create_audio_segments(
    import_job: MeetingImport,
    segments: list[LabeledSpeechSegment],
    sample_rate: int,
) -> None:
    next_sequence = (
        AudioSegment.objects.filter(meeting=import_job.meeting).aggregate(Max("sequence_number"))[
            "sequence_number__max"
        ]
        or 0
    ) + 1

    for offset, labeled in enumerate(segments):
        sequence_number = next_sequence + offset
        wav_data = write_wav_bytes(labeled.segment.samples, sample_rate)
        audio_segment = AudioSegment(
            meeting=import_job.meeting,
            user=import_job.user,
            client_segment_id=f"import_{import_job.id}_{sequence_number:06d}",
            sequence_number=sequence_number,
            speaker_label=labeled.label.label,
            speaker_confidence=labeled.label.confidence,
            client_start_ms=labeled.segment.start_ms,
            client_end_ms=labeled.segment.end_ms,
            codec="wav_pcm16",
            sample_rate=sample_rate,
            audio_content_type="audio/wav",
            audio_size_bytes=len(wav_data),
        )
        audio_segment.audio_file.save(
            f"import_{sequence_number:06d}-{labeled.label.label}.wav",
            ContentFile(wav_data),
            save=False,
        )
        audio_segment.save()


def write_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            clipped = min(1.0, max(-1.0, sample))
            frames.extend(int(clipped * 32767).to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(frames))
    return output.getvalue()


class VoiceFingerprintExtractor:
    fft_size = 1024

    def extract(self, samples: list[float], sample_rate: int) -> list[float]:
        spectral = self.spectral_stats(samples, sample_rate)
        pitch = self.pitch_stats(samples, sample_rate)
        return [
            zero_crossing_rate(samples) * 3.0,
            spectral["centroid_hz"] / 3000.0,
            spectral["rolloff_hz"] / 5000.0,
            math.log(pitch["median_hz"] + 1.0) / 6.0,
            pitch["spread_hz"] / 100.0,
            pitch["voiced_ratio"],
            *(band * 4.0 for band in spectral["bands"]),
        ]

    def distance(self, left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(size)))

    def blend(
        self,
        left: list[float],
        right: list[float],
        right_weight: float = 0.18,
    ) -> list[float]:
        left_weight = 1.0 - right_weight
        return [
            (left[index] * left_weight) + (right[index] * right_weight)
            for index in range(len(left))
        ]

    def spectral_stats(self, samples: list[float], sample_rate: int) -> dict:
        bands = [0.0] * 6
        if len(samples) < self.fft_size:
            return {"centroid_hz": 0.0, "rolloff_hz": 0.0, "bands": bands}

        average_power = [0.0] * ((self.fft_size // 2) + 1)
        frame_count = min(32, 1 + max((len(samples) - self.fft_size) // self.fft_size, 0))
        step = 0 if frame_count <= 1 else max(1, (len(samples) - self.fft_size) // (frame_count - 1))

        for frame_index in range(frame_count):
            start = frame_index * step
            real = [0.0] * self.fft_size
            imag = [0.0] * self.fft_size
            for index in range(self.fft_size):
                window = 0.5 - 0.5 * math.cos(2.0 * math.pi * index / (self.fft_size - 1))
                real[index] = samples[start + index] * window if start + index < len(samples) else 0.0
            fft(real, imag)
            for bin_index in range(len(average_power)):
                average_power[bin_index] += real[bin_index] ** 2 + imag[bin_index] ** 2

        total = 0.0
        weighted = 0.0
        for bin_index, raw_power in enumerate(average_power):
            power = raw_power / frame_count
            hz = bin_index * sample_rate / self.fft_size
            total += power
            weighted += power * hz
            band = band_index_for_frequency(hz)
            if band >= 0:
                bands[band] += power

        if total <= 0:
            return {"centroid_hz": 0.0, "rolloff_hz": 0.0, "bands": bands}

        cumulative = 0.0
        rolloff_hz = 0.0
        target = total * 0.85
        for bin_index, raw_power in enumerate(average_power):
            cumulative += raw_power / frame_count
            if cumulative >= target:
                rolloff_hz = bin_index * sample_rate / self.fft_size
                break

        return {
            "centroid_hz": weighted / total,
            "rolloff_hz": rolloff_hz,
            "bands": [band / total for band in bands],
        }

    def pitch_stats(self, samples: list[float], sample_rate: int) -> dict:
        frame_size = sample_rate * 40 // 1000
        if len(samples) < frame_size:
            return {"median_hz": 0.0, "spread_hz": 0.0, "voiced_ratio": 0.0}

        min_lag = sample_rate // 320
        max_lag = sample_rate // 70
        step = max(frame_size, (len(samples) - frame_size) // 10)
        pitches = []
        checked_frames = 0
        start = 0
        while start + frame_size <= len(samples):
            if root_mean_square(samples, start, start + frame_size) >= 0.01:
                checked_frames += 1
                pitch = best_pitch(samples, start, frame_size, sample_rate, min_lag, max_lag)
                if pitch is not None:
                    pitches.append(pitch)
            start += step

        if not pitches:
            return {"median_hz": 0.0, "spread_hz": 0.0, "voiced_ratio": 0.0}

        pitches.sort()
        return {
            "median_hz": percentile(pitches, 0.50),
            "spread_hz": percentile(pitches, 0.75) - percentile(pitches, 0.25),
            "voiced_ratio": len(pitches) / max(1, checked_frames),
        }


class BasicSpeakerLabeler:
    min_batch_segments = 8
    local_cluster_min_separation = 0.65
    local_cluster_iterations = 12
    short_sample_ms = 3_000
    new_speaker_sample_ms = 2_500

    def __init__(
        self,
        extractor: VoiceFingerprintExtractor | None = None,
        match_distance_threshold: float = 1.65,
        max_speakers: int = 6,
    ):
        self.extractor = extractor or VoiceFingerprintExtractor()
        self.match_distance_threshold = match_distance_threshold
        self.max_speakers = max_speakers
        self.clusters: list[SpeakerCluster] = []

    def label(self, samples: list[float], sample_rate: int) -> SpeakerLabel:
        trimmed = trim_silence(samples, sample_rate)
        embedding = self.extractor.extract(trimmed, sample_rate)
        duration_ms = len(samples) * 1000 // sample_rate
        return self.label_embedding(embedding, duration_ms)

    def label_batch(self, segments: list[SpeechSegment], sample_rate: int) -> list[SpeakerLabel]:
        if not segments:
            return []
        if len(segments) < self.min_batch_segments:
            return [self.label(segment.samples, sample_rate) for segment in segments]

        embeddings = [
            self.extractor.extract(trim_silence(segment.samples, sample_rate), sample_rate)
            for segment in segments
        ]
        assignments = self.local_assignments(embeddings)
        if len(set(assignments)) < 2:
            return [
                self.label_embedding(embedding, len(segment.samples) * 1000 // sample_rate)
                for segment, embedding in zip(segments, embeddings)
            ]

        should_seed_speakers = not self.clusters
        local_to_global = {}
        for assignment in sorted(set(assignments), key=assignments.index):
            items = [
                embedding
                for index, embedding in enumerate(embeddings)
                if assignments[index] == assignment
            ]
            duration_ms = sum(
                len(segment.samples) * 1000 // sample_rate
                for index, segment in enumerate(segments)
                if assignments[index] == assignment
            )
            local_to_global[assignment] = self.label_embedding(
                centroid(items),
                duration_ms,
                prefer_new_speaker=should_seed_speakers,
            )

        return [local_to_global[assignment] for assignment in assignments]

    def label_embedding(
        self,
        embedding: list[float],
        duration_ms: int,
        *,
        prefer_new_speaker: bool = False,
    ) -> SpeakerLabel:
        if not self.clusters:
            return self.create_cluster(embedding)

        if prefer_new_speaker and len(self.clusters) < self.max_speakers:
            return self.create_cluster(embedding)

        best_cluster = None
        best_distance = float("inf")
        for cluster in self.clusters:
            distance = self.extractor.distance(cluster.centroid, embedding)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

        threshold = (
            self.match_distance_threshold * 1.25
            if duration_ms < self.short_sample_ms
            else self.match_distance_threshold
        )
        if best_cluster and (
            best_distance <= threshold or self.should_reuse_existing(duration_ms)
        ):
            if best_distance <= threshold:
                self.update_cluster(best_cluster, embedding)
            confidence = min(0.98, max(0.35, 1.0 - (best_distance / threshold)))
            return SpeakerLabel(best_cluster.label, confidence)

        if len(self.clusters) < self.max_speakers and (
            prefer_new_speaker or duration_ms >= self.new_speaker_sample_ms
        ):
            return self.create_cluster(embedding)

        fallback = best_cluster or self.clusters[0]
        return SpeakerLabel(fallback.label, 0.35)

    def create_cluster(self, embedding: list[float]) -> SpeakerLabel:
        label = f"person_{len(self.clusters) + 1}"
        self.clusters.append(SpeakerCluster(label=label, centroid=list(embedding)))
        return SpeakerLabel(label, 0.62)

    def update_cluster(self, cluster: SpeakerCluster, embedding: list[float]) -> None:
        cluster.centroid = self.extractor.blend(cluster.centroid, embedding, right_weight=0.18)
        cluster.count += 1

    def should_reuse_existing(self, duration_ms: int) -> bool:
        return bool(self.clusters) and duration_ms < self.short_sample_ms

    def local_assignments(self, embeddings: list[list[float]]) -> list[int]:
        if len(embeddings) < self.min_batch_segments:
            return [0] * len(embeddings)

        first_center = embeddings[0]
        second_index = max(
            range(len(embeddings)),
            key=lambda index: self.extractor.distance(first_center, embeddings[index]),
        )
        initial_separation = self.extractor.distance(first_center, embeddings[second_index])
        if initial_separation < self.local_cluster_min_separation:
            return [0] * len(embeddings)

        left_center = list(first_center)
        right_center = list(embeddings[second_index])
        assignments = [0] * len(embeddings)
        for _ in range(self.local_cluster_iterations):
            for index, embedding in enumerate(embeddings):
                left_distance = self.extractor.distance(left_center, embedding)
                right_distance = self.extractor.distance(right_center, embedding)
                assignments[index] = 0 if left_distance <= right_distance else 1

            left_items = [
                embedding for index, embedding in enumerate(embeddings) if assignments[index] == 0
            ]
            right_items = [
                embedding for index, embedding in enumerate(embeddings) if assignments[index] == 1
            ]
            if len(left_items) < 2 or len(right_items) < 2:
                return [0] * len(embeddings)
            left_center = centroid(left_items)
            right_center = centroid(right_items)

        remap = {}
        next_assignment = 0
        output = []
        for assignment in assignments:
            if assignment not in remap:
                remap[assignment] = next_assignment
                next_assignment += 1
            output.append(remap[assignment])
        return output


def trim_silence(samples: list[float], sample_rate: int) -> list[float]:
    frame_size = sample_rate // 10
    if len(samples) <= frame_size * 2:
        return samples

    def is_speech_frame(start: int) -> bool:
        end = min(start + frame_size, len(samples))
        return root_mean_square(samples, start, end) > 0.006

    start = 0
    while start + frame_size < len(samples) and not is_speech_frame(start):
        start += frame_size

    end = len(samples)
    while end - frame_size > start and not is_speech_frame(end - frame_size):
        end -= frame_size

    return samples[start:end] if end - start >= frame_size * 2 else samples


def centroid(items: list[list[float]]) -> list[float]:
    output = [0.0] * len(items[0])
    for item in items:
        for index, value in enumerate(item):
            output[index] += value
    return [value / len(items) for value in output]


def band_index_for_frequency(hz: float) -> int:
    if hz < 80.0:
        return -1
    if hz < 250.0:
        return 0
    if hz < 500.0:
        return 1
    if hz < 1000.0:
        return 2
    if hz < 2000.0:
        return 3
    if hz < 3800.0:
        return 4
    if hz < 7600.0:
        return 5
    return -1


def best_pitch(
    samples: list[float],
    start: int,
    frame_size: int,
    sample_rate: int,
    min_lag: int,
    max_lag: int,
) -> float | None:
    best_lag = 0
    best_score = 0.0
    for lag in range(min_lag, max_lag + 1):
        limit = frame_size - lag
        corr = 0.0
        left_energy = 0.0
        right_energy = 0.0
        for index in range(limit):
            left = samples[start + index]
            right = samples[start + index + lag]
            corr += left * right
            left_energy += left * left
            right_energy += right * right
        score = corr / math.sqrt(max(left_energy * right_energy, 0.0000001))
        if score > best_score:
            best_score = score
            best_lag = lag

    return sample_rate / best_lag if best_lag > 0 and best_score > 0.35 else None


def percentile(values: list[float], requested_percentile: float) -> float:
    if not values:
        return 0.0
    index = int((len(values) - 1) * requested_percentile)
    index = min(max(index, 0), len(values) - 1)
    return values[index]


def root_mean_square(samples: list[float], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return math.sqrt(sum(samples[index] * samples[index] for index in range(start, end)) / (end - start))


def rms_db(samples: list[float], start: int, end: int) -> float:
    if end <= start:
        return -120.0
    rms = max(root_mean_square(samples, start, end), 0.000001)
    return 20.0 * math.log10(rms)


def zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous_positive = samples[0] >= 0
    for sample in samples[1:]:
        positive = sample >= 0
        if positive != previous_positive:
            crossings += 1
        previous_positive = positive
    return crossings / (len(samples) - 1)


def fft(real: list[float], imag: list[float]) -> None:
    n = len(real)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]

    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        w_length_real = math.cos(angle)
        w_length_imag = math.sin(angle)
        i = 0
        while i < n:
            w_real = 1.0
            w_imag = 0.0
            for k in range(length // 2):
                even = i + k
                odd = even + (length // 2)
                odd_real = real[odd] * w_real - imag[odd] * w_imag
                odd_imag = real[odd] * w_imag + imag[odd] * w_real

                real[odd] = real[even] - odd_real
                imag[odd] = imag[even] - odd_imag
                real[even] += odd_real
                imag[even] += odd_imag

                next_real = w_real * w_length_real - w_imag * w_length_imag
                w_imag = w_real * w_length_imag + w_imag * w_length_real
                w_real = next_real
            i += length
        length <<= 1
