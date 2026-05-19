SUPPORTED_IMPORT_AUDIO_EXTENSIONS = {"m4a", "mp3", "mp4", "wav"}
COMPRESSED_IMPORT_AUDIO_EXTENSIONS = {"m4a", "mp3", "mp4"}


def supported_import_audio_message() -> str:
    return ", ".join(sorted(SUPPORTED_IMPORT_AUDIO_EXTENSIONS))
