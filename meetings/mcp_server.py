import hmac
import os
from urllib.parse import urlparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
from asgiref.sync import sync_to_async
from django.conf import settings
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

django.setup()

from . import mcp_api


class StaticBearerTokenVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        expected = settings.MCP_AUTH_TOKEN
        if expected and hmac.compare_digest(token, expected):
            return AccessToken(
                token=token,
                client_id="meeting-transcribe-agent",
                scopes=["meetings:read", "meetings:write"],
            )
        return None


def build_server() -> FastMCP:
    if not settings.MCP_AUTH_TOKEN:
        raise RuntimeError("MCP_AUTH_TOKEN must be configured before starting the MCP server.")
    if not settings.MCP_DEFAULT_USERNAME:
        raise RuntimeError("MCP_DEFAULT_USERNAME must be configured before starting the MCP server.")

    public_url = settings.MCP_PUBLIC_URL
    issuer_url = public_url.rsplit("/mcp", 1)[0] or public_url
    public_host = urlparse(public_url).netloc
    return FastMCP(
        name="android-meeting-transcribe",
        instructions=(
            "Use this server to perform the same meeting-recorder tasks available to the "
            "configured authenticated Django user: inspect meetings, start/end meetings, upload "
            "audio segments, import previous recordings, check processing progress, rebuild "
            "message summaries, extract meeting minutes, and retrieve PM notes PDFs."
        ),
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        token_verifier=StaticBearerTokenVerifier(),
        auth=AuthSettings(
            issuer_url=issuer_url,
            resource_server_url=public_url,
            required_scopes=["meetings:read", "meetings:write"],
        ),
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                public_host,
                f"{settings.MCP_HOST}:{settings.MCP_PORT}",
                "localhost",
                f"localhost:{settings.MCP_PORT}",
            ],
            allowed_origins=[issuer_url],
        ),
    )


mcp = build_server()


@mcp.tool()
async def get_current_user() -> dict:
    """Return the configured Django user that this MCP server acts as."""
    return await sync_to_async(mcp_api.get_current_user, thread_sensitive=True)()


@mcp.tool()
async def list_meeting_types() -> dict:
    """List meeting types accepted by extract_meeting_minutes, including which support PDF output."""
    return await sync_to_async(mcp_api.list_meeting_types, thread_sensitive=True)()


@mcp.tool()
async def list_meetings(limit: int = 20, status: str = "") -> dict:
    """List meetings for the configured MCP user. Optional status: recording, ended, complete, failed."""
    return await sync_to_async(mcp_api.list_meetings, thread_sensitive=True)(
        limit=limit,
        status=status,
    )


@mcp.tool()
async def start_meeting(title: str = "") -> dict:
    """Create a new recording meeting for the configured user."""
    return await sync_to_async(mcp_api.start_meeting, thread_sensitive=True)(title=title)


@mcp.tool()
async def end_meeting(meeting_id: str, ended_at: str = "", rebuild_outputs: bool = True) -> dict:
    """End a currently recording meeting. ended_at may be an ISO-8601 datetime."""
    return await sync_to_async(mcp_api.end_meeting, thread_sensitive=True)(
        meeting_id,
        ended_at=ended_at,
        rebuild_outputs=rebuild_outputs,
    )


@mcp.tool()
async def get_meeting(meeting_id: str) -> dict:
    """Get meeting metadata, import status, transcript segments, processed messages, and minutes."""
    return await sync_to_async(mcp_api.get_meeting, thread_sensitive=True)(meeting_id)


@mcp.tool()
async def get_meeting_progress(meeting_id: str) -> dict:
    """Get the same processing progress payload shown on the meeting detail page."""
    return await sync_to_async(mcp_api.get_meeting_progress, thread_sensitive=True)(meeting_id)


@mcp.tool()
async def import_recording_from_url(
    recording_url: str,
    title: str = "",
    original_filename: str = "",
    process_now: bool = False,
) -> dict:
    """Queue a previous WAV, MP3, M4A, or MP4 recording from an HTTP(S) URL for background segmentation and transcription."""
    return await sync_to_async(mcp_api.import_recording_from_url, thread_sensitive=True)(
        recording_url=recording_url,
        title=title,
        original_filename=original_filename,
        process_now=process_now,
    )


@mcp.tool()
async def import_recording_from_base64(
    filename: str,
    content_base64: str,
    title: str = "",
    content_type: str = "",
) -> dict:
    """Queue a previous WAV, MP3, M4A, or MP4 recording supplied as base64 content."""
    return await sync_to_async(mcp_api.import_recording_from_base64, thread_sensitive=True)(
        filename=filename,
        content_base64=content_base64,
        title=title,
        content_type=content_type,
    )


@mcp.tool()
async def upload_audio_segment_from_url(
    meeting_id: str,
    audio_url: str,
    sequence_number: int,
    speaker_label: str,
    client_start_ms: int,
    client_end_ms: int,
    client_segment_id: str = "",
    speaker_confidence: float | None = None,
    codec: str = "",
    sample_rate: int | None = None,
    original_filename: str = "",
    process_now: bool = False,
) -> dict:
    """Upload one meeting audio segment from an HTTP(S) URL, matching the mobile app segment upload flow."""
    return await sync_to_async(mcp_api.upload_audio_segment_from_url, thread_sensitive=True)(
        meeting_id=meeting_id,
        audio_url=audio_url,
        sequence_number=sequence_number,
        speaker_label=speaker_label,
        client_start_ms=client_start_ms,
        client_end_ms=client_end_ms,
        client_segment_id=client_segment_id,
        speaker_confidence=speaker_confidence,
        codec=codec,
        sample_rate=sample_rate,
        original_filename=original_filename,
        process_now=process_now,
    )


@mcp.tool()
async def upload_audio_segment_from_base64(
    meeting_id: str,
    filename: str,
    content_base64: str,
    sequence_number: int,
    speaker_label: str,
    client_start_ms: int,
    client_end_ms: int,
    client_segment_id: str = "",
    speaker_confidence: float | None = None,
    codec: str = "",
    sample_rate: int | None = None,
    content_type: str = "",
    process_now: bool = False,
) -> dict:
    """Upload one meeting audio segment supplied as base64 content."""
    return await sync_to_async(mcp_api.upload_audio_segment_from_base64, thread_sensitive=True)(
        meeting_id=meeting_id,
        filename=filename,
        content_base64=content_base64,
        sequence_number=sequence_number,
        speaker_label=speaker_label,
        client_start_ms=client_start_ms,
        client_end_ms=client_end_ms,
        client_segment_id=client_segment_id,
        speaker_confidence=speaker_confidence,
        codec=codec,
        sample_rate=sample_rate,
        content_type=content_type,
        process_now=process_now,
    )


@mcp.tool()
async def process_one_import_job() -> dict:
    """Process one pending imported recording immediately. The background worker also does this automatically."""
    return await sync_to_async(mcp_api.process_one_import_job, thread_sensitive=True)()


@mcp.tool()
async def process_one_transcription_job() -> dict:
    """Process one pending audio segment immediately. The background worker also does this automatically."""
    return await sync_to_async(mcp_api.process_one_transcription_job, thread_sensitive=True)()


@mcp.tool()
async def extract_meeting_minutes(meeting_id: str, meeting_type: str, wait: bool = False) -> dict:
    """Queue minutes extraction. Set wait=true only when the caller can tolerate a long synchronous OpenAI run."""
    return await sync_to_async(mcp_api.extract_meeting_minutes, thread_sensitive=True)(
        meeting_id,
        meeting_type,
        wait=wait,
    )


@mcp.tool()
async def get_project_manager_notes_pdf(meeting_id: str, meeting_type: str = "") -> dict:
    """Return a PM notes PDF as base64 content. meeting_type may be project_manager_notes or compact_pm_notes."""
    return await sync_to_async(mcp_api.get_project_manager_notes_pdf, thread_sensitive=True)(
        meeting_id,
        meeting_type=meeting_type,
    )


@mcp.tool()
async def rebuild_messages_and_summaries(meeting_id: str) -> dict:
    """Rebuild grouped display messages, detailed summaries, short summaries, and the meeting title."""
    return await sync_to_async(mcp_api.rebuild_messages_and_summaries, thread_sensitive=True)(
        meeting_id
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
