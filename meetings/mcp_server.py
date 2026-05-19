import hmac
import os
from urllib.parse import urlparse

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
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
            "Use this server to inspect meeting recordings, import previous WAV recordings, "
            "check transcription and segmentation status, rebuild message summaries, and extract "
            "meeting minutes for the configured authenticated Django user."
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
def list_meetings(limit: int = 20, status: str = "") -> dict:
    """List meetings for the configured MCP user. Optional status: recording, ended, complete, failed."""
    return mcp_api.list_meetings(limit=limit, status=status)


@mcp.tool()
def get_meeting(meeting_id: str) -> dict:
    """Get meeting metadata, import status, transcript segments, processed messages, and minutes."""
    return mcp_api.get_meeting(meeting_id)


@mcp.tool()
def import_recording_from_url(
    recording_url: str,
    title: str = "",
    original_filename: str = "",
    process_now: bool = False,
) -> dict:
    """Queue a previous PCM WAV recording from an HTTP(S) URL for background segmentation and transcription."""
    return mcp_api.import_recording_from_url(
        recording_url=recording_url,
        title=title,
        original_filename=original_filename,
        process_now=process_now,
    )


@mcp.tool()
def process_one_import_job() -> dict:
    """Process one pending imported recording immediately. The background worker also does this automatically."""
    return mcp_api.process_one_import_job()


@mcp.tool()
def extract_meeting_minutes(meeting_id: str, meeting_type: str) -> dict:
    """Extract minutes or gathered requirements for a completed meeting using one of the configured meeting types."""
    return mcp_api.extract_meeting_minutes(meeting_id, meeting_type)


@mcp.tool()
def rebuild_messages_and_summaries(meeting_id: str) -> dict:
    """Rebuild grouped display messages, detailed summaries, short summaries, and the meeting title."""
    return mcp_api.rebuild_messages_and_summaries(meeting_id)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
