"""A generic HTTP tool.

This is the one tool that breaks the "nothing leaves the machine" property, so
it is explicit about it:

  * Requests to private and loopback addresses are refused. Otherwise the agent
    could be talked into probing the LAN, or into calling this project's own
    API on 127.0.0.1 -- a server-side request forgery against ourselves.
  * Only http and https schemes, so `file://` cannot be used to read the disk
    around the file sandbox.
  * Responses are truncated, because an unbounded body would blow the context.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from voiceagent.tools.base import Tool, ToolResult

MAX_RESPONSE_CHARS = 12_000
TIMEOUT_SECONDS = 15.0


def _is_private(hostname: str) -> bool:
    """True if the hostname resolves to a loopback/private/link-local address."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False  # unresolvable; let the request fail normally
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return True
    return False


class HttpRequestTool(Tool):
    name = "http_request"
    description = (
        "Make an HTTP GET or POST request to a public web API and return the "
        "response body. Use this for looking up live information."
    )
    #: This is the only tool that sends data off the machine.
    requires_confirmation = True

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full https:// URL."},
                "method": {"type": "string", "enum": ["GET", "POST"], "description": "Defaults to GET."},
                "json_body": {
                    "type": "object",
                    "description": "Optional JSON body for POST requests.",
                },
            },
            "required": ["url"],
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        url = (kwargs.get("url") or "").strip()
        method = (kwargs.get("method") or "GET").upper()

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                content="", ok=False,
                error=f"only http and https are allowed, got {parsed.scheme!r}",
            )
        if not parsed.hostname:
            return ToolResult(content="", ok=False, error="URL has no hostname")
        if _is_private(parsed.hostname):
            return ToolResult(
                content="", ok=False,
                error=(
                    f"refusing to call {parsed.hostname}: it resolves to a private or "
                    "loopback address, which is not a public API."
                ),
            )
        if method not in ("GET", "POST"):
            return ToolResult(content="", ok=False, error=f"unsupported method {method!r}")

        try:
            import httpx
        except ImportError:
            return ToolResult(
                content="", ok=False,
                error="httpx is not installed; run: uv sync --extra tools",
            )

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = await client.request(method, url, json=kwargs.get("json_body"))
        except Exception as exc:  # noqa: BLE001 -- network errors are expected
            return ToolResult(content="", ok=False, error=f"request failed: {exc}")

        body = response.text
        if len(body) > MAX_RESPONSE_CHARS:
            body = body[:MAX_RESPONSE_CHARS] + f"\n... (truncated at {MAX_RESPONSE_CHARS} chars)"

        if response.status_code >= 400:
            return ToolResult(content=body, ok=False, error=f"HTTP {response.status_code}")
        return ToolResult(content=f"HTTP {response.status_code}\n{body}")
