"""Optional upstream MCP connector, including an existing Unsloth Studio."""
import asyncio
import json
from .endpoint import failure
from .inference import connection_url, _credential


async def _request(connection, tool=None, arguments=None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    import httpx
    url = connection_url(connection, mcp=True)
    token = _credential(connection)
    headers = {"Authorization": "Bearer " + token} if token else {}

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout or 20, auth=auth, trust_env=False, follow_redirects=False)

    async with streamablehttp_client(url, headers=headers, timeout=20, sse_read_timeout=30,
                                     httpx_client_factory=client_factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if tool is None:
                response = await session.list_tools()
                result = {"ok": True, "connection_id": connection["id"], "state": "tools_listed",
                          "tools": [{"name": t.name, "description": (t.description or "")[:500], "input_schema": t.inputSchema} for t in response.tools[:100]],
                          "truncated": len(response.tools) > 100 or bool(response.nextCursor), "execution_verified": False}
            else:
                response = await session.call_tool(tool, arguments or {})
                result = {"ok": not response.isError, "connection_id": connection["id"], "tool": tool,
                          "content": [item.model_dump(mode="json") for item in response.content if item.type == "text"]}
            serialized = json.dumps(result)
            if len(serialized) > 1024 * 1024:
                return failure("response_too_large", "Upstream MCP response exceeded the limit.")
            if token:
                serialized = serialized.replace(token, "[REDACTED]")
            return json.loads(serialized)


def request(connection, tool=None, arguments=None):
    if tool is not None and (not isinstance(tool, str) or not 1 <= len(tool) <= 128 or not isinstance(arguments, dict)):
        return failure("invalid_input", "Invalid upstream tool call.")
    try:
        return asyncio.run(asyncio.wait_for(_request(connection, tool, arguments), timeout=60))
    except ImportError:
        return failure("dependency_missing", "Install the AI Lab MCP extra to use this connector.")
    except Exception:
        return failure("mcp_unverified", "The configured MCP server could not complete this request.")
