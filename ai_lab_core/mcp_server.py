"""Optional official MCP SDK adapter, using stdio only."""
import asyncio
from . import tool_api


async def serve():
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    server = Server('hermes-ai-lab', version=tool_api.API_VERSION)

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=item['name'], description=item['description'],
                           inputSchema=item['parameters']) for item in tool_api.schemas()]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        # Connector services may own an asyncio.run loop; keep them off ours.
        result = await asyncio.to_thread(tool_api.dispatch, name, arguments)
        return types.CallToolResult(content=[types.TextContent(type='text', text=tool_api.encode(result))],
                                    isError=not result['ok'])

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    try:
        asyncio.run(serve())
    except ImportError:
        # Protocol traffic owns stdout; installation diagnostics belong on stderr.
        import sys
        print('MCP requires the optional mcp extra (SDK >=1.28.1,<2).', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
