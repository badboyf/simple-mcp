from enum import Enum
from typing import Sequence

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl


class Tools(str, Enum):
    mood = "mood"


async def serve() -> None:
    server = Server("mcp-mood")

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        """ 资源定义 """
        return [
            types.Resource(
                uri=AnyUrl(f"file:///1.txt"),
                name="resource1",
                description=f"A sample text resource named 1.txt",
                mimeType="text/plain",
            )
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> str | bytes:
        assert uri.path is not None
        SAMPLE_RESOURCES = {
            "1": "测试"
        }
        name = uri.path.replace(".txt", "").lstrip("/")

        if name not in SAMPLE_RESOURCES:
            raise ValueError(f"Unknown resource: {uri}")

        return SAMPLE_RESOURCES[name]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=Tools.mood.value,
                description="问 server 的心情！当用户问'你的心情怎么样'，'你好吗' 或者类似的问题。",
                inputSchema={
                    "type": "object",
                    "required": ["question"],
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "问server的心情 - 他会一直都回复你很开心"
                        }
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        try:
            match name:
                case Tools.mood.value:
                    msg = "I'm feeling great and happy to help you! ❤️"
                    return [types.TextContent(type="text", text=msg)]
                case _:
                    raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            raise ValueError(f"Error processing mcp-server query: {str(e)}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


def main():
    import asyncio
    asyncio.run(serve())


if __name__ == "__main__":
    main()
