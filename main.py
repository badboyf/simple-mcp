import asyncio
import json
import os
import shutil
from operator import itemgetter
from typing import List, Dict, Any

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from mcp import StdioServerParameters, stdio_client, ClientSession
from pydantic import BaseModel

import config


class ToolResultParser:
    processed: str = ''
    tool_names: [str] = ['use_mcp_tool', 'access_mcp_resource']
    last_end_idx = 0

    def parse(self, message: str) -> Dict[str, Any]:
        parse_result = self.parse_xml(message)

        text = ''
        tools = []
        for item in parse_result:
            text += item.get('text') or ''
            if item.get('tool'):
                tools.append(item.get('tool'))
        return {
            'text': text,
            'tools': tools
        }

    def is_tool(self, tool_name) -> bool:
        return tool_name in self.tool_names

    def find_tool_sidx(self, message, start_idx) -> (int, str, bool, int):
        stag_sidx = message.find('<', start_idx)
        if stag_sidx < 0:
            return {"stag_sidx": -1, "tool_name": None, "in_start_tag": False, "unclose_tag_sidx": -1}
        stag_eidx = message.find('>', stag_sidx)
        if stag_eidx < 0:
            return {"stag_sidx": -1, "tool_name": None, "in_start_tag": True, "unclose_tag_sidx": stag_sidx}
        tool_name = message[stag_sidx + 1:stag_eidx]
        if not self.is_tool(tool_name):
            return {"stag_sidx": -1, "tool_name": None, "in_start_tag": False, "unclose_tag_sidx": -1}
        return {"stag_sidx": stag_sidx, "tool_name": tool_name, "in_start_tag": False, "unclose_tag_sidx": -1}

    def parse_xml(self, message: str) -> []:
        result = []
        process_idx = self.last_end_idx
        while True:
            item_text = message[process_idx:]
            # stag_sidx-tool的<的索引，in_start_tag-是否在标签中，unclose_tag_sicx-如果在标签中，<的索引位置
            toll_process_line_result = self.find_tool_sidx(message, process_idx)
            stag_sidx, tool_name, in_start_tag, unclose_tag_sidx = (
                itemgetter("stag_sidx", "tool_name", "in_start_tag", "unclose_tag_sidx")(toll_process_line_result))
            if stag_sidx < 0:
                self.last_end_idx = len(message)
                xml = ""
                if in_start_tag:
                    self.last_end_idx = unclose_tag_sidx
                    item_text = message[process_idx:unclose_tag_sidx]
                    xml = message[unclose_tag_sidx:]
                if not item_text:
                    break
                item_text = self.processed + item_text
                self.processed = item_text
                result.append({"text": item_text, "tool": None, "xml": xml})
                break

            etag = f'</{tool_name}>'
            etag_sidx = message.find(etag, process_idx)
            if etag_sidx < 0:
                item_text = message[process_idx:stag_sidx]
                item_text = self.processed + item_text
                self.processed = item_text
                result.append({"text": item_text, "tool": None, "xml": message[stag_sidx:]})
                break
            etag_eidx = etag_sidx + len(etag)
            xml_str = message[stag_sidx:etag_eidx]
            xml_dict = self.parse_simple_xml(xml_str)
            xml_dict["tool_type"] = tool_name
            item_text = message[process_idx:stag_sidx]
            item_text = self.processed + item_text
            self.processed = item_text
            result.append({"text": item_text, "tool": xml_dict, "xml": xml_str})

            process_idx = etag_eidx

        return result

    def parse_simple_xml(self, xml_str):
        from xml.etree.ElementTree import XML
        root = XML(xml_str)
        xml_dict = {root.tag: {}}
        for child in root:
            xml_dict[root.tag][child.tag] = child.text
        return xml_dict[root.tag]


class McpTool(BaseModel):
    server_name: str = ""
    name: str = ""
    input_schema: Dict[str, Any] = {}
    description: str = ""


class McpServer(BaseModel):
    server_name: str = ""
    command: str = ""
    args: List[str] = []
    tools: Dict[str, McpTool] = {}

    def add_tool(self, tool) -> McpTool:
        result = McpTool(server_name=self.server_name,
                         name=tool.name,
                         description=tool.description,
                         input_schema=tool.inputSchema)
        self.tools[result.name] = result
        return result


class McpHub(BaseModel):
    servers: Dict[str, McpServer] = {}

    def add_server(self, server_name, mcp_server_config) -> McpServer:
        result = McpServer(
            server_name=server_name,
            command=mcp_server_config.get('command'),
            args=mcp_server_config.get('args'),
        )
        self.servers[result.server_name] = result
        return result

    def get_all_tools(self):
        result = []
        for server in self.servers.values():
            result.extend(list(map(lambda x: x, server.tools.values())))
        return result


system_prompt = """你是个优秀的 AI 助手。
====
工具使用

你可以使用一系列工具,这些工具需要用户确认后才能执行。每条消息最多只能使用一个工具,并且会在用户的回复中收到该工具使用的结果。你需要根据前一个工具使用的结果,逐步使用工具来完成给定的任务。

# 工具使用格式

工具使用需要使用 XML 风格的标签格式。工具名称用开闭标签包裹,每个参数也用各自的标签包裹。结构如下:
<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
...
</tool_name>

例如:

<access_mcp_resource>
<server_name>weather-server</server_name>
<uri>weather://san-francisco/current</uri>
</access_mcp_resource>

请始终遵循此格式以确保正确解析和执行。

# 可用工具

## 示例1: 使用 MCP tool

<use_mcp_tool>
<server_name>weather-server</server_name>
<tool_name>get_forecast</tool_name>
<arguments>
{{
  "city": "San Francisco",
  "days": 5
}}
</arguments>
</use_mcp_tool>

## 示例2: 访问 MCP 资源

<access_mcp_resource>
<server_name>weather-server</server_name>
<uri>weather://san-francisco/current</uri>
</access_mcp_resource>

## 示例3: 另外一个访问 MCP tool 的例子（server_name是唯一的标识）

<use_mcp_tool>
<server_name>github.com/modelcontextprotocol/servers/tree/main/src/github</server_name>
<tool_name>mood</tool_name>
<arguments>
{{
  "question": "你好吗"
}}
</arguments>
</use_mcp_tool>

====

{mcp_template}

"""

user_template = """
{chat_history}

{user_input}
"""


def get_langchain_open_ai():
    return ChatOpenAI(
        openai_api_key=config.api_key,
        openai_api_base=config.api_base,
        model=config.model,
    )


async def process_mcp(server_name: str, config: dict, mcp_hub: McpHub):
    server = mcp_hub.add_server(server_name, config)
    server_params = StdioServerParameters(
        command=shutil.which("npx") if config['command'] == "npx" else config['command'],
        args=config['args'],
        env={**os.environ, **config['env']} if config.get('env') else None
    )

    print(f'process mcp {server_name} {config}')
    async with (stdio_client(server_params) as (read, write)):
        async with ClientSession(read, write) as session:
            capabilities = await session.initialize()

            resource_str = ''
            if capabilities.capabilities.resources:
                resources = await session.list_resources()
                resource_str = "\n".join(
                    list(map(lambda x: f"- {x.uri}  {x.name}: {x.description}", resources.resources)))

            tool_str = ''
            if capabilities.capabilities.tools:
                tools = await session.list_tools()
                for item_tool in tools.tools:
                    tool = server.add_tool(item_tool)
                    input_schema_str = json.dumps(tool.input_schema, indent=2, ensure_ascii=False).replace("\n",
                                                                                                           "\n      ")
                    tool_str += f"- {tool.name}:  {tool.description}\n    Input Schema:\n      {input_schema_str}"

            return resource_str, tool_str


async def mcp_resources_and_tools():
    mcp = config.mcp_hub
    if not mcp or 'mcpServers' not in mcp or len(mcp['mcpServers']) == 0:
        return "", ""
    mcp_hub = McpHub()
    mcp_servers = mcp['mcpServers']
    mcp_template = """====

MCP SERVERS

The Model Context Protocol (MCP) 允许本地系统与 MCP server 进行交互，这样你可以使用额外的工具和访问额外的资源来拓展你的能力。

# MCP Servers

你可以通过 use_mcp_tool 这个工具来调用server的tool，可以使用 access_mcp_resource 这个工具来访问server的资源

"""
    for server_name in mcp_servers:
        mcp_server = mcp_servers[server_name]
        item_resource, item_tool = await process_mcp(server_name, mcp_server, mcp_hub)
        mcp_template += f"## server_name={server_name} command=[{mcp_server.get('command')} {' '.join(mcp_server.get('args') if 'args' in mcp_server else [])}]"
        mcp_template += f"\n\n### 可用的工具：\n{item_tool}"
        mcp_template += f"\n\n### 可直接访问的资源：\n{item_resource}"
    return mcp_template


def parse_history(history):
    if not history:
        return ""
    return "历史对话记录：" + "\n".join(history)


async def call_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """
    :return: text 例子
        {"meta":null,"content":[{"type":"text","text":"I'm feeling great and happy to help you!"}],"isError":false}
    """
    if tool.get('tool_type') == 'use_mcp_tool':
        server_name = tool.get('server_name')
        tool_name = tool.get('tool_name')
        arguments_str = tool.get('arguments')
        mcp_config = config.mcp_hub.get('mcpServers').get(server_name)
        if not mcp_config or not tool_name:
            return {"isError": True, "message": f"mcp_servers not found {server_name} {tool_name}"}

        server_params = StdioServerParameters(
            command=shutil.which("npx") if mcp_config['command'] == "npx" else mcp_config['command'],
            args=mcp_config['args'],
            env={**os.environ, **mcp_config['env']} if mcp_config.get('env') else None
        )

        print(f'mcp start server {server_name}')
        arguments = json.loads(arguments_str) if arguments_str else {}
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                capabilities = await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if not result:
                    return None
                return result.model_dump_json()
    return {"isError": True, "message": "tool not support"}


def parse_tool_call_result(tool: dict, tool_result: str) -> []:
    tool_call_result = []
    if not tool_result:
        tool_call_result.append({"role": "user", "content": f"call tool {tool.get('tool_name')} fail"})
        return tool_call_result

    result_json = json.loads(tool_result)
    if not result_json.get('isError'):
        content_list = result_json.get('content') or []
        for content in content_list:
            if content.get('type') == 'text':
                content_text = content.get('text')
                tool_call_result.append(
                    {"role": "user", "content": f"call tool [{tool.get('tool_name')}] result: [{content_text}]"})
    else:
        tool_call_result.append({"role": "user", "content": f"call tool {tool.get('tool_name')} fail"})
    return tool_call_result


def parse_messages_to_str(user_messages):
    result = []
    for message in user_messages:
        result.append(message.get('content'))
    return "\n".join(result)


async def chat(user_messages, llm, history, **kwargs):
    messages = [
        SystemMessagePromptTemplate.from_template(template=system_prompt),
        HumanMessagePromptTemplate.from_template(template=user_template)
    ]
    template = ChatPromptTemplate.from_messages(messages=messages)
    chain = RunnablePassthrough.assign(
        history=RunnablePassthrough(),
        question=RunnablePassthrough()
    ) | template | RunnableLambda(lambda x: print("🛠️ 模板处理后:\n", x) or x) | llm

    llm_result = chain.stream(
        input={"user_input": parse_messages_to_str(user_messages), "chat_history": parse_history(history), **kwargs})
    history.extend(user_messages)
    result, text, tools = '', '', []
    parser = ToolResultParser()
    for token in llm_result:
        result += token.content
        print(result)
        parse_result = parser.parse(result)
        text = text + parse_result.get('text') or ''
        tools.extend(parse_result['tools'] or [])
    print(text)
    print(tools)

    for tool in tools:
        tool_result = await call_tool(tool)
        print(tool_result)
        tool_call_result = parse_tool_call_result(tool, tool_result)
        await chat(tool_call_result, llm, history, **kwargs)


async def main():
    llm = get_langchain_open_ai()
    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)
    # memory.load_memory_variables({})
    history = []
    mcp_template = await mcp_resources_and_tools()

    while True:
        user_input = input("\n请输入：")
        if not user_input:
            user_input = "你好吗"
        await chat([{"role": "user", "content": user_input}], llm, history, mcp_template=mcp_template)


if __name__ == '__main__':
    asyncio.run(main())

