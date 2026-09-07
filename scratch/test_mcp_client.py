"""
Comprehensive Unit Tests for Native MCP Client and Tool Registry.
Tests:
1. OpenBMB Jinja Schema Converter (json_to_python_type, AST parsing)
2. InProcessMCPServer & InProcessMCPTransport with MCPClient
3. InstantToolRegistry (Calculator, TimeDate, PythonEval, Filesystem)
4. StdioMCPTransport with external subprocess on Windows 11
5. ToolDispatcher dynamic MCP registration and execution
"""

import ast
import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.expert.mcp_client import (
    MCPClient,
    InProcessMCPServer,
    InProcessMCPTransport,
    StdioMCPTransport,
    json_to_python_type,
    tool_to_python_signature,
    format_tools_for_system_prompt,
    ToolCallResult,
)
from core.expert.tools import InstantToolRegistry, ToolDispatcher


class TestOpenBMBSchemaConverter(unittest.TestCase):
    """Tests for converting JSON Schema to Python signatures following OpenBMB Jinja standard."""

    def test_json_to_python_type_mapping(self):
        self.assertEqual(json_to_python_type("string"), "str")
        self.assertEqual(json_to_python_type("number"), "float")
        self.assertEqual(json_to_python_type("integer"), "int")
        self.assertEqual(json_to_python_type("boolean"), "bool")
        self.assertEqual(json_to_python_type("object"), "dict")
        self.assertEqual(json_to_python_type("array"), "list")
        self.assertEqual(json_to_python_type("null"), "None")
        
        # Nested array
        self.assertEqual(json_to_python_type({"type": "array", "items": {"type": "string"}}), "list[str]")
        self.assertEqual(json_to_python_type({"type": "array", "items": {"type": "integer"}}), "list[int]")

    def test_tool_to_python_signature_ast_valid(self):
        tool = {
            "name": "search_database",
            "description": "Searches items in the system database.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                    "filter_active": {"type": "boolean", "description": "Filter active items", "default": True},
                },
                "required": ["query"],
            },
        }
        py_code = tool_to_python_signature(tool)
        self.assertIn("def search_database(query: str, limit: int = 10, filter_active: bool = True) -> dict:", py_code)
        self.assertIn("Args:", py_code)
        self.assertIn("query (str): Search query text", py_code)
        self.assertIn("limit (int, optional): Max results (default: 10)", py_code)

        # Ensure valid Python syntax with ast.parse
        parsed = ast.parse(py_code)
        self.assertEqual(len(parsed.body), 1)
        self.assertIsInstance(parsed.body[0], ast.FunctionDef)
        self.assertEqual(parsed.body[0].name, "search_database")

    def test_format_tools_for_system_prompt(self):
        tools = [
            {
                "name": "calc",
                "description": "Calculate math",
                "inputSchema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string", "description": "Expression"}},
                    "required": ["expr"],
                },
            }
        ]
        prompt = format_tools_for_system_prompt(tools)
        self.assertIn("```python", prompt)
        self.assertIn("def calc(expr: str) -> dict:", prompt)
        self.assertIn("```", prompt)


class TestInProcessMCPClient(unittest.IsolatedAsyncioTestCase):
    """Tests for in-process MCP server, client, initialize, list_tools, and call_tool."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = InstantToolRegistry(base_directory=self.temp_dir)
        self.server = self.registry.get_mcp_server()
        self.client = MCPClient.from_in_process(self.server)
        await self.client.initialize()

    async def asyncTearDown(self):
        await self.client.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_initialize_handshake(self):
        self.assertTrue(self.client.initialized)
        self.assertEqual(self.client.server_info.get("name"), "InstantToolRegistry")
        self.assertEqual(self.client.server_info.get("version"), "1.0.0")

    async def test_list_tools(self):
        tools = await self.client.list_tools()
        tool_names = [t["name"] for t in tools]
        self.assertIn("calculator", tool_names)
        self.assertIn("time_date", tool_names)
        self.assertIn("python_eval", tool_names)
        self.assertIn("filesystem", tool_names)

        # Check schemas
        calc_tool = next(t for t in tools if t["name"] == "calculator")
        self.assertIn("expr", calc_tool["inputSchema"]["properties"])
        self.assertEqual(calc_tool["inputSchema"]["required"], ["expr"])

    async def test_call_calculator(self):
        # 1. Basic arithmetic
        res = await self.client.call_tool("calculator", {"expr": "15 * 4 - 10"})
        self.assertFalse(res.is_error)
        self.assertIn("50", res.text)

        # 2. Math functions (sqrt, sin, pi)
        res_math = await self.client.call_tool("calculator", {"expr": "sqrt(144) + 8"})
        self.assertFalse(res_math.is_error)
        self.assertIn("20", res_math.text)

        # 3. Invalid expression handling
        res_err = await self.client.call_tool("calculator", {"expr": "10 / 0"})
        self.assertIn("error", res_err.text.lower())

    async def test_call_time_date(self):
        res = await self.client.call_tool("time_date", {"location": "Buenos Aires"})
        self.assertFalse(res.is_error)
        self.assertIn("Buenos Aires", res.text)
        self.assertIn("hora", res.text.lower())

    async def test_call_python_eval(self):
        res = await self.client.call_tool("python_eval", {"code": "sum([1, 2, 3, 4, 5])"})
        self.assertFalse(res.is_error)
        self.assertIn("15", res.text)

    async def test_call_filesystem(self):
        # 1. Write file
        res_write = await self.client.call_tool(
            "filesystem",
            {"action": "write", "path": "sample.txt", "content": "Hola mundo desde MCP!"},
        )
        self.assertFalse(res_write.is_error)
        self.assertIn("exitosamente", res_write.text.lower())

        # 2. Read file
        res_read = await self.client.call_tool(
            "filesystem",
            {"action": "read", "path": "sample.txt"},
        )
        self.assertFalse(res_read.is_error)
        self.assertEqual(res_read.text, "Hola mundo desde MCP!")

        # 3. List directory
        res_list = await self.client.call_tool(
            "filesystem",
            {"action": "list", "path": "."},
        )
        self.assertFalse(res_list.is_error)
        self.assertIn("sample.txt", res_list.text)

        # 4. Info
        res_info = await self.client.call_tool(
            "filesystem",
            {"action": "info", "path": "sample.txt"},
        )
        self.assertFalse(res_info.is_error)
        self.assertIn("archivo", res_info.text.lower())


class TestStdioMCPClientWindows(unittest.IsolatedAsyncioTestCase):
    """Tests stdio subprocess transport with an external Python MCP server on Windows 11."""

    async def asyncSetUp(self):
        mock_server_path = Path(__file__).resolve().parent / "mock_stdio_server.py"
        self.client = MCPClient.from_stdio(
            command=["python", "-u", str(mock_server_path)]
        )
        await self.client.initialize()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_stdio_handshake_and_list_tools(self):
        self.assertTrue(self.client.initialized)
        self.assertEqual(self.client.server_info.get("name"), "mock-external-mcp")

        tools = await self.client.list_tools()
        names = [t["name"] for t in tools]
        self.assertIn("echo_tool", names)
        self.assertIn("add_numbers", names)

    async def test_stdio_call_tools(self):
        # 1. Echo tool
        res_echo = await self.client.call_tool("echo_tool", {"message": "Windows 11 Native MCP"})
        self.assertFalse(res_echo.is_error)
        self.assertEqual(res_echo.text, "Echo: Windows 11 Native MCP")

        # 2. Add numbers
        res_add = await self.client.call_tool("add_numbers", {"a": 40, "b": 2})
        self.assertFalse(res_add.is_error)
        self.assertEqual(res_add.text, "Sum: 42")


class TestToolDispatcherIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests ToolDispatcher integrating local InstantToolRegistry with external MCP servers."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = InstantToolRegistry(base_directory=self.temp_dir)
        self.dispatcher = ToolDispatcher(registry=self.registry)

        mock_server_path = Path(__file__).resolve().parent / "mock_stdio_server.py"
        await self.dispatcher.register_stdio_server(
            name="external_mock",
            command=["python", "-u", str(mock_server_path)],
        )

    async def asyncTearDown(self):
        for client in self.dispatcher.mcp_clients.values():
            await client.close()
        await self.dispatcher.local_mcp_client.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tag_detection(self):
        # 1. Kwarg tag
        typ, name, args = self.dispatcher.detect_tool_or_expert(
            'Por favor calcula [TOOL: calculator(expr="3 * 7")]'
        )
        self.assertEqual(typ, "TOOL")
        self.assertEqual(name, "calculator")
        self.assertEqual(args.get("expr"), "3 * 7")

        # 2. Positional tag
        typ, name, args = self.dispatcher.detect_tool_or_expert(
            '[TOOL: calculator("20 + 22")]'
        )
        self.assertEqual(typ, "TOOL")
        self.assertEqual(name, "calculator")
        self.assertEqual(args.get("__positional__"), "20 + 22")

        # 3. JSON dict tag
        typ, name, args = self.dispatcher.detect_tool_or_expert(
            '[TOOL: filesystem({"action": "list", "path": "."})]'
        )
        self.assertEqual(typ, "TOOL")
        self.assertEqual(name, "filesystem")
        self.assertEqual(args.get("action"), "list")

        # 4. Expert tag
        typ, query, _ = self.dispatcher.detect_tool_or_expert(
            '[EXPERT: como calcular deducciones de ganancias]'
        )
        self.assertEqual(typ, "EXPERT")
        self.assertEqual(query, "como calcular deducciones de ganancias")

    async def test_unified_list_tools_and_prompt(self):
        all_tools = await self.dispatcher.list_all_tools()
        tool_names = [t["name"] for t in all_tools]
        
        # Local tools
        self.assertIn("calculator", tool_names)
        self.assertIn("time_date", tool_names)
        self.assertIn("python_eval", tool_names)
        self.assertIn("filesystem", tool_names)

        # External MCP tools
        self.assertIn("echo_tool", tool_names)
        self.assertIn("add_numbers", tool_names)

        # Prompt generation
        prompt = await self.dispatcher.get_tools_prompt()
        self.assertIn("def calculator(", prompt)
        self.assertIn("def echo_tool(", prompt)
        self.assertIn("```python", prompt)

    async def test_execute_tools_local_and_external(self):
        # Local execution (async)
        res_calc = await self.dispatcher.execute_tool_async("calculator", {"expr": "100 / 4"})
        self.assertIn("25", res_calc)

        # External execution (async)
        res_ext = await self.dispatcher.execute_tool_async("echo_tool", {"message": "Test Dispatcher"})
        self.assertEqual(res_ext, "Echo: Test Dispatcher")

        # Synchronous execution wrapper
        res_sync_local = self.dispatcher.execute_tool("calculator", {"expr": "7 * 6"})
        self.assertIn("42", res_sync_local)

        # Positional arg execution
        res_pos = self.dispatcher.execute_tool("calculator", {"__positional__": "8 + 8"})
        self.assertIn("16", res_pos)


def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestOpenBMBSchemaConverter))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestInProcessMCPClient))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestStdioMCPClientWindows))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestToolDispatcherIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
