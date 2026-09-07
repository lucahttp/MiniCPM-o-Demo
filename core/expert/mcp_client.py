"""
Native Model Context Protocol (MCP) Client and In-Process Server.
Lightweight, asynchronous JSON-RPC 2.0 implementation supporting:
- stdio transport (MCP server subprocesses with Windows 11 compatibility)
- in-process transport (zero-overhead direct execution)
- initialize(), list_tools(), call_tool(name, args)
- JSON Schema to Python function signature converter (OpenBMB Jinja standard)
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"
JSONRPC_VERSION = "2.0"


# ============================================================================
# Exceptions
# ============================================================================

class MCPError(Exception):
    """Base exception for all MCP-related errors."""
    pass


class MCPProtocolError(MCPError):
    """Error returned by JSON-RPC peer or protocol violation."""
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"MCP Error [{code}]: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPTimeoutError(MCPError):
    """Raised when an MCP request times out."""
    pass


class MCPTransportError(MCPError):
    """Raised when communication over the transport fails."""
    pass


class MCPToolError(MCPError):
    """Raised when tool execution encounters an error."""
    pass


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ToolCallResult:
    """Encapsulates the result of calling an MCP tool."""
    content: List[Dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Extracts human-readable text from text content blocks."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            texts: List[str] = []
            for item in self.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(str(item.get("text", "")))
                    elif "text" in item:
                        texts.append(str(item["text"]))
                else:
                    texts.append(str(item))
            return "\n".join(texts).strip()
        return str(self.content)

    def __str__(self) -> str:
        return self.text


@dataclass
class ToolDefinition:
    """Tool metadata and schema matching MCP specification."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def to_python_signature(self) -> str:
        """Converts this tool schema into Python function signature using OpenBMB Jinja standard."""
        return tool_to_python_signature(self.to_dict())


# ============================================================================
# OpenBMB Jinja json_to_python_type Converter
# ============================================================================

def json_to_python_type(schema: Union[str, Dict[str, Any], None]) -> str:
    """
    Implements the OpenBMB Jinja json_to_python_type mapping standard:
    string -> str
    number -> float
    integer -> int
    boolean -> bool
    object -> dict
    array -> list or list[T]
    null -> None
    """
    if schema is None:
        return "Any"
    
    if isinstance(schema, str):
        type_str = schema.lower()
    elif isinstance(schema, dict):
        type_val = schema.get("type")
        if isinstance(type_val, list):
            # Union of types: e.g. ["string", "null"] -> Optional[str]
            non_null = [t for t in type_val if t != "null"]
            has_null = "null" in type_val
            types_mapped = [json_to_python_type(t) for t in non_null]
            base_type = types_mapped[0] if len(types_mapped) == 1 else f"Union[{', '.join(types_mapped)}]"
            return f"Optional[{base_type}]" if has_null else base_type
        if type_val is None:
            if "enum" in schema:
                return "str"
            return "Any"
        type_str = str(type_val).lower()
        if type_str == "array":
            items = schema.get("items")
            if items:
                item_type = json_to_python_type(items)
                return f"list[{item_type}]"
            return "list"
    else:
        return "Any"

    type_mapping = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "dict",
        "array": "list",
        "null": "None",
        "any": "Any",
    }
    return type_mapping.get(type_str, "Any")


def tool_to_python_signature(tool: Dict[str, Any]) -> str:
    """
    Converts an MCP tool definition (name, description, inputSchema) into a clean,
    executable-looking Python function signature with type annotations and docstring,
    matching OpenBMB MiniCPM Jinja formatting for system prompt injection.
    """
    name = tool.get("name", "unnamed_tool")
    desc = tool.get("description", "").strip() or "No description provided."
    schema = tool.get("inputSchema") or tool.get("parameters") or {}
    properties = schema.get("properties", {})
    required_keys = set(schema.get("required", []))

    # Build parameter list: required first, then optional
    required_params: List[str] = []
    optional_params: List[str] = []
    param_docs: List[str] = []

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            param_schema = {}
        
        py_type = json_to_python_type(param_schema)
        param_desc = param_schema.get("description", "").strip()
        default_val = param_schema.get("default")

        is_required = param_name in required_keys

        if is_required:
            required_params.append(f"{param_name}: {py_type}")
            doc_type = py_type
        else:
            if default_val is not None:
                if isinstance(default_val, str):
                    default_repr = f'"{default_val}"'
                else:
                    default_repr = repr(default_val)
                optional_params.append(f"{param_name}: {py_type} = {default_repr}")
                doc_type = f"{py_type}, optional"
            else:
                optional_params.append(f"{param_name}: Optional[{py_type}] = None")
                doc_type = f"{py_type}, optional"

        doc_entry = f"        {param_name} ({doc_type}): {param_desc}".rstrip()
        if not is_required and default_val is not None:
            doc_entry += f" (default: {repr(default_val)})"
        param_docs.append(doc_entry)

    all_params = required_params + optional_params
    params_str = ", ".join(all_params)

    docstring_lines = [f'    """{desc}']
    if param_docs:
        docstring_lines.append("")
        docstring_lines.append("    Args:")
        docstring_lines.extend(param_docs)
    docstring_lines.append('    """')
    docstring_str = "\n".join(docstring_lines)

    return f"def {name}({params_str}) -> dict:\n{docstring_str}\n    pass"


def format_tools_for_system_prompt(tools: List[Dict[str, Any]], intro: Optional[str] = None) -> str:
    """
    Formats a list of MCP tools into the OpenBMB Jinja Python style system prompt block.
    """
    if not tools:
        return ""
    
    header = intro or (
        "You have access to the following tools. "
        "Call them using standard tool tags [TOOL: tool_name(param=value)] when needed:\n"
    )
    code_blocks = [tool_to_python_signature(t) for t in tools]
    return f"{header}```python\n" + "\n\n".join(code_blocks) + "\n```"


# ============================================================================
# Transports
# ============================================================================

class BaseMCPTransport(ABC):
    """Abstract base class for MCP communication transports."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize and open the transport."""
        pass

    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON-RPC message dictionary."""
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Wait for and receive the next incoming JSON-RPC message dictionary."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is actively connected."""
        pass


class InProcessMCPServer:
    """
    High-performance in-process MCP server.
    Allows registering Python functions as MCP tools without subprocess overhead.
    """
    def __init__(self, name: str = "inprocess-mcp-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.initialized = False

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any]
    ) -> None:
        """Register a tool callable with its MCP schema definition."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }
        logger.debug(f"[InProcessMCPServer] Registered tool: {name}")

    def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry."""
        self.tools.pop(name, None)

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC 2.0 message."""
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        # Handle notifications (no id)
        if msg_id is None:
            if method in ("notifications/initialized", "initialized"):
                self.initialized = True
            return None

        # Handle RPC requests (with id)
        if method == "initialize":
            self.initialized = True
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": msg_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }
            }
        elif method == "tools/list":
            tools_list = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }
                for t in self.tools.values()
            ]
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": msg_id,
                "result": {"tools": tools_list}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if tool_name not in self.tools:
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found",
                    }
                }
            
            tool_entry = self.tools[tool_name]
            handler = tool_entry["handler"]
            try:
                if asyncio.iscoroutinefunction(handler):
                    res = await handler(**tool_args)
                else:
                    res = handler(**tool_args)
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": str(res)}],
                        "isError": False,
                    }
                }
            except Exception as e:
                logger.exception(f"[InProcessMCPServer] Error in tool {tool_name}: {e}")
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error in {tool_name}: {str(e)}"}],
                        "isError": True,
                    }
                }
        elif method == "ping":
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": msg_id,
                "result": {}
            }
        else:
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found",
                }
            }


class InProcessMCPTransport(BaseMCPTransport):
    """Transport connecting client to an InProcessMCPServer directly via asyncio queues."""

    def __init__(self, server: InProcessMCPServer):
        self.server = server
        self._incoming_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._connected = False

    async def start(self) -> None:
        self._connected = True

    async def send_message(self, message: Dict[str, Any]) -> None:
        if not self._connected:
            raise MCPTransportError("InProcess transport is not connected")
        response = await self.server.handle_message(message)
        if response is not None:
            await self._incoming_queue.put(response)

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        if not self._connected:
            return None
        try:
            return await self._incoming_queue.get()
        except asyncio.CancelledError:
            return None

    async def close(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


class StdioMCPTransport(BaseMCPTransport):
    """
    Subprocess stdio transport for external MCP servers.
    Fully compatible with Windows 11 paths, newline decoding, and graceful termination.
    """

    def __init__(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.command = command
        self.env = env
        self.cwd = cwd
        self.process: Optional[asyncio.subprocess.Process] = None
        self._incoming_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = True

    @property
    def is_connected(self) -> bool:
        return not self._closed and self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if not self.command:
            raise MCPTransportError("No command provided for StdioMCPTransport")

        # Resolve command path for Windows 11 compatibility
        cmd = list(self.command)
        first_cmd = cmd[0]

        # Use current Python interpreter when 'python' or 'python3' is passed
        if first_cmd.lower() in ("python", "python3", "py"):
            cmd[0] = sys.executable
        else:
            resolved = shutil.which(first_cmd)
            if resolved:
                cmd[0] = resolved

        # Setup environment
        spawn_env = os.environ.copy()
        spawn_env["PYTHONUNBUFFERED"] = "1"
        if self.env:
            spawn_env.update(self.env)

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=spawn_env,
            )
            self._closed = False
            self._read_task = asyncio.create_task(self._stdout_reader())
            self._stderr_task = asyncio.create_task(self._stderr_drainer())
            logger.info(f"[StdioMCPTransport] Spawned server process PID {self.process.pid}: {cmd}")
        except Exception as e:
            self._closed = True
            raise MCPTransportError(f"Failed to spawn MCP stdio process {cmd}: {e}") from e

    async def _stdout_reader(self) -> None:
        """Reads newline-delimited JSON messages from server stdout."""
        try:
            while not self._closed and self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    msg = json.loads(line_str)
                    await self._incoming_queue.put(msg)
                except json.JSONDecodeError:
                    logger.debug(f"[StdioMCPTransport stdout PID {self.process.pid}]: {line_str}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._closed:
                logger.error(f"[StdioMCPTransport] Error reading stdout: {e}")
        finally:
            if not self._closed:
                # Put a sentinel None to wake up any awaiting receivers
                await self._incoming_queue.put(None)

    async def _stderr_drainer(self) -> None:
        """Drains stderr to prevent buffer deadlocks on Windows."""
        try:
            while not self._closed and self.process and self.process.stderr:
                line = await self.process.stderr.readline()
                if not line:
                    break
                err_text = line.decode("utf-8", errors="replace").strip()
                if err_text:
                    logger.debug(f"[StdioMCPTransport stderr PID {self.process.pid}]: {err_text}")
        except (asyncio.CancelledError, Exception):
            pass

    async def send_message(self, message: Dict[str, Any]) -> None:
        if not self.is_connected or not self.process or not self.process.stdin:
            raise MCPTransportError("Transport is not connected")
        payload = json.dumps(message) + "\n"
        try:
            self.process.stdin.write(payload.encode("utf-8"))
            await self.process.stdin.drain()
        except Exception as e:
            raise MCPTransportError(f"Failed to send message over stdio: {e}") from e

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        if self._closed and self._incoming_queue.empty():
            return None
        try:
            item = await self._incoming_queue.get()
            return item
        except asyncio.CancelledError:
            return None

    async def close(self) -> None:
        self._closed = True
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()

        if self.process:
            try:
                if self.process.stdin and not self.process.stdin.is_closing():
                    self.process.stdin.close()
            except Exception:
                pass

            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError, Exception):
                try:
                    self.process.kill()
                    await self.process.wait()
                except Exception:
                    pass
        logger.info("[StdioMCPTransport] Closed.")


# ============================================================================
# MCP Client
# ============================================================================

class MCPClient:
    """
    Lightweight, asynchronous Model Context Protocol (MCP) JSON-RPC 2.0 client.
    Supports in-process and stdio subprocess transports.
    """

    def __init__(
        self,
        transport: BaseMCPTransport,
        client_name: str = "MiniCPM-o-MCPClient",
        client_version: str = "1.0.0",
    ):
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.initialized = False
        
        self._next_id = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._closed = True

    @classmethod
    def from_in_process(cls, server: InProcessMCPServer) -> MCPClient:
        """Construct an MCPClient wired to an InProcessMCPServer."""
        transport = InProcessMCPTransport(server)
        return cls(transport)

    @classmethod
    def from_stdio(
        cls,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> MCPClient:
        """Construct an MCPClient wired to an external subprocess via stdio."""
        transport = StdioMCPTransport(command=command, env=env, cwd=cwd)
        return cls(transport)

    async def connect(self) -> None:
        """Starts the transport and message dispatch loop."""
        if not self._closed:
            return
        await self.transport.start()
        self._closed = False
        self._reader_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        """Continuously reads incoming JSON-RPC messages and resolves pending futures."""
        try:
            while not self._closed:
                msg = await self.transport.receive_message()
                if msg is None:
                    break
                
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending_requests:
                    future = self._pending_requests.pop(msg_id)
                    if not future.done():
                        future.set_result(msg)
                else:
                    # Log notifications or unsolicited responses
                    logger.debug(f"[MCPClient] Received notification/unhandled: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._closed:
                logger.error(f"[MCPClient] Error in listen loop: {e}")
        finally:
            # Cancel any pending requests
            for fut in list(self._pending_requests.values()):
                if not fut.done():
                    fut.set_exception(MCPTransportError("Connection closed"))
            self._pending_requests.clear()

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Any:
        """Sends a JSON-RPC request and awaits the result."""
        if self._closed:
            raise MCPTransportError("Client is not connected. Call connect() first.")

        req_id = self._next_id
        self._next_id += 1

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        request_msg = {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        try:
            await self.transport.send_message(request_msg)
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise MCPTimeoutError(f"Request '{method}' (id={req_id}) timed out after {timeout}s")
        except Exception:
            self._pending_requests.pop(req_id, None)
            raise

        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(
                code=err.get("code", -32000),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )

        return response.get("result", {})

    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Sends a JSON-RPC notification (no id, fire-and-forget)."""
        if self._closed:
            raise MCPTransportError("Client is not connected. Call connect() first.")
        notification_msg = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params or {},
        }
        await self.transport.send_message(notification_msg)

    async def initialize(self) -> Dict[str, Any]:
        """
        Executes the standard MCP initialization sequence:
        1. Sends 'initialize' request
        2. Receives capabilities and serverInfo
        3. Sends 'notifications/initialized'
        """
        await self.connect()
        init_params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
            },
            "clientInfo": {
                "name": self.client_name,
                "version": self.client_version,
            },
        }
        res = await self.send_request("initialize", init_params)
        self.server_info = res.get("serverInfo", {})
        self.capabilities = res.get("capabilities", {})
        
        # Notify server that client initialization is complete
        await self.send_notification("notifications/initialized", {})
        self.initialized = True
        logger.info(f"[MCPClient] Initialized with server: {self.server_info.get('name', 'Unknown')}")
        return res

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Queries the server for its available tools via 'tools/list'."""
        if not self.initialized:
            await self.initialize()
        res = await self.send_request("tools/list", {})
        return res.get("tools", [])

    async def call_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> ToolCallResult:
        """
        Calls an MCP tool via 'tools/call'.
        Returns a ToolCallResult object containing content, text, and status.
        """
        if not self.initialized:
            await self.initialize()
        
        call_params = {
            "name": name,
            "arguments": args or {},
        }
        raw_result = await self.send_request("tools/call", call_params, timeout=timeout)
        content = raw_result.get("content", [])
        is_error = raw_result.get("isError", False)
        return ToolCallResult(content=content, is_error=is_error, raw=raw_result)

    async def close(self) -> None:
        """Closes the client connection and releases transport resources."""
        if self._closed:
            return
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        await self.transport.close()

    async def __aenter__(self) -> MCPClient:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
