import ast
import asyncio
import datetime
import inspect
import json
import logging
import math
import operator
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .mcp_client import (
    BaseMCPTransport,
    InProcessMCPServer,
    MCPClient,
    format_tools_for_system_prompt,
    tool_to_python_signature,
)

logger = logging.getLogger(__name__)


class InstantToolRegistry:
    """
    Standard instant tool registry providing safe, low-latency execution for:
    - Calculator (AST-based safe evaluation with math functions)
    - TimeDate (local time and date formatting)
    - PythonEval (sandboxed Python evaluation)
    - Filesystem (MCP standard filesystem tool: read, write, list, info)
    """

    def __init__(self, base_directory: Optional[Union[str, Path]] = None):
        self.base_directory = Path(base_directory).resolve() if base_directory else Path.cwd().resolve()
        self._allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Mod: operator.mod,
        }
        self._allowed_functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
        }

    def _eval_ast(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self._allowed_operators[type(node.op)](
                self._eval_ast(node.left), self._eval_ast(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self._allowed_operators[type(node.op)](self._eval_ast(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self._allowed_functions:
                func = self._allowed_functions[node.func.id]
                args = [self._eval_ast(arg) for arg in node.args]
                return func(*args)
            raise ValueError(
                f"Function {node.func.id if isinstance(node.func, ast.Name) else 'unknown'} not allowed"
            )
        elif isinstance(node, ast.Name):
            if node.id in self._allowed_functions:
                return self._allowed_functions[node.id]
            raise ValueError(f"Name {node.id} not allowed")
        raise ValueError(f"Unsupported operation: {type(node)}")

    # ------------------------------------------------------------------------
    # Tool 1: Calculator
    # ------------------------------------------------------------------------
    def calculator(self, expr: str) -> str:
        """Evaluación segura con ast (soporta sin, cos, sqrt, log, potencias, aritmética básica)."""
        try:
            clean_expr = str(expr).strip().replace("^", "**")
            tree = ast.parse(clean_expr, mode="eval").body
            result = self._eval_ast(tree)
            return f"El resultado es {result}."
        except Exception as e:
            return f"Lo siento, hubo un error al calcular: {str(e)}."

    # ------------------------------------------------------------------------
    # Tool 2: TimeDate
    # ------------------------------------------------------------------------
    def time_date(self, location: str = "") -> str:
        """Hora y fecha actual exacta."""
        now = datetime.datetime.now()
        meses = [
            "",
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ]
        loc_str = f" en {location.strip()}" if location and str(location).strip() else ""
        date_str = f"{now.day} de {meses[now.month]}"
        time_str = now.strftime("%H:%M")
        return f"Actualmente{loc_str} es el {date_str} y la hora es {time_str}."

    # ------------------------------------------------------------------------
    # Tool 3: PythonEval
    # ------------------------------------------------------------------------
    def python_eval(self, code: str) -> str:
        """Ejecución de cálculos rápidos en entorno seguro."""
        try:
            env = {"math": math, "datetime": datetime}
            clean_code = str(code).strip()
            exec(f"result = {clean_code}", env)
            val = env.get("result", "nada")
            return f"El resultado del código es {val}."
        except Exception:
            try:
                local_env: Dict[str, Any] = {}
                exec(code, env, local_env)
                if local_env:
                    last_val = list(local_env.values())[-1]
                    return f"La ejecución finalizó con el valor {last_val}."
                return "La ejecución fue exitosa."
            except Exception as e2:
                return f"Tuve un problema ejecutando el código: {str(e2)}."

    # ------------------------------------------------------------------------
    # Tool 4: Filesystem (Standard MCP Filesystem tool)
    # ------------------------------------------------------------------------
    def filesystem(self, action: str, path: str, content: str = "") -> str:
        """
        Herramienta de sistema de archivos MCP estándar.
        Operaciones disponibles: 'read', 'write', 'list', 'info'.
        """
        try:
            target_path = Path(path)
            if not target_path.is_absolute():
                target_path = (self.base_directory / target_path).resolve()

            action_lower = str(action).lower().strip()

            if action_lower in ("read", "read_file"):
                if not target_path.exists():
                    return f"Error: El archivo '{path}' no existe."
                if not target_path.is_file():
                    return f"Error: '{path}' es un directorio, no un archivo."
                text = target_path.read_text(encoding="utf-8", errors="replace")
                # Return content with length guard for speech
                if len(text) > 4000:
                    preview = text[:4000]
                    return f"Contenido de {target_path.name} (primeros 4000 caracteres):\n{preview}..."
                return text

            elif action_lower in ("write", "write_file"):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return f"Archivo '{target_path.name}' escrito exitosamente ({len(content)} caracteres)."

            elif action_lower in ("list", "list_directory", "ls"):
                if not target_path.exists():
                    return f"Error: El directorio '{path}' no existe."
                if not target_path.is_dir():
                    return f"Error: '{path}' es un archivo, no un directorio."
                entries = []
                for item in sorted(target_path.iterdir()):
                    kind = "[DIR]" if item.is_dir() else "[FILE]"
                    entries.append(f"{kind} {item.name}")
                items_str = ", ".join(entries) if entries else "Directorio vacío"
                return f"Contenido de {target_path.name}: {items_str}."

            elif action_lower in ("info", "stat"):
                if not target_path.exists():
                    return f"Error: La ruta '{path}' no existe."
                stat = target_path.stat()
                kind = "directorio" if target_path.is_dir() else "archivo"
                size = f"{stat.st_size} bytes"
                mod_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                return f"'{target_path.name}' es un {kind}, tamaño: {size}, modificado: {mod_time}."

            else:
                return f"Acción desconocida '{action}'. Las acciones permitidas son: 'read', 'write', 'list', 'info'."

        except Exception as e:
            return f"Error en operación de sistema de archivos: {str(e)}."

    # Convenience shortcuts for filesystem
    def read_file(self, path: str) -> str:
        """Lee el contenido de un archivo."""
        return self.filesystem(action="read", path=path)

    def write_file(self, path: str, content: str) -> str:
        """Escribe contenido en un archivo."""
        return self.filesystem(action="write", path=path, content=content)

    def list_directory(self, path: str = ".") -> str:
        """Lista los archivos y carpetas de un directorio."""
        return self.filesystem(action="list", path=path)

    # ------------------------------------------------------------------------
    # MCP Server Factory
    # ------------------------------------------------------------------------
    def get_mcp_server(self, name: str = "InstantToolRegistry") -> InProcessMCPServer:
        """
        Exposes this registry as a standard MCP InProcessMCPServer with JSON Schema definitions.
        """
        server = InProcessMCPServer(name=name, version="1.0.0")

        # 1. Calculator
        server.register_tool(
            name="calculator",
            description="Evaluación matemática segura con soporte para aritmética, potencias (^), raíces (sqrt), trigonometría (sin, cos, tan) y logaritmos.",
            input_schema={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Expresión matemática a evaluar, p. ej. '2 + 2', 'sqrt(144)', 'sin(pi/2)'",
                    }
                },
                "required": ["expr"],
            },
            handler=self.calculator,
        )

        # 2. TimeDate
        server.register_tool(
            name="time_date",
            description="Obtiene la hora y fecha actual exacta en lenguaje natural.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Ubicación opcional (p. ej. 'Madrid', 'Buenos Aires')",
                        "default": "",
                    }
                },
                "required": [],
            },
            handler=self.time_date,
        )

        # 3. PythonEval
        server.register_tool(
            name="python_eval",
            description="Ejecuta cálculos y scripts de Python rápidos en un entorno seguro.",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Código o cálculo de Python a ejecutar",
                    }
                },
                "required": ["code"],
            },
            handler=self.python_eval,
        )

        # 4. Filesystem
        server.register_tool(
            name="filesystem",
            description="Operaciones estándar de sistema de archivos: leer ('read'), escribir ('write'), listar ('list') y consultar metadatos ('info').",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Operación a ejecutar ('read', 'write', 'list', 'info')",
                        "enum": ["read", "write", "list", "info"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Ruta del archivo o carpeta objetivo",
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenido a escribir cuando la acción es 'write'",
                        "default": "",
                    },
                },
                "required": ["action", "path"],
            },
            handler=self.filesystem,
        )

        return server


class ToolDispatcher:
    """
    Manages local tools and dynamic external MCP server connections.
    Supports tool detection in model outputs, synchronous/asynchronous execution,
    and OpenBMB Jinja prompt generation.
    """

    def __init__(self, registry: Optional[InstantToolRegistry] = None):
        self.registry = registry or InstantToolRegistry()
        self.tool_tag_regex = re.compile(r"\[TOOL:\s*([a-zA-Z0-9_]+)\((.*?)\)\s*\]", re.IGNORECASE)
        self.expert_tag_regex = re.compile(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", re.IGNORECASE)

        # Local in-process MCP server and client
        self.in_process_server: InProcessMCPServer = self.registry.get_mcp_server()
        self.local_mcp_client: MCPClient = MCPClient.from_in_process(self.in_process_server)

        # External registered MCP clients: name -> MCPClient
        self.mcp_clients: Dict[str, MCPClient] = {}

    # ------------------------------------------------------------------------
    # Dynamic External MCP Server Registration
    # ------------------------------------------------------------------------
    async def register_mcp_client(self, name: str, client: MCPClient) -> None:
        """Registra e inicializa dinámicamente un cliente MCP externo."""
        await client.initialize()
        self.mcp_clients[name] = client
        logger.info(f"[ToolDispatcher] Registered external MCP client '{name}'")

    async def register_stdio_server(
        self,
        name: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> MCPClient:
        """
        Lanza e inicializa un servidor MCP externo como subproceso stdio (compatible con Windows 11).
        """
        client = MCPClient.from_stdio(command=command, env=env, cwd=cwd)
        await self.register_mcp_client(name, client)
        return client

    def register_tool_function(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Registra una función Python personalizada como herramienta en el servidor local."""
        self.in_process_server.register_tool(name, description, input_schema, handler)
        setattr(self.registry, name, handler)

    # ------------------------------------------------------------------------
    # Tool Discovery and OpenBMB Prompt Generation
    # ------------------------------------------------------------------------
    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """
        Retorna la lista unificada de herramientas disponibles en el registro local
        y en todos los servidores MCP externos conectados.
        """
        all_tools: List[Dict[str, Any]] = []
        # 1. Local tools
        local_tools = await self.local_mcp_client.list_tools()
        all_tools.extend(local_tools)

        # 2. External MCP tools
        for client_name, client in self.mcp_clients.items():
            try:
                ext_tools = await client.list_tools()
                all_tools.extend(ext_tools)
            except Exception as e:
                logger.error(f"[ToolDispatcher] Error listing tools from '{client_name}': {e}")

        return all_tools

    async def get_tools_prompt(self, intro: Optional[str] = None) -> str:
        """
        Genera el bloque de herramientas para el prompt del sistema siguiendo
        el estándar Jinja OpenBMB (firmas de funciones Python con tipos y docstrings).
        """
        tools = await self.list_all_tools()
        return format_tools_for_system_prompt(tools, intro=intro)

    def get_tools_system_prompt_sync(self, intro: Optional[str] = None) -> str:
        """
        Versión sincrónica para generar el prompt con las herramientas locales registradas.
        """
        tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self.in_process_server.tools.values()
        ]
        return format_tools_for_system_prompt(tools, intro=intro)

    # ------------------------------------------------------------------------
    # Tag Detection & Argument Parsing
    # ------------------------------------------------------------------------
    def detect_tool_or_expert(self, text: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detecta si hay una herramienta rápida o un experto en el texto.
        Retorna (tipo, nombre_herramienta_o_query, argumentos).
        Tipo puede ser "TOOL", "EXPERT" o "NONE".
        """
        if not text:
            return "NONE", None, None

        # Support OpenBMB native tokens, MiniCPM 5.0 XML, and legacy tags via fc2dict
        try:
            from .parser import fc2dict
            _, calls = fc2dict(text)
            if calls:
                first_call = calls[0]
                call_name = first_call.get("name", "")
                if call_name in ("expert", "delegate"):
                    return "EXPERT", first_call.get("arguments", {}).get("query", ""), None
                return "TOOL", call_name, first_call.get("arguments", {})
        except Exception as e:
            logger.debug(f"[ToolDispatcher] fc2dict parser fallback: {e}")

        tool_match = self.tool_tag_regex.search(text)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            args_str = tool_match.group(2).strip()
            args: Dict[str, Any] = {}

            if args_str:
                # 1. Check if args_str is a JSON object: {"key": "val"}
                if args_str.startswith("{") and args_str.endswith("}"):
                    try:
                        parsed = json.loads(args_str)
                        if isinstance(parsed, dict):
                            return "TOOL", tool_name, parsed
                    except Exception:
                        pass

                # 2. Parse key=value pairs (supports quotes, numbers, booleans)
                kwarg_pattern = re.compile(
                    r'([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,]+))'
                )
                matches = kwarg_pattern.findall(args_str)
                if matches:
                    for k, v_double, v_single, v_raw in matches:
                        v = v_double if v_double != "" else (v_single if v_single != "" else v_raw)
                        # Try to cast numeric or boolean values
                        if v.lower() == "true":
                            args[k] = True
                        elif v.lower() == "false":
                            args[k] = False
                        else:
                            try:
                                if "." in v:
                                    args[k] = float(v)
                                else:
                                    args[k] = int(v)
                            except ValueError:
                                args[k] = v

                # 3. Positional string fallback
                if not args and args_str:
                    clean_pos = args_str.strip()
                    if (clean_pos.startswith('"') and clean_pos.endswith('"')) or (
                        clean_pos.startswith("'") and clean_pos.endswith("'")
                    ):
                        clean_pos = clean_pos[1:-1]
                    args = {"__positional__": clean_pos}

            return "TOOL", tool_name, args

        expert_match = self.expert_tag_regex.search(text)
        if expert_match:
            query = expert_match.group(1).strip()
            return "EXPERT", query, None

        return "NONE", None, None

    # ------------------------------------------------------------------------
    # Tool Execution (Async & Sync)
    # ------------------------------------------------------------------------
    async def execute_tool_async(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Ejecuta la herramienta de forma asíncrona, resolviendo tanto herramientas
        del registro local como herramientas de clientes MCP externos conectados.
        """
        clean_args = dict(args)
        pos_arg = clean_args.pop("__positional__", None)

        # 1. Local tool execution
        if hasattr(self.registry, tool_name):
            func = getattr(self.registry, tool_name)
            try:
                if pos_arg is not None:
                    # If function expects named parameters, adapt first parameter
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if params and len(params) == 1:
                        res = func(pos_arg)
                    else:
                        first_param = params[0] if params else "expr"
                        clean_args[first_param] = pos_arg
                        res = func(**clean_args)
                else:
                    res = func(**clean_args)

                if asyncio.iscoroutine(res):
                    res = await res
                return str(res).strip()
            except TypeError as e:
                logger.error(f"[ToolDispatcher] Argument error executing local tool {tool_name}: {e}")
                return f"Hubo un error con los argumentos de la herramienta {tool_name}."
            except Exception as e:
                logger.error(f"[ToolDispatcher] Error executing local tool {tool_name}: {e}")
                return f"Error ejecutando {tool_name}: {str(e)}."

        # 2. External MCP client execution
        if pos_arg is not None:
            clean_args["query"] = pos_arg

        for client_name, client in self.mcp_clients.items():
            try:
                tools = await client.list_tools()
                tool_names = [t.get("name") for t in tools]
                if tool_name in tool_names:
                    result = await client.call_tool(tool_name, clean_args)
                    return result.text.strip()
            except Exception as e:
                logger.error(f"[ToolDispatcher] Error executing '{tool_name}' on '{client_name}': {e}")

        # 3. Fallback: Delegate to expert
        return f"[EXPERT: Utiliza tu conocimiento para resolver esto usando la herramienta {tool_name} con argumentos {args}]"

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Ejecución sincrónica con compatibilidad total para llamadas existentes.
        Formatea la respuesta en 1 oración limpia lista para MiniCPM-o.
        """
        # If it's a local tool and not a coroutine, execute immediately
        if hasattr(self.registry, tool_name):
            func = getattr(self.registry, tool_name)
            if not asyncio.iscoroutinefunction(func):
                clean_args = dict(args)
                pos_arg = clean_args.pop("__positional__", None)
                try:
                    if pos_arg is not None:
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        if params and len(params) == 1:
                            response = func(pos_arg)
                        else:
                            first_param = params[0] if params else "expr"
                            clean_args[first_param] = pos_arg
                            response = func(**clean_args)
                    else:
                        response = func(**clean_args)
                    return str(response).strip()
                except TypeError:
                    return f"Hubo un error con los argumentos de la herramienta {tool_name}."
                except Exception as e:
                    return f"Error ejecutando {tool_name}: {str(e)}."

        # For external MCP tools or async tools, execute safely via event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Run in new thread with isolated loop if already in an active event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self.execute_tool_async(tool_name, args)))
                return future.result()
        else:
            return asyncio.run(self.execute_tool_async(tool_name, args))
