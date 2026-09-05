import ast
import math
import re
import datetime
import operator
import time
from typing import Tuple, Optional, Any, Dict

class InstantToolRegistry:
    def __init__(self):
        self._allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Mod: operator.mod
        }
        self._allowed_functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'abs': abs,
            'pi': math.pi,
            'e': math.e
        }
        
    def _eval_ast(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Num):  # <python3.8
            return node.n
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self._allowed_operators[type(node.op)](self._eval_ast(node.left), self._eval_ast(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self._allowed_operators[type(node.op)](self._eval_ast(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self._allowed_functions:
                func = self._allowed_functions[node.func.id]
                args = [self._eval_ast(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Function {node.func.id if isinstance(node.func, ast.Name) else 'unknown'} not allowed")
        elif isinstance(node, ast.Name):
            if node.id in self._allowed_functions:
                return self._allowed_functions[node.id]
            raise ValueError(f"Name {node.id} not allowed")
        raise ValueError(f"Unsupported operation: {type(node)}")

    def calculator(self, expr: str) -> str:
        """Evaluación segura con ast (soporta sin, cos, sqrt, log, potencias, aritmética básica)."""
        try:
            expr = expr.replace('^', '**')
            tree = ast.parse(expr, mode='eval').body
            result = self._eval_ast(tree)
            # Formatea como 1 oración limpia
            return f"El resultado es {result}."
        except Exception as e:
            return f"Lo siento, hubo un error al calcular: {str(e)}."

    def time_date(self, location: str = "") -> str:
        """Hora y fecha actual exacta."""
        now = datetime.datetime.now()
        # En español local, podemos formatear usando nombres predeterminados o localizados si están disponibles
        meses = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        loc_str = f" en {location}" if location and location.strip() else ""
        date_str = f"{now.day} de {meses[now.month]}"
        time_str = now.strftime("%H:%M")
        return f"Actualmente{loc_str} es el {date_str} y la hora es {time_str}."

    def python_eval(self, code: str) -> str:
        """Ejecución de cálculos rápidos en entorno seguro."""
        try:
            env = {'math': math, 'datetime': datetime}
            exec(f"result = {code}", env)
            val = env.get('result', 'nada')
            return f"El resultado del código es {val}."
        except Exception as e:
            try:
                local_env = {}
                exec(code, env, local_env)
                if local_env:
                    last_val = list(local_env.values())[-1]
                    return f"La ejecución finalizó con el valor {last_val}."
                return "La ejecución fue exitosa."
            except Exception as e2:
                return f"Tuve un problema ejecutando el código: {str(e2)}."


class ToolDispatcher:
    def __init__(self):
        self.registry = InstantToolRegistry()
        self.tool_tag_regex = re.compile(r"\[TOOL:\s*([a-zA-Z0-9_]+)\((.*?)\)\s*\]", re.IGNORECASE)
        self.expert_tag_regex = re.compile(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", re.IGNORECASE)

    def detect_tool_or_expert(self, text: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detecta si hay una herramienta rápida o un experto.
        Retorna (tipo, nombre_herramienta_o_query, argumentos).
        Tipo puede ser "TOOL", "EXPERT" o "NONE".
        """
        tool_match = self.tool_tag_regex.search(text)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            args_str = tool_match.group(2).strip()
            args = {}
            if args_str:
                # Parse kwarg-like string, e.g., expr="2+2", location="Madrid"
                kwarg_matches = re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', args_str)
                for k, _, v in kwarg_matches:
                    args[k] = v
                
                # If we couldn't parse kwargs, we assume it's a positional string
                if not args and args_str:
                    # Strip quotes if present
                    if (args_str.startswith('"') and args_str.endswith('"')) or (args_str.startswith("'") and args_str.endswith("'")):
                        args_str = args_str[1:-1]
                    args = {'__positional__': args_str}

            return "TOOL", tool_name, args

        expert_match = self.expert_tag_regex.search(text)
        if expert_match:
            query = expert_match.group(1).strip()
            return "EXPERT", query, None
            
        return "NONE", None, None

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Ejecuta y formatea la respuesta en 1 oración limpia lista para que MiniCPM-o la hable de inmediato.
        """
        if not hasattr(self.registry, tool_name):
            # Delegar al experto si no es una herramienta local o requiere razonamiento profundo
            return f"[EXPERT: Utiliza tu conocimiento para resolver esto usando la herramienta {tool_name} con argumentos {args}]"

        func = getattr(self.registry, tool_name)
        
        pos_arg = args.pop('__positional__', None)
        try:
            if pos_arg is not None:
                response = func(pos_arg)
            else:
                response = func(**args)
            return response.strip()
        except TypeError:
            return f"Hubo un error con los argumentos de la herramienta {tool_name}."
