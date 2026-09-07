"""
MiniCPM Native Token and AST Parser Engine
Supports:
1. OpenBMB native function-calling syntax:
   - Thought delimiters: <|thought_start|> ... <|thought_end|>
   - Tool call delimiters: <|tool_call_start|> ... <|tool_call_end|> with ```python ... ``` blocks
   - Deterministic and safe AST parsing without eval()
   - Reversion of Python reserved keywords (from_ -> from, in_ -> in, class_ -> class, etc.)
   - Primary function: fc2dict(text) -> (thought_text, list_of_tool_calls)
2. MiniCPM 5.0 XML function-calling syntax:
   - <function=name><param=key>val</param></function>
3. Backward compatibility with legacy tags:
   - [TOOL:func(args)]
   - [EXPERT: query] / [DELEGATE: query]
"""

import ast
import json
import keyword
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Delimiters
THOUGHT_START = "<|thought_start|>"
THOUGHT_END = "<|thought_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

# Regular expressions
_THOUGHT_REGEX = re.compile(r"<\|thought_start\|>(.*?)<\|thought_end\|>", re.DOTALL | re.IGNORECASE)
_XML_THOUGHT_REGEX = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_REGEX = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL | re.IGNORECASE)

_XML_FUNCTION_REGEX = re.compile(r"<function=[\"']?([a-zA-Z0-9_.-]+)[\"']?\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_REGEX = re.compile(r"<param=[\"']?([a-zA-Z0-9_.-]+)[\"']?\s*>(.*?)</param>", re.DOTALL | re.IGNORECASE)

_LEGACY_TOOL_REGEX = re.compile(r"\[TOOL:\s*([a-zA-Z0-9_.-]+)\((.*?)\)\s*\]", re.DOTALL | re.IGNORECASE)
_LEGACY_EXPERT_REGEX = re.compile(r"\[(?:DELEGATE|EXPERT):\s*(.*?)\]", re.DOTALL | re.IGNORECASE)

# Python reserved keywords and common builtins that require trailing underscore in Python kwargs
_PYTHON_KEYWORDS = set(keyword.kwlist)
_PYTHON_BUILTINS = {
    "type", "id", "format", "input", "filter", "map", "list", "dict", "set",
    "str", "int", "float", "object", "open", "range", "dir", "help", "all",
    "any", "bin", "hex", "oct", "max", "min", "sum", "pow", "round", "vars",
    "zip", "hash", "schema", "fields", "json"
}
RESERVED_PYTHON_KEYWORDS = _PYTHON_KEYWORDS | _PYTHON_BUILTINS


def revert_reserved_keyword(name: str) -> str:
    """
    Reverts Python reserved keywords (e.g., 'from_' -> 'from', 'in_' -> 'in', 'class_' -> 'class').
    Preserves private attributes (starts with '__') and identifiers that are not reserved.
    """
    if not isinstance(name, str):
        return name
    if name.endswith("_") and not name.startswith("__"):
        base = name[:-1]
        if base in RESERVED_PYTHON_KEYWORDS or keyword.iskeyword(base):
            return base
    return name


def _resolve_ast_value(node: ast.AST) -> Any:
    """
    Deterministically and safely resolves an AST expression node to a Python object.
    Supports ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.UnaryOp, ast.BinOp, ast.Name.
    Safe AST evaluation without dynamic code execution.
    """
    # 1. Constant (handles numbers, strings, True, False, None, etc.)
    if isinstance(node, ast.Constant):
        return node.value

    # 2. Unary operations (e.g. -5, +3.14, not True, ~0)
    if isinstance(node, ast.UnaryOp):
        operand_val = _resolve_ast_value(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand_val
        elif isinstance(node.op, ast.UAdd):
            return +operand_val
        elif isinstance(node.op, ast.Not):
            return not operand_val
        elif isinstance(node.op, ast.Invert):
            return ~operand_val
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    # 3. Binary operations (safe arithmetic e.g. 2 + 3, 10 * 5)
    if isinstance(node, ast.BinOp):
        left_val = _resolve_ast_value(node.left)
        right_val = _resolve_ast_value(node.right)
        if isinstance(node.op, ast.Add):
            return left_val + right_val
        elif isinstance(node.op, ast.Sub):
            return left_val - right_val
        elif isinstance(node.op, ast.Mult):
            return left_val * right_val
        elif isinstance(node.op, ast.Div):
            return left_val / right_val
        elif isinstance(node.op, ast.FloorDiv):
            return left_val // right_val
        elif isinstance(node.op, ast.Mod):
            return left_val % right_val
        elif isinstance(node.op, ast.Pow):
            return left_val ** right_val
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    # 4. List literals: [1, 2, 3]
    if isinstance(node, ast.List):
        return [_resolve_ast_value(elt) for elt in node.elts]

    # 5. Tuple literals: (1, 2, 3)
    if isinstance(node, ast.Tuple):
        return tuple(_resolve_ast_value(elt) for elt in node.elts)

    # 6. Set literals: {1, 2, 3}
    if isinstance(node, ast.Set):
        return {_resolve_ast_value(elt) for elt in node.elts}

    # 7. Dict literals: {'a': 1, 'b': 2}
    if isinstance(node, ast.Dict):
        result_dict = {}
        for k_node, v_node in zip(node.keys, node.values):
            if k_node is None:
                # Unpacking **kwargs
                sub_dict = _resolve_ast_value(v_node)
                if isinstance(sub_dict, dict):
                    for sk, sv in sub_dict.items():
                        reverted_sk = revert_reserved_keyword(sk) if isinstance(sk, str) else sk
                        result_dict[reverted_sk] = sv
            else:
                k = _resolve_ast_value(k_node)
                if isinstance(k, str):
                    k = revert_reserved_keyword(k)
                v = _resolve_ast_value(v_node)
                result_dict[k] = v
        return result_dict

    # 8. Name identifiers: True, False, None, or unquoted strings/identifiers
    if isinstance(node, ast.Name):
        name_id = node.id
        if name_id.lower() in ("true",):
            return True
        elif name_id.lower() in ("false",):
            return False
        elif name_id.lower() in ("none", "null"):
            return None
        return name_id

    # 9. Expressions wrapped in ast.Expr
    if isinstance(node, ast.Expr):
        return _resolve_ast_value(node.value)

    # 10. Subscript or call values if nested
    if isinstance(node, ast.Call):
        # Nested call resolution
        resolved = resolve_ast_call(node)
        return resolved

    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def resolve_ast_call(call_node: Union[ast.Call, ast.Expr, str]) -> Dict[str, Any]:
    """
    Extracts tool call details (function name and resolved arguments) from an AST Call node or code string.
    Evaluates ast.Constant, ast.List, ast.Dict, ast.UnaryOp, ast.Name safely without eval().
    Automatically reverts reserved Python keywords (e.g., from_ -> from).

    Returns:
        {
            "name": <str>,
            "arguments": <dict>
        }
    """
    if isinstance(call_node, str):
        cleaned_str = call_node.strip()
        try:
            parsed = ast.parse(cleaned_str, mode="eval" if "\n" not in cleaned_str else "exec")
        except SyntaxError:
            # Fallback for multi-line or single expression
            parsed = ast.parse(cleaned_str, mode="exec")

        if isinstance(parsed, ast.Expression):
            node = parsed.body
        elif isinstance(parsed, ast.Module):
            if not parsed.body:
                raise ValueError("Empty code string passed to resolve_ast_call")
            stmt = parsed.body[0]
            if isinstance(stmt, ast.Expr):
                node = stmt.value
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                node = stmt.value
            else:
                node = stmt
        else:
            node = parsed
    elif isinstance(call_node, ast.Expr):
        node = call_node.value
    else:
        node = call_node

    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected ast.Call node, got {type(node).__name__}")

    # Resolve function name
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        curr = node.func
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
        func_name = ".".join(reversed(parts))
    else:
        func_name = ast.unparse(node.func) if hasattr(ast, "unparse") else str(node.func)

    arguments: Dict[str, Any] = {}

    # Resolve positional arguments
    if node.args:
        pos_values = [_resolve_ast_value(arg) for arg in node.args]
        arguments["__args__"] = pos_values
        if len(pos_values) == 1:
            arguments["__positional__"] = pos_values[0]
            # If the single positional argument is a dict (e.g. tool({"action": "list"})),
            # merge keys into arguments for direct access
            if isinstance(pos_values[0], dict):
                for dk, dv in pos_values[0].items():
                    if dk not in arguments:
                        arguments[dk] = dv
        for idx, val in enumerate(pos_values):
            arguments[f"arg_{idx}"] = val

    # Resolve keyword arguments
    for kw in node.keywords:
        if kw.arg is None:
            # Unpacking **kwargs
            unpacked = _resolve_ast_value(kw.value)
            if isinstance(unpacked, dict):
                for k, v in unpacked.items():
                    reverted_k = revert_reserved_keyword(str(k))
                    arguments[reverted_k] = v
        else:
            reverted_k = revert_reserved_keyword(kw.arg)
            arguments[reverted_k] = _resolve_ast_value(kw.value)

    return {
        "name": func_name,
        "arguments": arguments,
    }


def _parse_literal_value(val_str: str) -> Any:
    """
    Parses a string value into its native Python representation safely.
    Tries booleans, null, JSON, and Python AST literals before falling back to raw string.
    """
    val_str = val_str.strip()
    if not val_str:
        return ""

    lower_val = val_str.lower()
    if lower_val == "true":
        return True
    if lower_val == "false":
        return False
    if lower_val in ("null", "none"):
        return None

    # Try JSON parsing
    try:
        return json.loads(val_str)
    except Exception:
        pass

    # Try AST evaluation (handles Python literals: -10, [1, 2], {'a': 'b'}, etc.)
    try:
        tree = ast.parse(val_str, mode="eval").body
        return _resolve_ast_value(tree)
    except Exception:
        pass

    # Fallback to string without surrounding quotes if present
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    return val_str


def extract_thought(text: str) -> Optional[str]:
    """
    Extracts thought content delimited by <|thought_start|> ... <|thought_end|>
    or fallback <thought> ... </thought>.
    """
    if not text:
        return None

    matches = _THOUGHT_REGEX.findall(text)
    if matches:
        combined = "\n".join(m.strip() for m in matches if m.strip())
        return combined if combined else ""

    xml_matches = _XML_THOUGHT_REGEX.findall(text)
    if xml_matches:
        combined = "\n".join(m.strip() for m in xml_matches if m.strip())
        return combined if combined else ""

    return None


def _extract_code_from_block(block: str) -> str:
    """Extracts raw python code from markdown ```python ... ``` or returns block."""
    block = block.strip()
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", block, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return block


def parse_openbmb_native(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Parses OpenBMB native function-calling format:
    Delimiters:
      - <|thought_start|> ... <|thought_end|>
      - <|tool_call_start|> ```python ... ``` <|tool_call_end|>
    """
    thought = extract_thought(text)
    tool_calls: List[Dict[str, Any]] = []

    for block_match in _TOOL_CALL_BLOCK_REGEX.finditer(text):
        raw_block = block_match.group(1)
        code = _extract_code_from_block(raw_block)
        if not code:
            continue

        try:
            tree = ast.parse(code, mode="exec")
            for stmt in tree.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    tool_calls.append(resolve_ast_call(stmt.value))
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    tool_calls.append(resolve_ast_call(stmt.value))
        except SyntaxError:
            # Fallback: try parsing line-by-line
            for line in code.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    call_dict = resolve_ast_call(line)
                    tool_calls.append(call_dict)
                except Exception:
                    logger.debug("Failed to parse line as AST call: %s", line)

    return thought, tool_calls


def parse_minicpm_xml(text: str) -> List[Dict[str, Any]]:
    """
    Parses MiniCPM 5.0 XML format:
    <function=name><param=key>val</param></function>
    """
    tool_calls: List[Dict[str, Any]] = []

    for match in _XML_FUNCTION_REGEX.finditer(text):
        func_name = match.group(1).strip()
        body = match.group(2)
        params: Dict[str, Any] = {}

        for p_match in _XML_PARAM_REGEX.finditer(body):
            raw_key = p_match.group(1).strip()
            key = revert_reserved_keyword(raw_key)
            raw_val = p_match.group(2).strip()
            params[key] = _parse_literal_value(raw_val)

        tool_calls.append({
            "name": func_name,
            "arguments": params,
        })

    return tool_calls


def _extract_balanced_tool_tags(text: str) -> List[Tuple[str, str]]:
    """
    Extracts function name and argument string from [TOOL:func(args)]
    accounting for nested parentheses and quotes.
    """
    results = []
    pattern = re.compile(r"\[TOOL:\s*([a-zA-Z0-9_.-]+)\(", re.IGNORECASE)
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        func_name = match.group(1).strip()
        start_idx = match.end()

        # Find matching closing paren before ']'
        paren_depth = 1
        in_quote: Optional[str] = None
        escape = False
        curr = start_idx
        args_str = None

        while curr < len(text):
            ch = text[curr]
            if escape:
                escape = False
                curr += 1
                continue
            if ch == "\\":
                escape = True
                curr += 1
                continue

            if in_quote:
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in ('"', "'"):
                    in_quote = ch
                elif ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        # Check if followed by optional whitespace and ']'
                        rest = text[curr + 1:]
                        close_bracket = rest.find("]")
                        if close_bracket != -1 and rest[:close_bracket].strip() == "":
                            args_str = text[start_idx:curr]
                            pos = curr + 1 + close_bracket + 1
                            break
            curr += 1

        if args_str is not None:
            results.append((func_name, args_str.strip()))
        else:
            pos = match.end()

    return results


def parse_legacy_tags(text: str) -> List[Dict[str, Any]]:
    """
    Parses legacy [TOOL:func(args)] and [EXPERT: query] / [DELEGATE: query] tags.
    """
    tool_calls: List[Dict[str, Any]] = []

    # 1. Parse [TOOL:func(args)]
    extracted_tools = _extract_balanced_tool_tags(text)
    for func_name, args_str in extracted_tools:
        call_expr = f"{func_name}({args_str})" if args_str else f"{func_name}()"
        try:
            resolved = resolve_ast_call(call_expr)
            tool_calls.append(resolved)
        except Exception:
            # Fallback regex kwarg parser
            kwargs = {}
            kwarg_matches = re.findall(r'(\w+)\s*=\s*(["\'])(.*?)\2', args_str)
            for k, _, v in kwarg_matches:
                kwargs[revert_reserved_keyword(k)] = _parse_literal_value(v)

            if not kwargs and args_str:
                val = args_str.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                kwargs = {"__positional__": _parse_literal_value(val)}

            tool_calls.append({
                "name": func_name,
                "arguments": kwargs,
            })

    # 2. Parse [EXPERT: query] / [DELEGATE: query]
    for match in _LEGACY_EXPERT_REGEX.finditer(text):
        query = match.group(1).strip()
        tool_calls.append({
            "name": "expert",
            "arguments": {"query": query},
        })

    return tool_calls


def fc2dict(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Main parser function for MiniCPM.
    Converts model outputs into a tuple of (thought_text, list_of_tool_calls).

    Supports:
    1. OpenBMB native syntax (<|thought_start|>, <|tool_call_start|>```python ... ```<|tool_call_end|>)
    2. MiniCPM 5.0 XML syntax (<function=name><param=key>val</param></function>)
    3. Legacy tags ([TOOL:func(args)], [EXPERT: query])
    """
    if not text or not isinstance(text, str):
        return None, []

    # Extract thought
    thought_text = extract_thought(text)

    all_tool_calls: List[Dict[str, Any]] = []

    # 1. Native OpenBMB syntax
    if TOOL_CALL_START in text:
        _, native_calls = parse_openbmb_native(text)
        all_tool_calls.extend(native_calls)

    # 2. XML syntax (<function=...>)
    if "<function=" in text.lower():
        xml_calls = parse_minicpm_xml(text)
        all_tool_calls.extend(xml_calls)

    # 3. Legacy tags ([TOOL:...] / [EXPERT:...])
    if "[tool:" in text.lower() or "[expert:" in text.lower() or "[delegate:" in text.lower():
        legacy_calls = parse_legacy_tags(text)
        all_tool_calls.extend(legacy_calls)

    return thought_text, all_tool_calls


class MiniCPMParser:
    """Class wrapper for MiniCPM parsing engine."""

    @staticmethod
    def parse(text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        return fc2dict(text)

    @staticmethod
    def resolve_ast(call_node: Union[ast.Call, ast.Expr, str]) -> Dict[str, Any]:
        return resolve_ast_call(call_node)

    @staticmethod
    def revert_keyword(name: str) -> str:
        return revert_reserved_keyword(name)
