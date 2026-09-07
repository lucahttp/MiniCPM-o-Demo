"""
Unit tests for MiniCPM Native Token and AST Parser Engine
Validates:
- Simple and multiple tool calls
- Complex arguments (lists, dicts, negative numbers, floats, booleans)
- Native OpenBMB syntax with thoughts (<|thought_start|>, <|tool_call_start|>)
- Python reserved keyword reversion (from_, in_, class_, etc.)
- MiniCPM 5.0 XML syntax (<function=name><param=key>val</param></function>)
- Backward compatibility with legacy tags ([TOOL:func(args)], [EXPERT: query])
- Safe AST evaluation without eval()
"""

import ast
import os
import sys
import unittest

# Ensure MiniCPM-o-Demo directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.expert.parser import (
    fc2dict,
    resolve_ast_call,
    revert_reserved_keyword,
    parse_openbmb_native,
    parse_minicpm_xml,
    parse_legacy_tags,
    extract_thought,
    MiniCPMParser,
)


class TestMiniCPMParser(unittest.TestCase):

    def test_openbmb_native_single_call(self):
        sample = (
            "<|thought_start|>\n"
            "Necesito verificar el clima en Buenos Aires para los próximos 3 días.\n"
            "<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "get_weather(city='Buenos Aires', days=3)\n"
            "```\n"
            "<|tool_call_end|>"
        )
        thought, calls = fc2dict(sample)

        self.assertEqual(thought, "Necesito verificar el clima en Buenos Aires para los próximos 3 días.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Buenos Aires", "days": 3})

    def test_openbmb_native_multiple_calls_single_block(self):
        sample = (
            "<|thought_start|>\n"
            "Consultando dos herramientas en paralelo.\n"
            "<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "get_weather(city='Madrid')\n"
            "calculator(expr='25 * 4')\n"
            "```\n"
            "<|tool_call_end|>"
        )
        thought, calls = fc2dict(sample)

        self.assertEqual(thought, "Consultando dos herramientas en paralelo.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Madrid"})
        self.assertEqual(calls[1]["name"], "calculator")
        self.assertEqual(calls[1]["arguments"], {"expr": "25 * 4"})

    def test_openbmb_native_multiple_blocks(self):
        sample = (
            "<|thought_start|>Paso 1<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "fetch_user(id_=101)\n"
            "```\n"
            "<|tool_call_end|>\n"
            "Texto intermedio explicativo\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "notify_user(user_id=101, message='Hola!')\n"
            "```\n"
            "<|tool_call_end|>"
        )
        thought, calls = fc2dict(sample)

        self.assertEqual(thought, "Paso 1")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "fetch_user")
        self.assertEqual(calls[0]["arguments"], {"id": 101})  # id_ -> id
        self.assertEqual(calls[1]["name"], "notify_user")
        self.assertEqual(calls[1]["arguments"], {"user_id": 101, "message": "Hola!"})

    def test_complex_arguments_and_negative_numbers(self):
        sample = (
            "<|thought_start|>Cálculo de impuestos con valores negativos y listas complejas.<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "calculate_tax(\n"
            "    gross_income=150000,\n"
            "    deductions=[-5000, -2500.75],\n"
            "    coordinates=[-34.6037, -58.3816],\n"
            "    tax_brackets={'low': 0.10, 'mid': 0.25, 'exempt': True, 'min_loss': -1200},\n"
            "    flags=[True, False, None],\n"
            "    class_='Tier1',\n"
            "    from_='2026-01-01'\n"
            ")\n"
            "```\n"
            "<|tool_call_end|>"
        )
        thought, calls = fc2dict(sample)

        self.assertEqual(len(calls), 1)
        args = calls[0]["arguments"]
        self.assertEqual(args["gross_income"], 150000)
        self.assertEqual(args["deductions"], [-5000, -2500.75])
        self.assertEqual(args["coordinates"], [-34.6037, -58.3816])
        self.assertEqual(args["tax_brackets"], {"low": 0.10, "mid": 0.25, "exempt": True, "min_loss": -1200})
        self.assertEqual(args["flags"], [True, False, None])
        # Reserved keyword reversion
        self.assertEqual(args["class"], "Tier1")
        self.assertEqual(args["from"], "2026-01-01")

    def test_reserved_keywords_reversion_exhaustive(self):
        self.assertEqual(revert_reserved_keyword("from_"), "from")
        self.assertEqual(revert_reserved_keyword("in_"), "in")
        self.assertEqual(revert_reserved_keyword("class_"), "class")
        self.assertEqual(revert_reserved_keyword("def_"), "def")
        self.assertEqual(revert_reserved_keyword("import_"), "import")
        self.assertEqual(revert_reserved_keyword("type_"), "type")
        self.assertEqual(revert_reserved_keyword("id_"), "id")
        self.assertEqual(revert_reserved_keyword("return_"), "return")
        self.assertEqual(revert_reserved_keyword("pass_"), "pass")
        self.assertEqual(revert_reserved_keyword("for_"), "for")
        self.assertEqual(revert_reserved_keyword("while_"), "while")
        # Non-reserved should not be stripped
        self.assertEqual(revert_reserved_keyword("user_name_"), "user_name_")
        self.assertEqual(revert_reserved_keyword("custom_arg"), "custom_arg")
        self.assertEqual(revert_reserved_keyword("__private__"), "__private__")

    def test_minicpm_xml_syntax_simple(self):
        sample = (
            "<function=get_weather>\n"
            "<param=city>Tokyo</param>\n"
            "<param=days>7</param>\n"
            "</function>"
        )
        thought, calls = fc2dict(sample)

        self.assertIsNone(thought)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Tokyo", "days": 7})

    def test_minicpm_xml_syntax_complex_and_multiple(self):
        sample = (
            "<thought>Analizando vuelos y alojamiento</thought>\n"
            "<function=search_flights>\n"
            "<param=from_>EZE</param>\n"
            "<param=to>MAD</param>\n"
            "<param=prices>[-300, 1200.50]</param>\n"
            "<param=filters>{\"nonstop\": true, \"seats\": 2}</param>\n"
            "<param=class_>business</param>\n"
            "</function>\n"
            "<function=book_hotel>\n"
            "<param=name>Ritz</param>\n"
            "<param=rating>4.8</param>\n"
            "<param=active>true</param>\n"
            "</function>"
        )
        thought, calls = fc2dict(sample)

        self.assertEqual(thought, "Analizando vuelos y alojamiento")
        self.assertEqual(len(calls), 2)

        flight_call = calls[0]
        self.assertEqual(flight_call["name"], "search_flights")
        self.assertEqual(flight_call["arguments"]["from"], "EZE")
        self.assertEqual(flight_call["arguments"]["to"], "MAD")
        self.assertEqual(flight_call["arguments"]["prices"], [-300, 1200.50])
        self.assertEqual(flight_call["arguments"]["filters"], {"nonstop": True, "seats": 2})
        self.assertEqual(flight_call["arguments"]["class"], "business")

        hotel_call = calls[1]
        self.assertEqual(hotel_call["name"], "book_hotel")
        self.assertEqual(hotel_call["arguments"], {"name": "Ritz", "rating": 4.8, "active": True})

    def test_legacy_tool_tag_kwargs(self):
        sample = "Voy a calcular la fórmula: [TOOL:calculator(expr=\"sin(3.14) + 1\")] de inmediato."
        thought, calls = fc2dict(sample)

        self.assertIsNone(thought)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "calculator")
        self.assertEqual(calls[0]["arguments"]["expr"], "sin(3.14) + 1")

    def test_legacy_tool_tag_positional_and_negative(self):
        sample = "Resultado: [TOOL:calculator(-42)] y ahora [TOOL:time_date(\"Buenos Aires\")]"
        thought, calls = fc2dict(sample)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "calculator")
        self.assertEqual(calls[0]["arguments"]["__positional__"], -42)

        self.assertEqual(calls[1]["name"], "time_date")
        self.assertEqual(calls[1]["arguments"]["__positional__"], "Buenos Aires")

    def test_legacy_expert_and_delegate_tags(self):
        sample1 = "Necesito ayuda: [EXPERT: ¿Cómo optimizar un modelo LLM con LoRA?]"
        thought1, calls1 = fc2dict(sample1)

        self.assertEqual(len(calls1), 1)
        self.assertEqual(calls1[0]["name"], "expert")
        self.assertEqual(calls1[0]["arguments"], {"query": "¿Cómo optimizar un modelo LLM con LoRA?"})

        sample2 = "[DELEGATE: Redacta un contrato de alquiler comercial en CABA]"
        thought2, calls2 = fc2dict(sample2)

        self.assertEqual(len(calls2), 1)
        self.assertEqual(calls2[0]["name"], "expert")
        self.assertEqual(calls2[0]["arguments"], {"query": "Redacta un contrato de alquiler comercial en CABA"})

    def test_resolve_ast_call_direct_ast_node(self):
        code = "matrix_op(dim=[3, 3], weights={'w1': -0.5, 'w2': +1.2}, transpose=False)"
        node = ast.parse(code, mode="eval").body
        resolved = resolve_ast_call(node)

        self.assertEqual(resolved["name"], "matrix_op")
        self.assertEqual(resolved["arguments"]["dim"], [3, 3])
        self.assertEqual(resolved["arguments"]["weights"], {"w1": -0.5, "w2": 1.2})
        self.assertFalse(resolved["arguments"]["transpose"])

    def test_resolve_ast_call_unary_and_binary_ops(self):
        # Negative numbers, unary negation, basic safe arithmetic
        resolved = resolve_ast_call("calc(a=-10, b=+20, diff=-5.5, mult=2*8)")
        self.assertEqual(resolved["name"], "calc")
        self.assertEqual(resolved["arguments"]["a"], -10)
        self.assertEqual(resolved["arguments"]["b"], 20)
        self.assertEqual(resolved["arguments"]["diff"], -5.5)
        self.assertEqual(resolved["arguments"]["mult"], 16)

    def test_no_eval_used_and_safety(self):
        # Inspect module AST to verify no eval() or exec() calls exist anywhere in parser.py
        import inspect
        from core.expert import parser

        parser_source = inspect.getsource(parser)
        tree = ast.parse(parser_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("eval", "exec"), f"Unsafe function {node.func.id} used in parser!")

    def test_thought_multiline_and_fallback(self):
        sample = (
            "<|thought_start|>\n"
            "Línea 1 del pensamiento\n"
            "Línea 2 con caracteres especiales: áéíóú, ñ, $100 & %.\n"
            "<|thought_end|>\n"
            "Texto normal sin tool call."
        )
        thought, calls = fc2dict(sample)
        self.assertIn("Línea 1 del pensamiento", thought)
        self.assertIn("caracteres especiales", thought)
        self.assertEqual(len(calls), 0)

    def test_openbmb_raw_code_without_markdown_fences(self):
        sample = (
            "<|thought_start|>Pensando...<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "get_status(server='prod-1', ping_ms=12.4)\n"
            "<|tool_call_end|>"
        )
        thought, calls = fc2dict(sample)
        self.assertEqual(thought, "Pensando...")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_status")
        self.assertEqual(calls[0]["arguments"], {"server": "prod-1", "ping_ms": 12.4})

    def test_attribute_function_call(self):
        code = "agent.tools.calculator(expr='100 / 2')"
        res = resolve_ast_call(code)
        self.assertEqual(res["name"], "agent.tools.calculator")
        self.assertEqual(res["arguments"], {"expr": "100 / 2"})

    def test_empty_and_invalid_inputs_resilience(self):
        self.assertEqual(fc2dict(""), (None, []))
        self.assertEqual(fc2dict(None), (None, []))
        self.assertEqual(fc2dict(12345), (None, []))
        self.assertEqual(fc2dict("Solo texto normal sin tags."), (None, []))

    def test_minicpm_parser_class_interface(self):
        sample = (
            "<|thought_start|>Pensamiento vía clase MiniCPMParser<|thought_end|>\n"
            "<|tool_call_start|>\n"
            "```python\n"
            "ping(host='127.0.0.1', count=3)\n"
            "```\n"
            "<|tool_call_end|>"
        )
        thought, calls = MiniCPMParser.parse(sample)
        self.assertEqual(thought, "Pensamiento vía clase MiniCPMParser")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "ping")
        self.assertEqual(calls[0]["arguments"], {"host": "127.0.0.1", "count": 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
