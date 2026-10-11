"""Tools the agent can call. Two, both offline and deterministic.

`calculator` uses a restricted AST walk rather than `eval`, because the input to a tool is
untrusted text and `eval` on model output is how you get an RCE in a demo.

`formula_lookup` is deliberately a curated table rather than a web search: a study assistant over
a fixed set of notes should cite the notes, and a network call would make the evaluation
irreproducible. Swapping in a real search tool is a one-line change to `TOOLS` (see the README).
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass
class ToolResult:
    name: str
    ok: bool
    text: str
    detail: dict


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> ToolResult:
    """Evaluate a numeric arithmetic expression safely."""
    original = expression.strip()
    expr = expression.strip().replace("^", "**").replace("x", "*").replace("X", "*")
    expr = re.sub(r"\b\u03c0\b", repr(math.pi), expr)
    expr = re.sub(r"\bpi\b", repr(math.pi), expr)
    expr = re.sub(r"\be\b", repr(math.e), expr)
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree.body)
        if value == int(value):
            shown = str(int(value))
        else:
            shown = f"{value:.6g}"
        return ToolResult("calculator", True,
                          f"Using the calculator: {original} = {shown}.",
                          {"expression": original, "value": value})
    except Exception as exc:
        return ToolResult("calculator", False,
                          f"The calculator could not evaluate \u201c{original}\u201d "
                          f"({type(exc).__name__}).",
                          {"expression": original, "error": str(exc)})


# key -> (canonical formula, one-line meaning, source document)
FORMULAS: dict[str, tuple[str, str, str]] = {
    "nyquist": ("fs > 2 * f_max", "sample above twice the highest frequency or alias",
                "01_signals_and_systems"),
    "shannon": ("C = B * log2(1 + S/N)", "channel capacity in bits per second",
                "04_communication_systems"),
    "carson": ("BW = 2 * (delta_f + fm)", "approximate FM bandwidth (Carson's rule)",
               "04_communication_systems"),
    "modulation": ("m = Am / Ac", "AM modulation index; keep 0 < m <= 1 to avoid distortion",
                   "04_communication_systems"),
    "bandwidth": ("BW_AM = 2 * fm", "standard AM bandwidth is twice the message bandwidth",
                  "04_communication_systems"),
    "settling": ("Ts = 4 / (zeta * omega_n)", "2 percent settling time of a second-order system",
                 "03_control_systems"),
    "overshoot": ("Mp = exp(-pi*zeta / sqrt(1 - zeta^2))", "peak overshoot for zeta < 1",
                  "03_control_systems"),
    "damping": ("omega_d = omega_n * sqrt(1 - zeta^2)", "damped natural frequency",
                "03_control_systems"),
    "pid": ("u(t) = Kp*e(t) + Ki*integral e dtau + Kd*de/dt",
            "parallel PID control law", "03_control_systems"),
    "vswr": ("VSWR = (1 + |Gamma|) / (1 - |Gamma|)", "standing-wave ratio from reflection coeff",
             "06_transmission_lines"),
    "reflection": ("Gamma = (ZL - Z0) / (ZL + Z0)", "reflection coefficient of a load mismatch",
                   "06_transmission_lines"),
    "quarter-wave": ("Z0' = sqrt(Z0 * ZL)", "quarter-wave transformer matching impedance",
                     "06_transmission_lines"),
    "zener": ("Vt \u2248 26 mV at 300 K", "thermal voltage; Zener diodes regulate in breakdown",
              "05_semiconductor_devices"),
    "transconductance": ("gm = 2 * Id / (Vgs - Vth)", "MOSFET transconductance in saturation",
                         "05_semiconductor_devices"),
    "cutoff": ("f_c = 1 / (2*pi*R*C)", "cut-off frequency of a first-order RC low-pass filter",
               "03_control_systems"),
    "fft": ("O(N log N)", "fast Fourier transform complexity vs O(N^2) for a direct DFT",
            "08_digital_signal_processing"),
    "dft": ("X[k] = sum x[n] e^(-j*2*pi*k*n/N)", "N-point discrete Fourier transform",
            "08_digital_signal_processing"),
    "resolution": ("delta_f = fs / N", "frequency resolution of an N-point DFT",
                   "08_digital_signal_processing"),
}


def formula_lookup(key: str) -> ToolResult:
    """Look up a canonical ECE formula or constant from the curated table."""
    k = key.strip().lower()
    match = None
    for name in FORMULAS:
        if name in k or k in name:
            match = name
            break
    if match is None:
        return ToolResult("formula_lookup", False,
                          f"No curated formula matches \u201c{key}\u201d.",
                          {"query": key, "available": sorted(FORMULAS)})
    formula, meaning, source = FORMULAS[match]
    return ToolResult("formula_lookup", True,
                      f"{meaning.capitalize()}: `{formula}` [source: {source}]",
                      {"key": match, "formula": formula, "source": source})


TOOLS = {
    "calculator": calculator,
    "formula_lookup": formula_lookup,
}


def run_tool(name: str, tool_input: str | None) -> ToolResult:
    if not name or name not in TOOLS:
        return ToolResult(name or "none", False, f"No such tool: {name!r}.", {})
    if not tool_input:
        return ToolResult(name, False, f"The {name} tool needs an input.", {})
    return TOOLS[name](tool_input)
