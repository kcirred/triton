"""Shared helpers for pattern-docs generation and round-trip dumping.

Exported:
  clean_ir(text)                    — strip loc//#loc noise, add section breaks
  extract_tagged_tests(module)      — yield PatternEntry for @pattern-decorated tests
  PatternEntry                      — dataclass for a single tagged test
  extract_mlir_from_test(fn)        — pull the MLIR string literal from a test body
  split_docstring(doc)              — (title, body) from a docstring
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType


# ---------------------------------------------------------------------------
# IR cleaning — strip loc(...) / #loc noise, add section-break blank lines.
# Shared between round-trip dumps and pattern docs (MLIR snippets).
# ---------------------------------------------------------------------------

# Match ``loc(...)`` including one level of nesting (e.g. ``loc("x"(#loc))``).
_LOC_CALL = re.compile(r"\s*loc\((?:[^()]|\([^()]*\))*\)")

# Ops that mark a new logical section; a blank line is inserted before the
# first occurrence of each in a run.
_SECTION_BREAK_OPS = (
    "tt.get_program_id",
    "tt.make_tensor_descriptor",
    "ktdp.construct_memory_view",
    "ktdp.construct_access_tile",
    "ktdp.construct_indirect_access_tile",
    "ktdp.get_compute_tile_id",
    "scf.for",
    "tt.return",
    "func.return",
)


def _match_section_op(line: str) -> str | None:
    stripped = line.lstrip()
    op_part = re.sub(r"^%\S+\s*=\s*", "", stripped)
    for op in _SECTION_BREAK_OPS:
        if op_part.startswith(op):
            return op
    return None


def _add_section_breaks(lines: list[str]) -> str:
    out: list[str] = []
    last_section_op: str | None = None
    for line in lines:
        op = _match_section_op(line)
        if op is not None and op != last_section_op:
            if out and out[-1].strip() and not out[-1].rstrip().endswith("{"):
                out.append("")
        if line.strip():
            last_section_op = op
        out.append(line)
    return "\n".join(out)


def clean_ir(text: str) -> str:
    """Strip ``loc(...)`` / ``#loc`` noise and add section-break blank lines."""
    text = _LOC_CALL.sub("", text)
    lines = [l for l in text.split("\n") if not l.strip().startswith("#loc")]
    return _add_section_breaks(lines)


# ---------------------------------------------------------------------------
# Docstring splitting — (title, body) from a raw docstring.
# ---------------------------------------------------------------------------

def split_docstring(doc: str | None) -> tuple[str | None, str]:
    """Return ``(title, body)`` from a docstring.

    ``title`` is the first sentence (period stripped).
    ``body``  is the remainder, paragraphs preserved.
    Returns ``(None, "")`` when ``doc`` is falsy.
    """
    if not doc:
        return None, ""
    text = doc.strip()
    first_para, _, rest = text.partition("\n\n")
    first_para_one_line = " ".join(first_para.split())
    if ". " in first_para_one_line:
        first_sentence, remainder = first_para_one_line.split(". ", 1)
        body_prefix = remainder.strip()
    else:
        first_sentence = first_para_one_line
        body_prefix = ""
    title = first_sentence.rstrip(".").strip()
    body_parts: list[str] = []
    if body_prefix:
        body_parts.append(body_prefix)
    if rest.strip():
        body_parts.append(rest.strip())
    return title, "\n\n".join(body_parts)


# ---------------------------------------------------------------------------
# MLIR snippet extraction — pull the string literal passed to self.run(...)
# from a test function's AST.  Also extracts assert_stderr substrings for
# negative entries.
# ---------------------------------------------------------------------------

def _find_self_run_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "run"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)):
            return sub
    return None


def _find_self_run_in_setup(cls_node: ast.ClassDef) -> ast.Call | None:
    """Return the self.run() call from setup_method, if present."""
    for fn in cls_node.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "setup_method":
            call = _find_self_run_call(fn)
            if call:
                return call
    return None


def _find_stderr_substrings(func_node: ast.FunctionDef, after_lineno: int) -> list[str]:
    out: list[str] = []
    for sub in ast.walk(func_node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "assert_stderr"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"
                and getattr(sub, "lineno", 0) >= after_lineno):
            for arg in sub.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.append(arg.value)
    return out


def _is_pytest_raises_with(item: ast.AST) -> bool:
    if not isinstance(item, ast.With):
        return False
    for w in item.items:
        call = w.context_expr
        if (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "raises"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "pytest"):
            return True
    return False


def extract_mlir_from_test(fn) -> str | None:
    """Return the MLIR string passed to ``self.run(...)`` in ``fn``.

    Handles two shapes:
    - Inline: ``self.run("<mlir>")`` directly in the test body.
    - Setup-based: the test's class has ``setup_method`` that calls
      ``self.run(...)``.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    # Try inline self.run() first.
    call = _find_self_run_call(tree)
    if call:
        return textwrap.dedent(call.args[0].value).strip("\n")

    # Fall back to setup_method on the enclosing class (resolved at harvest time).
    return None


def extract_mlir_from_setup(source_file: Path, class_name: str) -> str | None:
    """Return the MLIR from ``setup_method`` of ``class_name`` in ``source_file``."""
    try:
        tree = ast.parse(source_file.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            call = _find_self_run_in_setup(node)
            if call:
                return textwrap.dedent(call.args[0].value).strip("\n")
    return None


def extract_stderr_substrings_from_test(fn) -> list[str]:
    """Return assert_stderr substrings from a negative test function."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return []
    for func_node in ast.walk(tree):
        if not isinstance(func_node, ast.FunctionDef):
            continue
        for stmt in func_node.body:
            if not _is_pytest_raises_with(stmt):
                continue
            with_end = getattr(stmt, "end_lineno", stmt.lineno)
            return _find_stderr_substrings(func_node, with_end)
    return []


# ---------------------------------------------------------------------------
# Tagged test harvesting — walk a module for @pattern-decorated functions.
# ---------------------------------------------------------------------------

@dataclass
class PatternEntry:
    tag: str
    category: str
    negative: bool
    fn_name: str
    class_name: str
    source_file: Path
    lineno: int
    docstring: str | None
    example: str | None            # Triton Python snippet from decorator
    mlir: str | None               # MLIR snippet (may come from setup_method)
    stderr_substrings: list[str] = field(default_factory=list)


def extract_tagged_tests(module: ModuleType) -> list[PatternEntry]:
    """Walk ``module`` and return one ``PatternEntry`` per ``@pattern``-decorated test.

    Handles both inline ``self.run()`` and ``setup_method``-based MLIR.
    """
    source_file = Path(inspect.getfile(module))
    entries: list[PatternEntry] = []

    for cls_name, cls_obj in inspect.getmembers(module, inspect.isclass):
        for fn_name, fn_obj in inspect.getmembers(cls_obj, inspect.isfunction):
            meta = getattr(fn_obj, "_pattern", None)
            if meta is None:
                continue

            lineno = getattr(fn_obj, "__code__", None)
            lineno = lineno.co_firstlineno if lineno else 0

            mlir = extract_mlir_from_test(fn_obj)
            if mlir is None:
                # Try the class's setup_method.
                mlir = extract_mlir_from_setup(source_file, cls_name)

            stderr_subs: list[str] = []
            if meta["negative"]:
                stderr_subs = extract_stderr_substrings_from_test(fn_obj)

            entries.append(PatternEntry(
                tag=meta["tag"],
                category=meta["category"],
                negative=meta["negative"],
                fn_name=fn_name,
                class_name=cls_name,
                source_file=source_file,
                lineno=lineno,
                docstring=inspect.getdoc(fn_obj),
                example=meta.get("example"),
                mlir=mlir,
                stderr_substrings=stderr_subs,
            ))

    return entries
