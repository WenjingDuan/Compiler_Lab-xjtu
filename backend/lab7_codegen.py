#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验7：源程序 -> 词法 -> 语法/语义 -> 四元式 -> 内存布局 -> 目标汇编（Web API 内存运行）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

_LAB7_DIR = Path(__file__).resolve().parent.parent.parent / "实验7"
if str(_LAB7_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB7_DIR))

from lexer import format_lex_process, format_token_stream, tokenize  # noqa: E402
from memory_layout import (  # noqa: E402
    build_memory_layout,
    format_memory_layout_text,
    format_symbol_table_with_addr,
)
from rd_parser import Parser, format_parse_steps  # noqa: E402
from slr_codegen import (  # noqa: E402
    ast_to_dot_source,
    build_slr1_tables,
    format_quadruples_text,
    format_semantic_errors_text,
    format_slr_table_text,
    load_grammar_with_actions,
)
from target_codegen import generate_all_targets  # noqa: E402

DEFAULT_GRAMMAR_PATH = _LAB7_DIR / "grammar_input.txt"


def _load_default_grammar_text() -> str:
    if DEFAULT_GRAMMAR_PATH.is_file():
        return DEFAULT_GRAMMAR_PATH.read_text(encoding="utf-8")
    return ""


def run_compile(source_code: str) -> Dict[str, object]:
    """对源程序执行完整编译流水线，返回各阶段文本结果。"""
    src = source_code or ""
    if not src.strip():
        return {"ok": False, "errors": ["请输入源程序。"]}

    gtext = _load_default_grammar_text()
    if not gtext:
        return {"ok": False, "errors": ["未找到实验7文法文件 grammar_input.txt。"]}

    grammar_path = DEFAULT_GRAMMAR_PATH if DEFAULT_GRAMMAR_PATH.is_file() else Path("（网页输入）")

    tokens, lex_errors, lex_process = tokenize(src)
    lexical_process = format_lex_process(lex_process, lex_errors)
    token_stream = format_token_stream(tokens)

    empty_targets = {"target_mips": "(无)\n", "target_x86": "(无)\n", "target_arm": "(无)\n"}

    if lex_errors:
        return {
            "ok": True,
            "success": False,
            "lexical_process": lexical_process,
            "token_stream": token_stream,
            "parse_trace": "(词法错误，未进行语法分析)\n",
            "ast_tree": "(无)\n",
            "ast_dot": "digraph AST {}\n",
            "symbol_table": "(无)\n",
            "memory_layout": "(无)\n",
            "semantic_errors": "\n".join(lex_errors) + "\n",
            "quadruples": "(无四元式)\n",
            "slr1_table": "",
            "slr_conflicts": "",
            **empty_targets,
            "report": _format_report(
                src, grammar_path, len(tokens), 0, 0, token_stream, lexical_process,
                "(无)\n", "(无)\n", "(无)\n", "(无)\n", "\n".join(lex_errors) + "\n",
                "(无四元式)\n", empty_targets,
            ),
            "grammar_warnings": [],
        }

    gerr: List[str] = []
    slr1_table = ""
    slr_conflicts = "(无)\n"
    grammar_warnings: List[str] = []
    base_prods, _, gerr = load_grammar_with_actions(DEFAULT_GRAMMAR_PATH)

    grammar_warnings = list(gerr)
    if base_prods:
        productions, action_cols, _, goto_nts, flat_action, goto_rows, conflicts = build_slr1_tables(base_prods)
        slr1_table = format_slr_table_text(productions, action_cols, goto_nts, flat_action, goto_rows, conflicts)
        slr_conflicts = "\n".join(conflicts) + "\n" if conflicts else "(无)\n"

    parser = Parser(tokens)
    root = parser.parse()
    ctx = parser.ctx
    parse_trace = format_parse_steps(parser.steps)
    ast_tree = (root.to_text() if root else "(无)\n") + "\n"
    semantic_errors = format_semantic_errors_text(ctx)
    quadruples = format_quadruples_text(ctx)
    ast_dot = ast_to_dot_source(root)

    layout = build_memory_layout(ctx)
    memory_layout = format_memory_layout_text(layout)
    symbol_table = format_symbol_table_with_addr(ctx, layout)

    targets = empty_targets.copy()
    if root and not ctx.errors:
        for isa, asm in generate_all_targets(ctx, layout).items():
            targets[f"target_{isa}"] = asm

    report = _format_report(
        src,
        grammar_path,
        len(tokens),
        len(parser.steps),
        len(ctx.quads),
        token_stream,
        lexical_process,
        parse_trace,
        ast_tree,
        symbol_table,
        memory_layout,
        semantic_errors,
        quadruples,
        targets,
    )

    return {
        "ok": True,
        "success": root is not None and not ctx.errors and not lex_errors,
        "lexical_process": lexical_process,
        "token_stream": token_stream,
        "parse_trace": parse_trace,
        "ast_tree": ast_tree,
        "ast_dot": ast_dot,
        "symbol_table": symbol_table,
        "memory_layout": memory_layout,
        "semantic_errors": semantic_errors,
        "quadruples": quadruples,
        "slr1_table": slr1_table,
        "slr_conflicts": slr_conflicts,
        **targets,
        "report": report,
        "grammar_warnings": grammar_warnings,
    }


def _format_report(
    source: str,
    grammar_path: Path,
    n_tokens: int,
    n_steps: int,
    n_quads: int,
    token_stream: str,
    lexical_process: str,
    parse_trace: str,
    ast_tree: str,
    symbol_table: str,
    memory_layout: str,
    semantic_errors: str,
    quadruples: str,
    targets: Dict[str, str],
) -> str:
    parts = [
        "实验7 编译器 — 汇总报告\n",
        f"源程序长度: {len(source)} 字符\n",
        f"文法参考: {grammar_path}\n",
        f"词法 Token 数: {n_tokens}\n",
        f"语法展开步数: {n_steps}\n",
        f"四元式条数: {n_quads}\n",
        "\n===== 词法分析过程 =====\n",
        lexical_process,
        "\n===== Token 流 =====\n",
        token_stream,
        "\n===== 语法分析过程 =====\n",
        parse_trace,
        "\n===== 内存布局 =====\n",
        memory_layout,
        "\n===== 符号表（含地址） =====\n",
        symbol_table,
        "\n===== 四元式 =====\n",
        quadruples,
        "\n===== AST =====\n",
        ast_tree,
        "\n===== 错误 =====\n",
        semantic_errors,
    ]
    if any(v.strip() and v != "(无)\n" for v in targets.values()):
        parts.append("\n===== 目标汇编 =====\n")
        for key in ("target_mips", "target_x86", "target_arm"):
            if targets.get(key, "(无)\n").strip() and targets[key] != "(无)\n":
                label = key.replace("target_", "").upper()
                parts.append(f"\n--- {label} ({key}.s) ---\n")
                parts.append(targets[key])
    return "".join(parts)
