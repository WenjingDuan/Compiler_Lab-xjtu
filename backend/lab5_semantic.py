#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验5：SLR(1) 引导的语义分析
- 从文法文件读取产生式，支持右部末尾 @动作名
- 与实验四一致：自动增广起点、构造 LR(0) 项目集族与 SLR(1) ACTION/GOTO
- 在移进/归约过程中维护「状态栈 + 文法符号栈 + 属性栈」三列并行结构（与语法栈旁属性栈一致），构建 AST、符号表，做类型检查与重复声明检测
- 可选：移进时对 LBR 自动进入作用域（见 SHIFT_HOOKS）
- 运行结束后将所有结果写入 `--out-dir`（默认 实验5/output）：AST 文本/Graphviz/PNG/符号表/错误/轨迹/Token 序列/SLR 表/并行栈摘要/汇总报告
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

EPS = "\x01__EPS__\x01"


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: List[str]


@dataclass(frozen=True, order=True)
class LR0Item:
    prod_id: int
    dot_pos: int


@dataclass
class State:
    items: Set[LR0Item]


class ActionKind(str, Enum):
    none = "none"
    shift = "shift"
    reduce = "reduce"
    accept = "accept"
    conflict = "conflict"


@dataclass
class ActionCell:
    kind: ActionKind = ActionKind.none
    num: int = -1
    conflict_note: str = ""

    def to_string(self) -> str:
        if self.conflict_note:
            return self.conflict_note
        if self.kind == ActionKind.shift:
            return f"s{self.num}"
        if self.kind == ActionKind.reduce:
            return f"r{self.num}"
        if self.kind == ActionKind.accept:
            return "acc"
        return ""


def _trim(s: str) -> str:
    return s.strip()


def _tokenize_rhs(rhs: str) -> List[str]:
    t = _trim(rhs)
    if not t:
        return []
    if any(c.isspace() for c in t):
        tokens: List[str] = []
        for tok in t.split():
            if tok in {"epsilon", "eps", "@", "#"}:
                continue
            tokens.append(tok)
        return tokens
    tokens = []
    i = 0
    while i < len(t):
        if t[i].isspace():
            i += 1
            continue
        if t[i].isalnum() or t[i] in {"_", "'"}:
            j = i
            while j < len(t) and (t[j].isalnum() or t[j] in {"_", "'"}):
                j += 1
            tok = t[i:j]
            if tok not in {"epsilon", "eps"}:
                tokens.append(tok)
            i = j
        else:
            tokens.append(t[i])
            i += 1
    return tokens


def _closure(kernel: Set[LR0Item], productions: List[Production], prod_of_nt: Dict[str, List[int]]) -> Set[LR0Item]:
    res: Set[LR0Item] = set(kernel)
    q: Deque[LR0Item] = deque(kernel)
    while q:
        cur = q.popleft()
        p = productions[cur.prod_id]
        if cur.dot_pos >= len(p.rhs):
            continue
        sym = p.rhs[cur.dot_pos]
        if sym not in prod_of_nt:
            continue
        for pid in prod_of_nt[sym]:
            nxt = LR0Item(pid, 0)
            if nxt not in res:
                res.add(nxt)
                q.append(nxt)
    return res


def _goto(state_items: Set[LR0Item], symbol: str, productions: List[Production], prod_of_nt: Dict[str, List[int]]) -> Set[LR0Item]:
    moved: Set[LR0Item] = set()
    for it in state_items:
        p = productions[it.prod_id]
        if it.dot_pos < len(p.rhs) and p.rhs[it.dot_pos] == symbol:
            moved.add(LR0Item(it.prod_id, it.dot_pos + 1))
    if not moved:
        return set()
    return _closure(moved, productions, prod_of_nt)


def load_grammar_with_actions_from_text(text: str) -> Tuple[List[Production], List[str], List[str]]:
    """返回 (产生式列表, 每条产生式语义动作名, 原始错误列表)。"""
    errs: List[str] = []
    lines = [_trim(x) for x in text.splitlines()]
    if not lines:
        return [], [], ["文法文件为空"]
    try:
        n = int(lines[0])
    except ValueError:
        return [], [], ["文法文件首行必须是产生式条数(整数)"]
    prods: List[Production] = []
    actions: List[str] = []
    got = 0
    action_re = re.compile(r"^(.+?)\s+@([A-Za-z_][A-Za-z0-9_]*)\s*$")
    for line in lines[1:]:
        if not line or line.startswith("#"):
            continue
        if got >= n:
            break
        if "->" not in line:
            errs.append(f"产生式缺少 -> : {line}")
            continue
        lhs, rhs_all = line.split("->", 1)
        lhs = _trim(lhs)
        rhs_all = _trim(rhs_all)
        if not lhs:
            errs.append(f"左部为空: {line}")
            continue
        branches = [b.strip() for b in rhs_all.split("|")]
        for br in branches:
            if not br:
                continue
            act = "noop"
            rhs_tokens: List[str]
            m = action_re.match(br)
            if m:
                rhs_part, act = m.group(1).strip(), m.group(2)
                rhs_tokens = _tokenize_rhs(rhs_part)
            else:
                rhs_tokens = _tokenize_rhs(br)
            prods.append(Production(lhs=lhs, rhs=rhs_tokens))
            actions.append(act)
        got += 1
    if got != n:
        errs.append(f"声明 {n} 条产生式行，实际解析到 {got} 条")
    return prods, actions, errs


def load_grammar_with_actions(path: Path) -> Tuple[List[Production], List[str], List[str]]:
    return load_grammar_with_actions_from_text(path.read_text(encoding="utf-8"))


def _compute_nullable_first_follow(
    productions: List[Production],
    nonterminals: Set[str],
    terminals: Set[str],
    augmented_start: str,
) -> Tuple[Dict[str, Set[str]], Dict[str, bool], Dict[str, Set[str]]]:
    nullable: Dict[str, bool] = {nt: False for nt in nonterminals}
    changed = True
    while changed:
        changed = False
        for p in productions:
            if not p.rhs:
                if not nullable.get(p.lhs, False):
                    nullable[p.lhs] = True
                    changed = True
                continue
            all_null = True
            for sym in p.rhs:
                if sym in terminals or not nullable.get(sym, False):
                    all_null = False
                    break
            if all_null and not nullable.get(p.lhs, False):
                nullable[p.lhs] = True
                changed = True

    first: Dict[str, Set[str]] = {t: {t} for t in terminals}
    for nt in nonterminals:
        first[nt] = set()

    changed = True
    while changed:
        changed = False
        for p in productions:
            a, rhs = p.lhs, p.rhs
            i = 0
            while i < len(rhs):
                sym = rhs[i]
                if sym in terminals:
                    if sym not in first[a]:
                        first[a].add(sym)
                        changed = True
                    break
                old_sz = len(first[a])
                first[a].update(t for t in first[sym] if t != EPS)
                if len(first[a]) > old_sz:
                    changed = True
                if not nullable.get(sym, False):
                    break
                i += 1
            if i == len(rhs):
                if EPS not in first[a]:
                    first[a].add(EPS)
                    changed = True

    follow: Dict[str, Set[str]] = {nt: set() for nt in nonterminals}
    follow[augmented_start].add("$")

    def first_of_suffix(rhs: List[str], start: int) -> Tuple[Set[str], bool]:
        terms: Set[str] = set()
        if start >= len(rhs):
            return terms, True
        for j in range(start, len(rhs)):
            sym = rhs[j]
            if sym in terminals:
                terms.add(sym)
                return terms, False
            terms.update(t for t in first[sym] if t != EPS)
            if not nullable.get(sym, False):
                return terms, False
        return terms, True

    changed = True
    while changed:
        changed = False
        for p in productions:
            a, rhs = p.lhs, p.rhs
            for i, sym in enumerate(rhs):
                if sym not in nonterminals:
                    continue
                fb, beta_null = first_of_suffix(rhs, i + 1)
                for t in fb:
                    if t not in follow[sym]:
                        follow[sym].add(t)
                        changed = True
                if beta_null:
                    for t in follow[a]:
                        if t not in follow[sym]:
                            follow[sym].add(t)
                            changed = True

    return first, nullable, follow


def _merge_action(cell: ActionCell, kind: ActionKind, num: int, detail: str, conflicts: List[str]) -> None:
    if kind == ActionKind.none:
        return
    if cell.kind == ActionKind.none:
        cell.kind = kind
        cell.num = num
        return
    if cell.kind == kind and cell.num == num:
        return
    a = cell.to_string()
    if kind == ActionKind.shift:
        b = f"s{num}"
    elif kind == ActionKind.reduce:
        b = f"r{num}"
    elif kind == ActionKind.accept:
        b = "acc"
    else:
        b = ""
    conflicts.append(f"ACTION 冲突: {detail} 已有 [{a}] 拟加入 [{b}]")
    cell.kind = ActionKind.conflict
    cell.num = -1
    if not cell.conflict_note:
        cell.conflict_note = f"{a}/{b}"
    else:
        cell.conflict_note += f"/{b}"


def _merge_goto(row: Dict[str, int], nt: str, j: int, conflicts: List[str], detail: str) -> None:
    if nt not in row:
        row[nt] = j
        return
    if row[nt] == j:
        return
    conflicts.append(f"GOTO 冲突: {detail} 已有 {row[nt]} 拟加入 {j}")


def build_slr1_tables(
    input_prods: List[Production],
) -> Tuple[
    List[Production],
    List[str],
    List[State],
    List[str],
    List[ActionCell],
    List[Dict[str, int]],
    List[str],
]:
    """返回 productions(含增广), action_cols, states, goto_nts, flat_action, goto_rows, conflicts。"""
    nonterminals = {p.lhs for p in input_prods}
    original_start = input_prods[0].lhs
    augmented_start = original_start + "'"
    while augmented_start in nonterminals:
        augmented_start += "'"

    productions: List[Production] = [Production(lhs=augmented_start, rhs=[original_start])]
    productions.extend(input_prods)

    nonterminals.add(augmented_start)
    symbols: Set[str] = set()
    for p in productions:
        symbols.add(p.lhs)
        symbols.update(p.rhs)

    terminals: Set[str] = set()
    for s in symbols:
        if s not in nonterminals:
            terminals.add(s)

    prod_of_nt: Dict[str, List[int]] = {}
    for i, p in enumerate(productions):
        prod_of_nt.setdefault(p.lhs, []).append(i)

    states: List[State] = []
    transitions: Dict[Tuple[int, str], int] = {}

    start_kernel = {LR0Item(0, 0)}
    states.append(State(items=_closure(start_kernel, productions, prod_of_nt)))
    q: Deque[int] = deque([0])
    while q:
        sid = q.popleft()
        for sym in sorted(symbols):
            nxt = _goto(states[sid].items, sym, productions, prod_of_nt)
            if not nxt:
                continue
            target = -1
            for i, st in enumerate(states):
                if st.items == nxt:
                    target = i
                    break
            if target == -1:
                target = len(states)
                states.append(State(items=nxt))
                q.append(target)
            transitions[(sid, sym)] = target

    _, _, follow = _compute_nullable_first_follow(productions, nonterminals, terminals, augmented_start)

    action_cols = sorted(terminals)
    if "$" not in action_cols:
        action_cols.append("$")

    goto_nts = sorted(nonterminals - {augmented_start})

    n_states = len(states)
    action: List[List[ActionCell]] = [[ActionCell() for _ in action_cols] for _ in range(n_states)]
    goto_tab: List[Dict[str, int]] = [{} for _ in range(n_states)]
    conflicts: List[str] = []

    for i in range(n_states):
        for ci, a in enumerate(action_cols):
            tr = transitions.get((i, a))
            if tr is not None:
                _merge_action(action[i][ci], ActionKind.shift, tr, f"状态 I{i} 输入 {a}", conflicts)

        for item in states[i].items:
            p = productions[item.prod_id]
            if item.dot_pos < len(p.rhs):
                continue
            if item.prod_id == 0:
                dollar_idx = action_cols.index("$") if "$" in action_cols else -1
                if dollar_idx >= 0:
                    _merge_action(action[i][dollar_idx], ActionKind.accept, 0, f"状态 I{i} 对 $ ", conflicts)
                continue

            lhs = p.lhs
            for la in follow.get(lhs, set()):
                if la not in action_cols:
                    continue
                ci = action_cols.index(la)
                _merge_action(action[i][ci], ActionKind.reduce, item.prod_id, f"状态 I{i} 输入 {la}", conflicts)

        for nt in goto_nts:
            tr = transitions.get((i, nt))
            if tr is not None:
                _merge_goto(goto_tab[i], nt, tr, conflicts, f"I{i},{nt}")

    flat: List[ActionCell] = []
    for si in range(n_states):
        flat.extend(action[si])

    return productions, action_cols, states, goto_nts, flat, goto_tab, conflicts


def format_slr_table_text(
    productions: List[Production],
    action_cols: List[str],
    goto_nts: List[str],
    flat_action: List[ActionCell],
    goto_rows: List[Dict[str, int]],
    conflicts: List[str],
) -> str:
    """与实验四 Python 版风格接近的 ACTION/GOTO 文本，便于对照实验四输出。"""
    n_states = len(flat_action) // len(action_cols)
    lines: List[str] = []
    lines.append("===== SLR(1) 分析表（实验五由文法重建）=====\n")
    lines.append(f"状态数: {n_states}\n")
    lines.append("\n===== ACTION 表 =====\n")
    w0 = max(6, len("State") + 2)
    ws = [max(6, len(c) + 2) for c in action_cols]
    hdr = [f"{'State':<{w0}}"] + [f"{c:<{ws[i]}}" for i, c in enumerate(action_cols)]
    lines.append("".join(hdr).rstrip() + "\n")
    for si in range(n_states):
        row = [f"{('I' + str(si)):<{w0}}"]
        for ci, c in enumerate(action_cols):
            row.append(f"{flat_action[si * len(action_cols) + ci].to_string():<{ws[ci]}}")
        lines.append("".join(row).rstrip() + "\n")
    lines.append("\n===== GOTO 表 =====\n")
    wg = [max(6, len(c) + 2) for c in goto_nts]
    hdr2 = [f"{'State':<{w0}}"] + [f"{c:<{wg[i]}}" for i, c in enumerate(goto_nts)]
    lines.append("".join(hdr2).rstrip() + "\n")
    for si in range(n_states):
        row = [f"{('I' + str(si)):<{w0}}"]
        for ki, nt in enumerate(goto_nts):
            v = str(goto_rows[si][nt]) if nt in goto_rows[si] else ""
            row.append(f"{v:<{wg[ki]}}")
        lines.append("".join(row).rstrip() + "\n")
    lines.append("\n===== 产生式编号（归约 rj 对应 j）=====\n")
    for i, p in enumerate(productions):
        rhs = " ".join(p.rhs) if p.rhs else "ε"
        lines.append(f"  ({i}) {p.lhs} -> {rhs}\n")
    lines.append("\n===== 冲突 =====\n")
    if not conflicts:
        lines.append("无。\n")
    else:
        for c in conflicts:
            lines.append(f"  - {c}\n")
    return "".join(lines)


# ---------- AST & 符号表 ----------

@dataclass
class AstNode:
    label: str
    children: List["AstNode"] = field(default_factory=list)
    lexeme: str = ""
    dtype: str = ""  # 表达式类型: int / float / ""

    def to_text(self, indent: int = 0) -> str:
        pad = "  " * indent
        extra = f" [{self.dtype}]" if self.dtype else ""
        lx = f" '{self.lexeme}'" if self.lexeme else ""
        lines = [f"{pad}{self.label}{lx}{extra}"]
        for c in self.children:
            lines.append(c.to_text(indent + 1))
        return "\n".join(lines)

    def to_dot(self, out: List[str], parent: Optional[int], nid: int) -> int:
        me = nid
        label = self.label.replace("\\", "\\\\").replace('"', '\\"')
        lx = self.lexeme.replace("\\", "\\\\").replace('"', '\\"')
        dtype = self.dtype
        parts = [label]
        if lx:
            parts.append(lx)
        if dtype:
            parts.append(dtype)
        lbl = "\\n".join(parts)
        out.append(f'n{me}[label="{lbl}"];')
        if parent is not None:
            out.append(f"n{parent} -> n{me};")
        nid += 1
        for c in self.children:
            nid = c.to_dot(out, me, nid)
        return nid


def ast_to_dot_source(root: Optional["AstNode"]) -> str:
    lines: List[str] = ["digraph AST {", 'node [shape=box, fontname="Consolas"];']
    if root:
        root.to_dot(lines, None, 0)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _find_dot_executable() -> Optional[str]:
    """
    解析 Graphviz `dot` 可执行文件路径。
    优先顺序：环境变量 GRAPHVIZ_DOT（完整路径）→ GRAPHVIZ_HOME → PATH → Windows 常见安装目录。
    """
    gv_dot = os.environ.get("GRAPHVIZ_DOT", "").strip()
    if gv_dot:
        p = Path(gv_dot)
        if p.is_file():
            return str(p.resolve())

    gv_home = os.environ.get("GRAPHVIZ_HOME", "").strip()
    if gv_home:
        base = Path(gv_home)
        for cand in (base / "bin" / "dot.exe", base / "bin" / "dot", base / "dot.exe", base / "dot"):
            if cand.is_file():
                return str(cand.resolve())

    w = shutil.which("dot")
    if w:
        return w

    if sys.platform == "win32":
        for root in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ):
            try:
                base = Path(root)
            except Exception:
                continue
            if not base.is_dir():
                continue
            try:
                for d in sorted(base.glob("Graphviz*/bin/dot.exe")):
                    if d.is_file():
                        return str(d.resolve())
            except OSError:
                continue
    return None


def save_ast_dot_and_render_png(
    dot_path: Path, png_path: Path, root: Optional["AstNode"], try_png: bool = True
) -> Tuple[str, Optional[str], str]:
    """
    将 AST 写入 Graphviz dot；若 try_png 为真则尝试执行 `dot -Tpng`。
    返回 (dot 文件绝对路径, png 绝对路径或 None, ast_png_status 多行说明)。
    """
    dot_path = dot_path.resolve()
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(ast_to_dot_source(root), encoding="utf-8")
    dot_abs = str(dot_path)
    if not try_png:
        return dot_abs, None, "SKIPPED (--no-ast-png)\n"

    png_path = png_path.resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    dot_exe = _find_dot_executable()
    if not dot_exe:
        msg = (
            "FAILED\n"
            "reason: 未找到 Graphviz dot。请将 GRAPHVIZ_DOT 设为 dot.exe 的完整路径，"
            "或安装 Graphviz 并加入 PATH。\n"
        )
        print("警告: 未找到 Graphviz dot，跳过 ast.png（可设置环境变量 GRAPHVIZ_DOT）", file=sys.stderr)
        return dot_abs, None, msg

    cmd = [dot_exe, "-Tpng", "-o", str(png_path), str(dot_path)]
    try:
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        msg = f"FAILED\nreason: 无法执行 {dot_exe!r}（FileNotFoundError）\n"
        print(f"警告: {msg.strip()}", file=sys.stderr)
        return dot_abs, None, msg
    except subprocess.TimeoutExpired:
        print("警告: 调用 Graphviz dot 超时，未生成 PNG。", file=sys.stderr)
        return dot_abs, None, "FAILED\nreason: dot 进程超时（60s）\n"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        print(f"警告: dot 生成 PNG 失败 (exit {proc.returncode}): {err}", file=sys.stderr)
        return dot_abs, None, f"FAILED\nexit_code: {proc.returncode}\nstderr: {err}\n"

    if not png_path.is_file():
        return dot_abs, None, "FAILED\nreason: dot 已退出 0 但未找到输出 png 文件\n"

    ok_msg = f"OK\ndot_executable: {dot_exe}\npng: {png_path}\n"
    return dot_abs, str(png_path), ok_msg


def format_symbol_table_text(ctx: SemContext) -> str:
    lines = [
        "变量名\t类型\t作用域\t偏移",
        "------\t----\t------\t----",
    ]
    for e in ctx.symtab.all_entries:
        lines.append(f"{e.name}\t{e.dtype}\t{e.scope}\t{e.offset}")
    if not ctx.symtab.all_entries:
        lines.append("(无声明记录)")
    return "\n".join(lines) + "\n"


def format_semantic_errors_text(ctx: SemContext) -> str:
    if not ctx.errors:
        return "(无)\n"
    return "\n".join(ctx.errors) + "\n"


def format_report_text(
    ctx: SemContext,
    trace: List[str],
    grammar_label: str,
    token_label: str,
    conflicts: List[str],
    ast_dot_path: str,
    png_path: Optional[str],
) -> str:
    parts: List[str] = []
    parts.append("实验5 SLR(1) 语义分析 — 汇总报告\n")
    parts.append(f"文法: {grammar_label}\n")
    parts.append(f"Token: {token_label}\n")
    parts.append("\n===== 抽象语法树（缩进文本）=====\n")
    parts.append(ctx.root.to_text() if ctx.root else "(无)\n")
    parts.append("\n===== 符号表（变量名、类型、作用域、偏移）=====\n")
    parts.append(format_symbol_table_text(ctx))
    parts.append("\n===== 语义 / 语法错误报告 =====\n")
    parts.append(format_semantic_errors_text(ctx))
    parts.append("\n===== AST 可视化文件 =====\n")
    parts.append(f"Graphviz: {ast_dot_path}\n")
    parts.append(f"PNG 图片: {png_path or '(未生成或未安装 Graphviz)'}\n")
    parts.append("\n===== SLR(1) 冲突（若有）=====\n")
    if conflicts:
        parts.extend(f"  - {c}\n" for c in conflicts)
    else:
        parts.append("(无)\n")
    parts.append(f"\n===== 分析轨迹（共 {len(trace)} 步）=====\n")
    parts.extend(t + "\n" for t in trace)
    parts.append("\n===== 并行栈终态（状态栈 / 文法符号栈 / 属性栈）=====\n")
    parts.append(ctx.stack_summary or "(未记录)\n")
    return "".join(parts)


@dataclass
class SymEntry:
    name: str
    dtype: str
    scope: int
    offset: int


class SymbolTable:
    def __init__(self) -> None:
        self.stack: List[Dict[str, SymEntry]] = [{}]
        self.scope_level = 0
        self.offsets: List[int] = [0]
        self.all_entries: List[SymEntry] = []

    def push_scope(self) -> None:
        self.scope_level += 1
        self.stack.append({})
        self.offsets.append(0)

    def pop_scope(self) -> None:
        if self.scope_level <= 0:
            return
        self.stack.pop()
        self.offsets.pop()
        self.scope_level -= 1

    def declare(self, name: str, dtype: str) -> Tuple[bool, str]:
        cur = self.stack[-1]
        if name in cur:
            return False, f"重复声明: 标识符 '{name}' 在当前作用域(深度 {self.scope_level})已存在"
        off = self.offsets[-1]
        self.offsets[-1] += 1
        e = SymEntry(name=name, dtype=dtype, scope=self.scope_level, offset=off)
        cur[name] = e
        self.all_entries.append(e)
        return True, ""

    def lookup(self, name: str) -> Optional[SymEntry]:
        for d in reversed(self.stack):
            if name in d:
                return d[name]
        return None


@dataclass
class SemanticValue:
    """属性栈元素：综合属性（表达式类型、子树指针、终结符词素等）。"""

    lexeme: str = ""
    pos: str = ""
    ast: Optional[AstNode] = None
    dtype: str = ""


@dataclass
class StackItem:
    """归约语义子程序入参：由属性栈与符号栈对应片段组装，便于按产生式子结点访问。"""

    sym: str
    state: int
    lexeme: str = ""
    pos: str = ""
    ast: Optional[AstNode] = None
    dtype: str = ""


@dataclass
class SemContext:
    symtab: SymbolTable
    errors: List[str]
    root: Optional[AstNode] = None
    stack_summary: str = ""

    def err(self, msg: str) -> None:
        self.errors.append(msg)


def _format_semantic_value_short(a: SemanticValue) -> str:
    bits: List[str] = []
    if a.lexeme:
        bits.append(a.lexeme)
    if a.dtype:
        bits.append(f"ty={a.dtype}")
    if a.ast:
        bits.append(f"@{a.ast.label}")
    return "/".join(bits) if bits else "—"


def _snapshot_parallel_stacks(
    ctx: SemContext,
    state_stack: List[int],
    symbol_stack: List[str],
    attr_stack: List[SemanticValue],
) -> None:
    """记录状态栈、文法符号栈、属性栈终态（底→顶），供实验报告与 parallel_stacks.txt。"""
    lines = [
        "并行栈终态（自栈底至栈顶；栈顶即当前分析格局）",
        "",
        "【状态栈 state_stack】",
        "  " + " ".join(str(s) for s in state_stack),
        "",
        "【文法符号栈 symbol_stack】",
        "  " + " ".join(symbol_stack),
        "",
        "【属性栈 attr_stack】（与符号栈一一对齐：词素 / 类型 / AST 结点类型）",
        "  " + " | ".join(_format_semantic_value_short(a) for a in attr_stack),
        "",
        f"三栈长度一致: {len(state_stack)} == {len(symbol_stack)} == {len(attr_stack)}",
    ]
    ctx.stack_summary = "\n".join(lines) + "\n"


def _arith_type(a: str, b: str) -> str:
    if a == "float" or b == "float":
        return "float"
    return "int"


def _coerce_assign(var_t: str, expr_t: str) -> Tuple[bool, str]:
    if var_t == expr_t:
        return True, var_t
    if var_t == "float" and expr_t == "int":
        return True, "float"
    if var_t == "int" and expr_t == "float":
        return False, ""
    return False, ""


def parse_token_line(line: str) -> Optional[Tuple[str, str, str]]:
    """返回 (类型, 词素, 附加信息)。`<T, lex>` 或 `(T, lex)` 后可有任意附加文本（如 位置(...)），也可省略。"""
    line = _trim(line)
    if not line:
        return None
    m = re.match(r"^<([^,>]+),\s*([^>]+)>\s*(.*)$", line)
    if m:
        return _trim(m.group(1)), _trim(m.group(2)), _trim(m.group(3))
    m2 = re.match(r"^\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*(.*)$", line)
    if m2:
        return _trim(m2.group(1)), _trim(m2.group(2)), _trim(m2.group(3))
    return None


SemanticFn = Callable[[SemContext, Production, List[StackItem]], StackItem]


def build_semantic_handlers(action_names: List[str]) -> Dict[str, SemanticFn]:
    """动作名 -> 归约语义函数（不含增广产生式 0，其在驱动里处理）。"""

    def noop(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        return StackItem(sym=p.lhs, state=-1, ast=None, dtype="")

    def program(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        # P -> SL
        sl = kids[0]
        return StackItem(sym=p.lhs, state=-1, ast=sl.ast, dtype="")

    def append_stmt(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        sl, s = kids[0], kids[1]
        n = AstNode("stmt_list")
        if sl.ast:
            if sl.ast.label == "stmt_list":
                n.children.extend(sl.ast.children)
            else:
                n.children.append(sl.ast)
        if s.ast:
            n.children.append(s.ast)
        return StackItem(sym=p.lhs, state=-1, ast=n, dtype="")

    def single_stmt(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        s = kids[0]
        n = AstNode("stmt_list")
        if s.ast:
            n.children.append(s.ast)
        return StackItem(sym=p.lhs, state=-1, ast=n, dtype="")

    def decl_int(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        _, idk, _ = kids
        name = idk.lexeme
        ok, msg = ctx.symtab.declare(name, "int")
        if not ok:
            ctx.err(msg)
        node = AstNode("decl", lexeme=name, dtype="int")
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="void")

    def decl_float(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        _, idk, _ = kids
        name = idk.lexeme
        ok, msg = ctx.symtab.declare(name, "float")
        if not ok:
            ctx.err(msg)
        node = AstNode("decl", lexeme=name, dtype="float")
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="void")

    def block(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        _, sl, _ = kids
        node = AstNode("block")
        if sl.ast:
            node.children.append(sl.ast)
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="void")

    def assign(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        idk, _, e, _ = kids
        name = idk.lexeme
        ent = ctx.symtab.lookup(name)
        if ent is None:
            ctx.err(f"赋值错误: 未声明的标识符 '{name}'")
            return StackItem(sym=p.lhs, state=-1, ast=AstNode("assign", lexeme=name, children=[e.ast] if e.ast else []), dtype="void")
        ok, _ = _coerce_assign(ent.dtype, e.dtype)
        if not ok:
            ctx.err(f"类型不匹配: 变量 '{name}' 为 {ent.dtype}，表达式为 {e.dtype}")
        node = AstNode("assign", lexeme=name, dtype=ent.dtype, children=[e.ast] if e.ast else [])
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="void")

    def bin_plus(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        e, _, t = kids
        dt = _arith_type(e.dtype, t.dtype)
        node = AstNode("bin", lexeme="+", dtype=dt, children=[e.ast, t.ast] if e.ast and t.ast else [])
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype=dt)

    def bin_minus(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        e, _, t = kids
        dt = _arith_type(e.dtype, t.dtype)
        node = AstNode("bin", lexeme="-", dtype=dt, children=[e.ast, t.ast] if e.ast and t.ast else [])
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype=dt)

    def bin_mul(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        t, _, f = kids
        dt = _arith_type(t.dtype, f.dtype)
        node = AstNode("bin", lexeme="*", dtype=dt, children=[t.ast, f.ast] if t.ast and f.ast else [])
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype=dt)

    def bin_div(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        t, _, f = kids
        dt = _arith_type(t.dtype, f.dtype)
        node = AstNode("bin", lexeme="/", dtype=dt, children=[t.ast, f.ast] if t.ast and f.ast else [])
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype=dt)

    def pass_e(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        t = kids[0]
        return StackItem(sym=p.lhs, state=-1, ast=t.ast, dtype=t.dtype)

    def pass_t(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        f = kids[0]
        return StackItem(sym=p.lhs, state=-1, ast=f.ast, dtype=f.dtype)

    def paren(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        _, e, _ = kids
        return StackItem(sym=p.lhs, state=-1, ast=e.ast, dtype=e.dtype)

    def id_ref(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        idk = kids[0]
        ent = ctx.symtab.lookup(idk.lexeme)
        if ent is None:
            ctx.err(f"表达式错误: 未声明的标识符 '{idk.lexeme}'")
            return StackItem(sym=p.lhs, state=-1, ast=AstNode("id", lexeme=idk.lexeme, dtype="int"), dtype="int")
        node = AstNode("id", lexeme=idk.lexeme, dtype=ent.dtype)
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype=ent.dtype)

    def int_lit(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        k = kids[0]
        node = AstNode("literal", lexeme=k.lexeme, dtype="int")
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="int")

    def float_lit(ctx: SemContext, p: Production, kids: List[StackItem]) -> StackItem:
        k = kids[0]
        node = AstNode("literal", lexeme=k.lexeme, dtype="float")
        return StackItem(sym=p.lhs, state=-1, ast=node, dtype="float")

    reg: Dict[str, SemanticFn] = {
        "noop": noop,
        "program": program,
        "append_stmt": append_stmt,
        "single_stmt": single_stmt,
        "decl_int": decl_int,
        "decl_float": decl_float,
        "block": block,
        "assign": assign,
        "bin_plus": bin_plus,
        "bin_minus": bin_minus,
        "bin_mul": bin_mul,
        "bin_div": bin_div,
        "pass_e": pass_e,
        "pass_t": pass_t,
        "paren": paren,
        "id_ref": id_ref,
        "int_lit": int_lit,
        "float_lit": float_lit,
    }
    for name in action_names:
        if name not in reg:
            reg[name] = noop
    return reg


SHIFT_HOOKS: Dict[str, Callable[[SemContext], None]] = {
    "LBR": lambda ctx: ctx.symtab.push_scope(),
}


def run_semantic_analysis(
    grammar_text: str,
    token_text: str,
) -> Dict[str, object]:
    """内存运行语义分析，供 Web API 调用。"""
    base_prods, file_actions, gerr = load_grammar_with_actions_from_text(grammar_text)
    if gerr and not base_prods:
        return {"ok": False, "errors": gerr}
    if not base_prods:
        return {"ok": False, "errors": gerr or ["文法为空或无法解析"]}

    productions, action_cols, states, goto_nts, flat_action, goto_rows, conflicts = build_slr1_tables(
        base_prods
    )
    slr1_table = format_slr_table_text(productions, action_cols, goto_nts, flat_action, goto_rows, conflicts)
    slr_conflicts = "\n".join(conflicts) + "\n" if conflicts else "(无)\n"

    action_names = file_actions
    if len(action_names) != len(base_prods):
        while len(action_names) < len(base_prods):
            action_names.append("noop")

    toks: List[Tuple[str, str, str]] = []
    for line in token_text.splitlines():
        pr = parse_token_line(line)
        if pr:
            toks.append(pr)

    ctx, trace = drive_lr(
        productions,
        action_cols,
        goto_nts,
        flat_action,
        goto_rows,
        action_names,
        toks,
    )

    ast_tree = (ctx.root.to_text() if ctx.root else "(无)\n") + "\n"
    symbol_table = format_symbol_table_text(ctx)
    semantic_errors = format_semantic_errors_text(ctx)
    parse_trace = "\n".join(trace) + ("\n" if trace else "")
    tok_lines: List[str] = []
    for a, b, c in toks:
        if c:
            tok_lines.append(f"<{a}, {b}> {c}")
        else:
            tok_lines.append(f"<{a}, {b}>")
    token_stream = "\n".join(tok_lines) + ("\n" if tok_lines else "")
    ast_dot = ast_to_dot_source(ctx.root)
    parallel_stacks = ctx.stack_summary or "(未记录)\n"
    report = format_report_text(
        ctx, trace, "（网页输入）", "（网页输入）", conflicts, "(ast.dot)", None
    )

    return {
        "ok": True,
        "success": not ctx.errors and ctx.root is not None,
        "ast_tree": ast_tree,
        "ast_dot": ast_dot,
        "symbol_table": symbol_table,
        "semantic_errors": semantic_errors,
        "parse_trace": parse_trace,
        "token_stream": token_stream,
        "slr1_table": slr1_table,
        "slr_conflicts": slr_conflicts,
        "parallel_stacks": parallel_stacks,
        "report": report,
        "grammar_warnings": gerr,
    }


def run_pipeline(
    grammar_path: Path,
    token_path: Path,
    dump_trace: bool,
    dump_table: Optional[Path],
    out_dir: Path,
    no_ast_png: bool,
) -> int:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_prods, file_actions, gerr = load_grammar_with_actions(grammar_path)
    if gerr:
        err_file = out_dir / "grammar_load_errors.txt"
        err_file.write_text("\n".join(gerr) + "\n", encoding="utf-8")
        for e in gerr:
            print(e, file=sys.stderr)
    if not base_prods:
        return 1

    productions, action_cols, states, goto_nts, flat_action, goto_rows, conflicts = build_slr1_tables(base_prods)
    table_text = format_slr_table_text(productions, action_cols, goto_nts, flat_action, goto_rows, conflicts)
    (out_dir / "slr1_table.txt").write_text(table_text, encoding="utf-8")
    if dump_table:
        dump_table.parent.mkdir(parents=True, exist_ok=True)
        dump_table.write_text(table_text, encoding="utf-8")
    if conflicts:
        (out_dir / "slr_conflicts.txt").write_text("\n".join(conflicts) + "\n", encoding="utf-8")
        print(f"===== SLR(1) 冲突（已写入 {out_dir / 'slr_conflicts.txt'}）=====", file=sys.stderr)
        for c in conflicts:
            print(c, file=sys.stderr)
    else:
        (out_dir / "slr_conflicts.txt").write_text("(无)\n", encoding="utf-8")

    action_names = file_actions
    if len(action_names) != len(base_prods):
        print("警告: 动作表长度与产生式不一致，已用 noop 补齐", file=sys.stderr)
        while len(action_names) < len(base_prods):
            action_names.append("noop")

    toks: List[Tuple[str, str, str]] = []
    for line in token_path.read_text(encoding="utf-8").splitlines():
        pr = parse_token_line(line)
        if pr:
            toks.append(pr)

    ctx, trace = drive_lr(
        productions,
        action_cols,
        goto_nts,
        flat_action,
        goto_rows,
        action_names,
        toks,
    )

    ast_tree_path = out_dir / "ast_tree.txt"
    ast_dot_path = out_dir / "ast.dot"
    ast_png_path = out_dir / "ast.png"
    sym_path = out_dir / "symbol_table.txt"
    err_path = out_dir / "semantic_errors.txt"
    trace_path = out_dir / "parse_trace.txt"
    report_path = out_dir / "report.txt"

    ast_tree_path.write_text(
        (ctx.root.to_text() if ctx.root else "(无)\n") + "\n",
        encoding="utf-8",
    )
    sym_path.write_text(format_symbol_table_text(ctx), encoding="utf-8")
    err_path.write_text(format_semantic_errors_text(ctx), encoding="utf-8")
    trace_path.write_text("\n".join(trace) + ("\n" if trace else ""), encoding="utf-8")

    tok_lines: List[str] = []
    for a, b, c in toks:
        if c:
            tok_lines.append(f"<{a}, {b}> {c}")
        else:
            tok_lines.append(f"<{a}, {b}>")
    (out_dir / "token_stream.txt").write_text("\n".join(tok_lines) + ("\n" if tok_lines else ""), encoding="utf-8")

    dot_abs, png_abs, ast_png_status = save_ast_dot_and_render_png(
        ast_dot_path, ast_png_path, ctx.root, try_png=not no_ast_png
    )
    (out_dir / "ast_png_status.txt").write_text(ast_png_status, encoding="utf-8")

    report_path.write_text(
        format_report_text(ctx, trace, str(grammar_path), str(token_path), conflicts, dot_abs, png_abs),
        encoding="utf-8",
    )
    (out_dir / "parallel_stacks.txt").write_text(ctx.stack_summary or "(未记录)\n", encoding="utf-8")

    print("===== 运行完成：所有结果已写入文件 =====")
    print(f"输出目录: {out_dir}")
    print("  ast_tree.txt       抽象语法树（缩进文本）")
    print("  ast.dot            AST Graphviz")
    print("  ast.png            AST 图片（默认调用 Graphviz；仅 --no-ast-png 时跳过）")
    print("  symbol_table.txt   符号表（变量名、类型、作用域、偏移）")
    print("  semantic_errors.txt 语义/语法错误报告")
    print("  token_stream.txt   本次参与分析的 Token 序列（从输入文件解析）")
    print("  parse_trace.txt    完整 SLR 分析轨迹")
    print("  slr1_table.txt     重建的 SLR(1) 分析表")
    print("  slr_conflicts.txt  SLR 冲突说明")
    print("  parallel_stacks.txt 状态/符号/属性三栈终态摘要")
    if dump_table:
        print(f"  (另存分析表) {dump_table.resolve()}")

    if dump_trace:
        print("\n--- 分析轨迹（末尾 15 行预览）---")
        for t in trace[-15:]:
            print(t)

    return 0 if not ctx.errors and ctx.root else 1


def drive_lr(
    productions: List[Production],
    action_cols: List[str],
    goto_nts: List[str],
    action: List[ActionCell],
    goto_rows: List[Dict[str, int]],
    action_names: List[str],
    tokens: List[Tuple[str, str, str]],
) -> Tuple[SemContext, List[str]]:
    """
    SLR(1) 驱动 + 归约语义。

    采用与教材一致的「语法分析栈 + 属性栈」并行结构：
    - state_stack：各步 LR 状态号（栈顶对应当前格局）；
    - symbol_stack：栈上文法符号（终结符/非终结符）；
    - attr_stack：与 symbol_stack 等长的综合属性（AST、类型、词素等）。

    移进时三栈同步 push；归约时同步 pop |rhs| 次，再按 GOTO 压入新非终结符及其属性。
    """
    n_cols = len(action_cols)
    ctx = SemContext(symtab=SymbolTable(), errors=[])
    sem = build_semantic_handlers(action_names)

    def name_for(pid: int) -> str:
        if pid == 0:
            return "accept_root"
        idx = pid - 1
        if 0 <= idx < len(action_names):
            return action_names[idx]
        return "noop"

    def cell_at(st: int, sym: str) -> ActionCell:
        if sym not in action_cols:
            return ActionCell()
        return action[st * n_cols + action_cols.index(sym)]

    def kids_from_stacks(syms: List[str], attrs: List[SemanticValue], n: int) -> List[StackItem]:
        """取栈顶 n 个符号及其属性，按产生式右部从左到右顺序。"""
        out: List[StackItem] = []
        base = len(syms) - n
        for i in range(n):
            idx = base + i
            a = attrs[idx]
            out.append(
                StackItem(
                    sym=syms[idx],
                    state=-1,
                    lexeme=a.lexeme,
                    pos=a.pos,
                    ast=a.ast,
                    dtype=a.dtype,
                )
            )
        return out

    state_stack: List[int] = [0]
    symbol_stack: List[str] = ["$"]
    attr_stack: List[SemanticValue] = [SemanticValue()]

    inp = list(tokens) + [("$", "$", "")]
    ip = 0
    trace: List[str] = []

    while True:
        st = state_stack[-1]
        la, lex, pos = inp[ip]
        cell = cell_at(st, la)
        pos_note = f" {pos}" if pos else ""
        trace.append(f"I{st} + {la}({lex}) => {cell.to_string()}{pos_note}")

        if cell.kind == ActionKind.shift:
            if la in SHIFT_HOOKS:
                SHIFT_HOOKS[la](ctx)
            state_stack.append(cell.num)
            symbol_stack.append(la)
            attr_stack.append(SemanticValue(lexeme=lex, pos=pos))
            ip += 1
            continue

        if cell.kind == ActionKind.reduce:
            pid = cell.num
            p = productions[pid]
            n = len(p.rhs)
            kids = kids_from_stacks(symbol_stack, attr_stack, n)
            for _ in range(n):
                state_stack.pop()
                symbol_stack.pop()
                attr_stack.pop()
            gstate = state_stack[-1]
            nm = name_for(pid)
            if pid == 0:
                new_it = StackItem(sym=p.lhs, state=-1, ast=kids[0].ast if kids else None, dtype="")
            elif nm == "block":
                new_it = sem["block"](ctx, p, kids)
                ctx.symtab.pop_scope()
            else:
                new_it = sem.get(nm, sem["noop"])(ctx, p, kids)
            g = goto_rows[gstate].get(p.lhs, -1)
            if g < 0:
                ctx.err(f"GOTO 缺失: I{gstate} . {p.lhs}")
                _snapshot_parallel_stacks(ctx, state_stack, symbol_stack, attr_stack)
                return ctx, trace
            state_stack.append(g)
            symbol_stack.append(p.lhs)
            attr_stack.append(
                SemanticValue(lexeme=new_it.lexeme, pos=new_it.pos, ast=new_it.ast, dtype=new_it.dtype)
            )
            continue

        if cell.kind == ActionKind.accept:
            if attr_stack:
                ctx.root = attr_stack[-1].ast
            _snapshot_parallel_stacks(ctx, state_stack, symbol_stack, attr_stack)
            return ctx, trace

        if cell.kind == ActionKind.none:
            ctx.err(f"语法错误: I{st} 无法处理 {la}{f' {pos}' if pos else ''}")
            _snapshot_parallel_stacks(ctx, state_stack, symbol_stack, attr_stack)
            return ctx, trace

        ctx.err(f"分析表冲突: {cell.conflict_note}")
        _snapshot_parallel_stacks(ctx, state_stack, symbol_stack, attr_stack)
        return ctx, trace


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="实验5 SLR(1) 语义分析")
    ap.add_argument("--grammar", type=Path, default=Path(__file__).with_name("grammar_input.txt"))
    ap.add_argument("--tokens", type=Path, default=Path(__file__).with_name("sample_tokens.txt"))
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--dump-table", type=Path, default=None, help="将 SLR(1) 表额外复制到指定路径（默认已写入输出目录 slr1_table.txt）")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="所有生成文件保存目录（默认 实验5/output）",
    )
    ap.add_argument("--no-ast-png", action="store_true", help="不调用 dot 生成 ast.png（仍保存 ast.dot）")
    args = ap.parse_args()
    return run_pipeline(args.grammar, args.tokens, args.trace, args.dump_table, args.out_dir, args.no_ast_png)


if __name__ == "__main__":
    raise SystemExit(main())
