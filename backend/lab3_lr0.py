from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class LR0Request(BaseModel):
    grammar_text: str = Field(default="")


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


def _trim(s: str) -> str:
    return s.strip()


def _split_by_char(s: str, delim: str) -> List[str]:
    return s.split(delim)


def _tokenize_rhs(rhs: str) -> List[str]:
    t = _trim(rhs)
    if not t:
        return []

    # If whitespace-separated, follow C++ behavior.
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
        c = t[i]
        if c.isalnum() or c in {"_", "'"}:
            j = i
            while j < len(t) and (t[j].isalnum() or t[j] in {"_", "'"}):
                j += 1
            tok = t[i:j]
            if tok not in {"epsilon", "eps"}:
                tokens.append(tok)
            i = j
        else:
            tokens.append(c)
            i += 1
    return tokens


def _item_to_string(item: LR0Item, productions: List[Production]) -> str:
    p = productions[item.prod_id]
    parts: List[str] = [p.lhs, "->"]
    for i in range(0, len(p.rhs) + 1):
        if i == item.dot_pos:
            parts.append(".")
        if i < len(p.rhs):
            parts.append(p.rhs[i])
    return " ".join(parts).strip()


def _closure(kernel: Set[LR0Item], productions: List[Production], prod_of_nt: Dict[str, List[int]]) -> Set[LR0Item]:
    res: Set[LR0Item] = set(kernel)
    q: deque[LR0Item] = deque(kernel)
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


def _dot_escape(s: str) -> str:
    out = []
    for c in s:
        if c in {'\\', '"', "{", "}", "|"}:
            out.append("\\")
        out.append(c)
    return "".join(out)


def _export_dot(states: List[State], transitions: Dict[Tuple[int, str], int], productions: List[Production]) -> str:
    lines: List[str] = []
    lines.append("digraph LR0 {")
    lines.append("  rankdir=LR;")
    lines.append('  node [shape=box, fontsize=10, fontname="Consolas"];')
    lines.append('  edge [fontsize=10, fontname="Consolas"];')
    lines.append("")
    for i, st in enumerate(states):
        label = [f"I{i}\\n"]
        for item in sorted(st.items):
            label.append(_item_to_string(item, productions) + "\\l")
        lines.append(f'  I{i} [label="{_dot_escape("".join(label))}"];')
    lines.append("")
    for (frm, sym), to in transitions.items():
        lines.append(f'  I{frm} -> I{to} [label="{_dot_escape(sym)}"];')
    lines.append("}")
    return "\n".join(lines)


def _parse_lr0_input(text: str) -> Tuple[Optional[List[Production]], List[str]]:
    errors: List[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ["请输入文法。"]

    # Preferred format: first token is an integer N
    n: Optional[int] = None
    first = lines[0].split()[0]
    if first.isdigit():
        n = int(first)
        lines = lines[1:]
        if n <= 0:
            return None, ["产生式数量必须为正整数。"]
        if len(lines) < n:
            return None, [f"期望 {n} 条产生式, 实际读取到 {len(lines)} 条。"]
        lines = lines[:n]
    else:
        # Fallback: treat each non-empty line as one production line.
        n = len(lines)

    input_prods: List[Production] = []
    loaded = 0
    for line in lines:
        loaded += 1
        if "->" not in line:
            errors.append(f"第 {loaded} 条产生式缺少 '->'。")
            continue
        lhs, rhs_part = line.split("->", 1)
        lhs = lhs.strip()
        if not lhs:
            errors.append(f"第 {loaded} 条产生式左部为空。")
            continue
        branches = _split_by_char(rhs_part, "|")
        if not branches:
            errors.append(f"第 {loaded} 条产生式右部为空。")
            continue
        for b in branches:
            input_prods.append(Production(lhs=lhs, rhs=_tokenize_rhs(b)))

    if errors:
        return None, errors
    return input_prods, []


def lr0_run(req: LR0Request):
    input_prods, errs = _parse_lr0_input(req.grammar_text or "")
    if errs:
        return {"ok": False, "errors": errs}
    assert input_prods is not None and len(input_prods) > 0

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
        for sym in p.rhs:
            symbols.add(sym)

    prod_of_nt: Dict[str, List[int]] = {}
    for i, p in enumerate(productions):
        prod_of_nt.setdefault(p.lhs, []).append(i)

    states: List[State] = []
    transitions: Dict[Tuple[int, str], int] = {}

    start_kernel = {LR0Item(0, 0)}
    states.append(State(items=_closure(start_kernel, productions, prod_of_nt)))
    q: deque[int] = deque([0])
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

    is_lr0 = True
    conflicts: List[str] = []
    for i, st in enumerate(states):
        has_reduce = False
        has_shift = False
        reduce_count = 0
        for item in st.items:
            p = productions[item.prod_id]
            if item.dot_pos < len(p.rhs):
                has_shift = True
            else:
                has_reduce = True
                reduce_count += 1
        if has_reduce and has_shift:
            is_lr0 = False
            conflicts.append(f"I{i} 发生移进-归约冲突")
        if reduce_count > 1:
            is_lr0 = False
            conflicts.append(f"I{i} 发生归约-归约冲突")

    out_lines: List[str] = []

    def out(s: str) -> None:
        out_lines.append(s)

    out("===== 增广文法 =====\n")
    for i, p in enumerate(productions):
        rhs = "epsilon" if len(p.rhs) == 0 else " ".join(p.rhs) + " "
        out(f"({i}) {p.lhs} -> {rhs}\n")

    out("\n===== LR(0) 项目集规范族 =====\n")
    for i, st in enumerate(states):
        out(f"I{i}:\n")
        out("  Kernel:\n")
        for item in sorted(st.items):
            if item.dot_pos != 0 or item.prod_id == 0:
                out(f"    {_item_to_string(item, productions)}\n")
        out("  Closure:\n")
        for item in sorted(st.items):
            out(f"    {_item_to_string(item, productions)}\n")
        out("  Goto:\n")
        has_goto = False
        for (frm, sym), to in transitions.items():
            if frm == i:
                has_goto = True
                out(f"    {sym} -> I{to}\n")
        if not has_goto:
            out("    (无)\n")
        out("\n")

    out("===== LR(0) 判定结果 =====\n")
    if is_lr0:
        out("该文法是 LR(0) 文法。\n")
    else:
        out("该文法不是 LR(0) 文法，冲突如下:\n")
        for c in conflicts:
            out(f"  - {c}\n")

    dot = _export_dot(states, transitions, productions)

    return {"ok": True, "output_text": "".join(out_lines), "dot": dot}

