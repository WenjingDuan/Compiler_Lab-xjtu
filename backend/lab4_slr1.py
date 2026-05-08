from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from backend.lab3_lr0 import (
    LR0Item,
    Production,
    State,
    _closure,
    _goto,
    _parse_lr0_input,
)

EPS = "\x01__EPS__\x01"


class SLR1Request(BaseModel):
    grammar_text: str = Field(default="")


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

    first = {t: {t} for t in terminals}
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


def _format_table(title: str, cols: List[str], n_states: int, cell: Callable[[int, int], str]) -> str:
    lines: List[str] = []
    lines.append(title + "\n")
    n = len(cols)
    w = [6] * (n + 1)
    for si in range(n_states):
        w[0] = max(w[0], len(f"I{si}") + 2)
    for k in range(n):
        w[k + 1] = max(w[k + 1], len(cols[k]) + 2)
    for si in range(n_states):
        for k in range(n):
            w[k + 1] = max(w[k + 1], len(cell(si, k)) + 2)

    hdr_parts = [f"{'State':<{w[0]}}"]
    for k in range(n):
        hdr_parts.append(f"{cols[k]:<{w[k + 1]}}")
    lines.append("".join(hdr_parts).rstrip() + "\n")

    for si in range(n_states):
        parts = [f"{('I' + str(si)):<{w[0]}}"]
        for k in range(n):
            parts.append(f"{cell(si, k):<{w[k + 1]}}")
        lines.append("".join(parts).rstrip() + "\n")

    return "".join(lines)


def slr1_run(req: SLR1Request):
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

    _, _, follow = _compute_nullable_first_follow(
        productions, nonterminals, terminals, augmented_start
    )

    action_cols = sorted(terminals)
    if "$" not in action_cols:
        action_cols.append("$")

    goto_nts = sorted(nonterminals - {augmented_start})

    n_states = len(states)
    action: List[List[ActionCell]] = [
        [ActionCell() for _ in action_cols] for _ in range(n_states)
    ]
    goto_tab: List[Dict[str, int]] = [{} for _ in range(n_states)]
    conflicts: List[str] = []

    for i in range(n_states):
        for ci, a in enumerate(action_cols):
            tr = transitions.get((i, a))
            if tr is not None:
                _merge_action(
                    action[i][ci],
                    ActionKind.shift,
                    tr,
                    f"状态 I{i} 输入 {a}",
                    conflicts,
                )

        for item in states[i].items:
            p = productions[item.prod_id]
            if item.dot_pos < len(p.rhs):
                continue
            if item.prod_id == 0:
                dollar_idx = action_cols.index("$") if "$" in action_cols else -1
                if dollar_idx >= 0:
                    _merge_action(
                        action[i][dollar_idx],
                        ActionKind.accept,
                        0,
                        f"状态 I{i} 对 $ ",
                        conflicts,
                    )
                continue

            lhs = p.lhs
            for la in follow.get(lhs, set()):
                if la not in action_cols:
                    continue
                ci = action_cols.index(la)
                _merge_action(
                    action[i][ci],
                    ActionKind.reduce,
                    item.prod_id,
                    f"状态 I{i} 输入 {la}",
                    conflicts,
                )

        for nt in goto_nts:
            tr = transitions.get((i, nt))
            if tr is not None:
                _merge_goto(goto_tab[i], nt, tr, conflicts, f"I{i},{nt}")

    out_lines: List[str] = []

    def out(s: str) -> None:
        out_lines.append(s)

    out("===== SLR(1) 分析表 =====\n")
    out("FOLLOW: 由 FIRST/FOLLOW 算法自动计算\n")
    out(f"状态数: {n_states}\n")

    out("\n===== FOLLOW 集（自动计算）=====\n")
    for nt in sorted(nonterminals):
        syms = sorted(follow.get(nt, set()))
        out(nt + ": " + (" ".join(syms) if syms else "") + "\n")

    out("\n")
    out(
        _format_table(
            "===== ACTION 表 =====",
            action_cols,
            n_states,
            lambda si, k: action[si][k].to_string(),
        )
    )
    out("\n")
    out(
        _format_table(
            "===== GOTO 表 =====",
            goto_nts,
            n_states,
            lambda si, k: (
                str(goto_tab[si][goto_nts[k]]) if goto_nts[k] in goto_tab[si] else ""
            ),
        )
    )

    out("\n===== 符号说明 =====\n")
    out("ACTION: si = 移进到状态 i; rj = 按产生式编号 j 归约; acc = 接受\n")
    out("GOTO: 数字为目标状态编号\n")
    out("若格中为 a/b 形式，表示 SLR(1) 在该位置仍存在冲突。\n\n")

    if not conflicts:
        out("===== 冲突 =====\n无(在当前 FOLLOW 下未发现多重填入同一格)。\n")
    else:
        out("===== 冲突 =====\n")
        for c in conflicts:
            out(f"  - {c}\n")

    return {"ok": True, "output_text": "".join(out_lines)}
