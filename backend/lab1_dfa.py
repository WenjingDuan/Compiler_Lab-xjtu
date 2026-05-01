from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class DFAFromGrammarRequest(BaseModel):
    grammar_text: str = Field(default="")


class DFAFromTextRequest(BaseModel):
    dfa_text: str = Field(default="")


class DFARunRequest(BaseModel):
    dfa_text: str
    input_str: str = Field(default="")


class DFAEnumRequest(BaseModel):
    dfa_text: str
    max_len: int = Field(default=0, ge=0, le=12)


class Grammar(BaseModel):
    terminals: Set[str]
    nonterminals: Set[str]
    start: str
    prods: Dict[str, List[str]]


def _split_words(s: str) -> List[str]:
    return [x for x in s.strip().split() if x]


def parse_grammar(text: str) -> Tuple[Optional[Grammar], List[str]]:
    terminals: Set[str] = set()
    nonterminals: Set[str] = set()
    start = ""
    prods: Dict[str, List[str]] = {}
    errors: List[str] = []
    in_prods = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not in_prods:
            if line.startswith("terminals:"):
                for tok in _split_words(line[len("terminals:") :]):
                    if len(tok) != 1:
                        errors.append(f"终结符必须单字符: {tok}")
                    else:
                        terminals.add(tok)
            elif line.startswith("nonterminals:"):
                for tok in _split_words(line[len("nonterminals:") :]):
                    nonterminals.add(tok)
            elif line.startswith("start:"):
                start = line[len("start:") :].strip()
            elif line == "productions:":
                in_prods = True
            else:
                errors.append(f"文法头部无法识别: {line}")
        else:
            if line == "end":
                break
            if "->" not in line:
                errors.append(f"产生式缺少 ->: {line}")
                continue
            lhs, rhs_all = line.split("->", 1)
            lhs = lhs.strip()
            if lhs not in nonterminals:
                errors.append(f"左部不在非终结符集: {lhs}")
                continue
            branches = [x.strip() for x in rhs_all.split("|")]
            for br in branches:
                rhs = "".join(br.split())
                if not rhs:
                    errors.append(f"空右部: {lhs} -> (空)")
                    continue
                prods.setdefault(lhs, []).append(rhs)

    if start and start not in nonterminals:
        errors.append("开始符号非法")

    if not terminals:
        errors.append("缺少 terminals 定义")
    if not nonterminals:
        errors.append("缺少 nonterminals 定义")
    if not start:
        errors.append("缺少 start 定义")
    if not prods:
        errors.append("缺少 productions 定义")

    if errors:
        return None, errors

    for lhs, rhs_list in prods.items():
        for rhs in rhs_list:
            if rhs in {"ε", "eps", "epsilon"}:
                continue
            if len(rhs) == 1:
                if rhs[0] not in {t for t in terminals}:
                    errors.append(f"非法右部: {lhs}->{rhs}")
            else:
                a = rhs[0]
                B = rhs[1:]
                if a not in terminals:
                    errors.append(f"非法终结符: {lhs}->{rhs}")
                if B not in nonterminals:
                    errors.append(f"非法非终结符: {lhs}->{rhs}")

    if errors:
        return None, errors

    return Grammar(terminals=terminals, nonterminals=nonterminals, start=start, prods=prods), []


@dataclass(frozen=True)
class DFA:
    states: List[str]
    alphabet: List[str]
    start: str
    accept: Set[str]
    trans: Dict[Tuple[str, str], str]
    accept_token: Dict[str, str]


def _subset_name(sub: Set[str]) -> str:
    return "{" + ",".join(sorted(sub)) + "}"


def grammar_to_dfa(g: Grammar) -> DFA:
    FINAL = "__FINAL__"
    nfa_acc: Set[str] = {FINAL}
    nfa_trans: Dict[Tuple[str, str], Set[str]] = {}

    for A, rhs_list in g.prods.items():
        for rhs in rhs_list:
            if rhs in {"ε", "eps", "epsilon"}:
                nfa_acc.add(A)
            elif len(rhs) == 1:
                key = (A, rhs[0])
                nfa_trans.setdefault(key, set()).add(FINAL)
            else:
                key = (A, rhs[0])
                nfa_trans.setdefault(key, set()).add(rhs[1:])

    alphabet = sorted(g.terminals)
    subset_to_name: Dict[frozenset[str], str] = {}
    trans: Dict[Tuple[str, str], str] = {}
    states: List[str] = []
    accept: Set[str] = set()

    start_subset = frozenset({g.start})
    start_name = _subset_name(set(start_subset))
    subset_to_name[start_subset] = start_name
    states.append(start_name)
    if any(s in nfa_acc for s in start_subset):
        accept.add(start_name)

    q: deque[frozenset[str]] = deque([start_subset])
    while q:
        cur = q.popleft()
        cur_name = subset_to_name[cur]
        cur_set = set(cur)
        for a in alphabet:
            nxt: Set[str] = set()
            for st in cur_set:
                got = nfa_trans.get((st, a))
                if got:
                    nxt |= got
            nxt_f = frozenset(nxt)
            if nxt_f not in subset_to_name:
                nm = _subset_name(nxt)
                subset_to_name[nxt_f] = nm
                states.append(nm)
                if any(s in nfa_acc for s in nxt_f):
                    accept.add(nm)
                q.append(nxt_f)
            trans[(cur_name, a)] = subset_to_name[nxt_f]

    return DFA(states=states, alphabet=alphabet, start=start_name, accept=accept, trans=trans, accept_token={})


def parse_dfa_text(text: str) -> Tuple[Optional[DFA], List[str]]:
    states: Set[str] = set()
    alphabet: Set[str] = set()
    start = ""
    accept: Set[str] = set()
    accept_token: Dict[str, str] = {}
    trans: Dict[Tuple[str, str], str] = {}
    errors: List[str] = []
    mode = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("alphabet:"):
            for tok in _split_words(line[len("alphabet:") :]):
                if len(tok) != 1:
                    errors.append("alphabet 只允许单字符")
                else:
                    alphabet.add(tok)
            continue
        if line.startswith("states:"):
            for tok in _split_words(line[len("states:") :]):
                states.add(tok)
            continue
        if line.startswith("start:"):
            start = line[len("start:") :].strip()
            continue
        if line == "accept:":
            mode = "accept"
            continue
        if line.startswith("accept:"):
            for tok in _split_words(line[len("accept:") :]):
                accept.add(tok)
            continue
        if line == "transitions:":
            mode = "trans"
            continue
        if line == "end":
            mode = ""
            continue

        if mode == "accept":
            parts = _split_words(line)
            if len(parts) == 1:
                accept.add(parts[0])
            elif len(parts) == 2:
                accept.add(parts[0])
                accept_token[parts[0]] = parts[1]
            else:
                errors.append("accept 行格式错误: " + line)
            continue

        if mode == "trans":
            parts = _split_words(line)
            if len(parts) != 3 or len(parts[1]) != 1:
                errors.append("transition 行格式错误: " + line)
                continue
            frm, ch, to = parts
            trans[(frm, ch)] = to
            continue

        errors.append("无法识别: " + line)

    if not start or start not in states:
        errors.append("开始状态不合法")
    for s in accept:
        if s not in states:
            errors.append("接受状态不在状态集: " + s)
    for (frm, ch), to in trans.items():
        if frm not in states:
            errors.append("转移起点不在状态集: " + frm)
        if to not in states:
            errors.append("转移终点不在状态集: " + to)
        if ch not in alphabet:
            errors.append("转移字符不在字母表: " + ch)

    if errors:
        return None, errors

    return DFA(
        states=list(states),
        alphabet=sorted(alphabet),
        start=start,
        accept=set(accept),
        trans=dict(trans),
        accept_token=dict(accept_token),
    ), []


def dfa_to_text(dfa: DFA) -> str:
    lines: List[str] = []
    lines.append("alphabet: " + " ".join(sorted(dfa.alphabet)))
    lines.append("states: " + " ".join(dfa.states))
    lines.append("start: " + dfa.start)
    lines.append("accept:")
    if dfa.accept_token:
        for s in dfa.accept:
            lines.append("  " + s + " " + dfa.accept_token.get(s, "ACCEPT"))
    else:
        for s in dfa.accept:
            lines.append("  " + s)
    lines.append("transitions:")
    for (frm, ch), to in dfa.trans.items():
        lines.append(f"  {frm} {ch} {to}")
    lines.append("end")
    return "\n".join(lines)


def dfa_to_dot(dfa: DFA) -> str:
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines: List[str] = [
        "digraph DFA {",
        "  rankdir=LR;",
        "  node [shape=point]; start;",
        f"  start -> {q(dfa.start)};",
        "  node [shape=circle];",
    ]
    for st in dfa.states:
        lines.append(f"  {q(st)} [shape={'doublecircle' if st in dfa.accept else 'circle'}];")
    for (frm, ch), to in dfa.trans.items():
        lines.append(f'  {q(frm)} -> {q(to)} [label="{ch}"];')
    lines.append("}")
    return "\n".join(lines)


def tuple_string(dfa: DFA) -> str:
    Q = "{" + ", ".join(dfa.states) + "}"
    S = "{" + ", ".join(sorted(dfa.alphabet)) + "}"
    F = "{" + ", ".join(sorted(dfa.accept)) + "}"
    return f"""M = (Q, Σ, δ, q0, F)
Q = {Q}
Σ = {S}
q0 = {dfa.start}
F = {F}"""


def run_dfa(dfa: DFA, input_str: str) -> Tuple[bool, List[str]]:
    cur = dfa.start
    log = [f"输入串: {'ε' if input_str == '' else input_str}", f"初始状态: {cur}"]
    for ch in input_str:
        if ch not in set(dfa.alphabet):
            return False, log + [f"错误: '{ch}' 不在字母表"]
        to = dfa.trans.get((cur, ch))
        if not to:
            return False, log + [f"错误: 无转移 δ({cur},{ch})"]
        log.append(f"读入 '{ch}': {cur} -> {to}")
        cur = to
    ok = cur in dfa.accept
    log.append(f"最终状态: {cur}")
    log.append(f"判定结果: {'接受' if ok else '拒绝'}")
    return ok, log


def enum_accepted(dfa: DFA, max_len: int) -> List[str]:
    alpha = sorted(dfa.alphabet)
    q: deque[str] = deque([""])
    ans: List[str] = []
    while q:
        s = q.popleft()
        ok, _ = run_dfa(dfa, s)
        if ok:
            ans.append("ε" if s == "" else s)
        if len(s) >= max_len:
            continue
        for a in alpha:
            q.append(s + a)
    return ans


def minimize(dfa: DFA) -> DFA:
    alpha = list(dfa.alphabet)
    reachable: Set[str] = set()
    dq: deque[str] = deque([dfa.start])
    reachable.add(dfa.start)
    while dq:
        s = dq.popleft()
        for a in alpha:
            to = dfa.trans.get((s, a))
            if to and to not in reachable:
                reachable.add(to)
                dq.append(to)

    acc = {s for s in reachable if s in dfa.accept}
    nonacc = {s for s in reachable if s not in dfa.accept}
    P: List[Set[str]] = []
    if acc:
        P.append(set(acc))
    if nonacc:
        P.append(set(nonacc))

    def find_idx(st: str) -> int:
        for i, g in enumerate(P):
            if st in g:
                return i
        return -1

    changed = True
    while changed:
        changed = False
        newP: List[Set[str]] = []
        for group in P:
            bins: Dict[str, Set[str]] = {}
            for st in group:
                sig = "|".join(str(find_idx(dfa.trans.get((st, a), ""))) for a in alpha)
                bins.setdefault(sig, set()).add(st)
            if len(bins) == 1:
                newP.append(group)
            else:
                changed = True
                newP.extend(bins.values())
        P = newP

    mp: Dict[str, str] = {}
    for i, group in enumerate(P):
        for st in group:
            mp[st] = f"S{i}"

    new_states = [f"S{i}" for i in range(len(P))]
    new_start = mp[dfa.start]
    new_accept = {mp[s] for s in acc}
    new_trans: Dict[Tuple[str, str], str] = {}
    for group in P:
        rep = next(iter(group))
        frm = mp[rep]
        for a in alpha:
            to = dfa.trans.get((rep, a))
            if to and to in mp:
                new_trans[(frm, a)] = mp[to]

    return DFA(
        states=new_states,
        alphabet=list(alpha),
        start=new_start,
        accept=new_accept,
        trans=new_trans,
        accept_token={},
    )


def dfa_from_grammar(req: DFAFromGrammarRequest):
    g, errors = parse_grammar(req.grammar_text or "")
    if errors:
        return {"ok": False, "errors": errors}
    dfa = grammar_to_dfa(g)  # type: ignore[arg-type]
    return {
        "ok": True,
        "dfa_text": dfa_to_text(dfa),
        "tuple": tuple_string(dfa),
        "dot": dfa_to_dot(dfa),
    }


def dfa_from_text(req: DFAFromTextRequest):
    dfa, errors = parse_dfa_text(req.dfa_text or "")
    if errors:
        return {"ok": False, "errors": errors}
    assert dfa is not None
    return {
        "ok": True,
        "dfa_text": dfa_to_text(dfa),
        "tuple": tuple_string(dfa),
        "dot": dfa_to_dot(dfa),
    }


def dfa_run(req: DFARunRequest):
    dfa, errors = parse_dfa_text(req.dfa_text or "")
    if errors:
        return {"ok": False, "errors": errors}
    assert dfa is not None
    ok, trace = run_dfa(dfa, req.input_str or "")
    return {"ok": True, "accepted": ok, "trace": "\n".join(trace)}


def dfa_enum(req: DFAEnumRequest):
    dfa, errors = parse_dfa_text(req.dfa_text or "")
    if errors:
        return {"ok": False, "errors": errors}
    assert dfa is not None
    ans = enum_accepted(dfa, int(req.max_len))
    return {"ok": True, "accepted_strings": ans}


def dfa_minimize(req: DFAFromTextRequest):
    dfa, errors = parse_dfa_text(req.dfa_text or "")
    if errors:
        return {"ok": False, "errors": errors}
    assert dfa is not None
    m = minimize(dfa)
    return {
        "ok": True,
        "dfa_text": dfa_to_text(m),
        "tuple": tuple_string(m),
        "dot": dfa_to_dot(m),
    }

