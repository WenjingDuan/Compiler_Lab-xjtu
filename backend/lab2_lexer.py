from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from backend.lab1_dfa import DFA, parse_dfa_text


class LexRequest(BaseModel):
    source_code: str = Field(default="")
    mode: str = Field(default="default_lexer")  # default_lexer | custom_dfa
    dfa_text: Optional[str] = None


@dataclass(frozen=True)
class LexDFA:
    start: str
    states: Set[str]
    alphabet: Set[str]
    accept: Set[str]
    accept_token: Dict[str, str]
    trans: Dict[Tuple[str, str], str]


def _add(d: LexDFA, frm: str, ch: str, to: str) -> None:
    d.states.add(frm)
    d.states.add(to)
    d.alphabet.add(ch)
    d.trans[(frm, ch)] = to


def _add_range(d: LexDFA, frm: str, l: str, r: str, to: str) -> None:
    for c in range(ord(l), ord(r) + 1):
        _add(d, frm, chr(c), to)


def build_default_lex_dfa() -> LexDFA:
    d = LexDFA(
        start="0",
        states=set(),
        alphabet=set(),
        accept=set(),
        accept_token={},
        trans={},
    )

    def ac(st: str, tp: str) -> None:
        d.accept.add(st)
        d.accept_token[st] = tp

    ac("ID", "ID")
    ac("INT", "INT")
    ac("FLO_FRAC", "FLOAT")
    ac("FLO_SCI", "FLOAT")

    ac("ASSIGN", "ASSIGN")
    ac("PLUS", "PLUS")
    ac("MINUS", "MINUS")
    ac("MUL", "MUL")
    ac("DIV", "DIV")
    ac("DOT", "DOT")
    ac("SEMI", "SEMI")
    ac("CMA", "CMA")
    ac("LT", "LT")
    ac("GT", "GT")
    ac("LE", "LE")
    ac("GE", "GE")
    ac("EQ", "EQ")
    ac("NE", "NE")
    ac("AND", "AND")
    ac("OR", "OR")
    ac("LPA", "LPA")
    ac("RPA", "RPA")
    ac("LBR", "LBR")
    ac("RBR", "RBR")
    ac("LBK", "LBK")
    ac("RBK", "RBK")

    _add_range(d, "0", "a", "z", "ID")
    _add_range(d, "0", "A", "Z", "ID")
    _add(d, "0", "_", "ID")
    _add_range(d, "ID", "a", "z", "ID")
    _add_range(d, "ID", "A", "Z", "ID")
    _add_range(d, "ID", "0", "9", "ID")
    _add(d, "ID", "_", "ID")

    _add_range(d, "0", "0", "9", "INT")
    _add_range(d, "INT", "0", "9", "INT")
    _add(d, "INT", ".", "FLO_DOT")
    _add_range(d, "FLO_DOT", "0", "9", "FLO_FRAC")
    _add_range(d, "FLO_FRAC", "0", "9", "FLO_FRAC")

    _add(d, "INT", "e", "FLO_EXP")
    _add(d, "INT", "E", "FLO_EXP")
    _add(d, "FLO_FRAC", "e", "FLO_EXP")
    _add(d, "FLO_FRAC", "E", "FLO_EXP")
    _add(d, "FLO_EXP", "+", "FLO_SIGN")
    _add(d, "FLO_EXP", "-", "FLO_SIGN")
    _add_range(d, "FLO_EXP", "0", "9", "FLO_SCI")
    _add_range(d, "FLO_SIGN", "0", "9", "FLO_SCI")
    _add_range(d, "FLO_SCI", "0", "9", "FLO_SCI")

    _add(d, "0", "=", "ASSIGN")
    _add(d, "ASSIGN", "=", "EQ")
    _add(d, "0", "+", "PLUS")
    _add(d, "0", "-", "MINUS")
    _add(d, "0", "*", "MUL")
    _add(d, "0", "/", "DIV")
    _add(d, "0", ".", "DOT")
    _add(d, "0", ";", "SEMI")
    _add(d, "0", ",", "CMA")
    _add(d, "0", "<", "LT")
    _add(d, "LT", "=", "LE")
    _add(d, "0", ">", "GT")
    _add(d, "GT", "=", "GE")
    _add(d, "0", "!", "NOT")
    _add(d, "NOT", "=", "NE")
    _add(d, "0", "&", "AND1")
    _add(d, "AND1", "&", "AND")
    _add(d, "0", "|", "OR1")
    _add(d, "OR1", "|", "OR")
    _add(d, "0", "(", "LPA")
    _add(d, "0", ")", "RPA")
    _add(d, "0", "{", "LBR")
    _add(d, "0", "}", "RBR")
    _add(d, "0", "[", "LBK")
    _add(d, "0", "]", "RBK")
    return d


def _dfa_from_custom(dfa_text: str) -> Tuple[Optional[LexDFA], List[str]]:
    dfa, errs = parse_dfa_text(dfa_text)
    if errs:
        return None, errs
    assert dfa is not None
    alpha = set(dfa.alphabet)
    return (
        LexDFA(
            start=dfa.start,
            states=set(dfa.states),
            alphabet=alpha,
            accept=set(dfa.accept),
            accept_token=dict(dfa.accept_token) if dfa.accept_token else {s: "ACCEPT" for s in dfa.accept},
            trans=dict(dfa.trans),
        ),
        [],
    )


def lex_by_dfa(dfa: LexDFA, code: str) -> Tuple[List[str], List[str]]:
    kw = {"void", "int", "float", "double", "char", "if", "else", "while", "for", "return"}
    tokens: List[str] = []
    errors: List[str] = []
    i = 0
    line = 1
    col = 1
    n = len(code)

    while i < n:
        ch = code[i]
        if ch in {" ", "\t", "\r"}:
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue

        cur = dfa.start
        j = i
        last_pos = -1
        last_state = ""
        while j < n:
            to = dfa.trans.get((cur, code[j]))
            if not to:
                break
            cur = to
            if cur in dfa.accept:
                last_pos = j
                last_state = cur
            j += 1

        if last_pos < 0:
            errors.append(f"[LEXICAL_ERROR] 第 {line} 行，第 {col} 列无法识别字符: {ch}")
            i += 1
            col += 1
            continue

        lexeme = code[i : last_pos + 1]
        tp = dfa.accept_token.get(last_state, "TOKEN")
        if tp == "ID" and lexeme in kw:
            tp = lexeme.upper()
        tokens.append(f"<{tp}, {lexeme}> 位置({line},{col})")

        for c in lexeme:
            if c == "\n":
                line += 1
                col = 1
            else:
                col += 1
        i = last_pos + 1

    return tokens, errors


def lex_run(req: LexRequest):
    src = req.source_code or ""
    if not src.strip():
        return {"ok": False, "errors": ["请先输入或读取代码文本。"]}

    if req.mode == "custom_dfa":
        if not req.dfa_text:
            return {"ok": False, "errors": ["custom_dfa 模式下必须提供 dfa_text。"]}
        dfa, errs = _dfa_from_custom(req.dfa_text)
        if errs:
            return {"ok": False, "errors": errs}
        assert dfa is not None
    else:
        dfa = build_default_lex_dfa()

    tokens, errors = lex_by_dfa(dfa, src)
    return {"ok": True, "tokens": tokens, "errors": errors}

