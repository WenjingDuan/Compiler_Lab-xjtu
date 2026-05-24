from __future__ import annotations

from pydantic import BaseModel, Field

from backend.lab5_semantic import run_semantic_analysis


class SLR5Request(BaseModel):
    grammar_text: str = Field(default="")
    token_text: str = Field(default="")


def slr5_run(req: SLR5Request) -> dict:
    grammar = (req.grammar_text or "").strip()
    tokens = (req.token_text or "").strip()
    if not grammar:
        return {"ok": False, "errors": ["请输入文法（含 @动作 的产生式）。"]}
    if not tokens:
        return {"ok": False, "errors": ["请输入 Token 流（每行 <类型, 词素>）。"]}

    result = run_semantic_analysis(req.grammar_text, req.token_text)
    if not result.get("ok"):
        return result

    warnings = result.pop("grammar_warnings", None)
    if warnings:
        result["warnings"] = warnings
    return result
