#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验6 Web API：源程序 -> 四元式。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.lab6_codegen import run_compile


class Lab6Request(BaseModel):
    source_code: str = Field(default="", description="源程序文本")


def lab6_compile(req: Lab6Request) -> dict:
    return run_compile(req.source_code)
