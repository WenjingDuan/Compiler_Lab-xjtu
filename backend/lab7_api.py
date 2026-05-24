#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验7 Web API：源程序 -> 四元式与目标汇编。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from backend.lab7_codegen import run_compile


class Lab7Request(BaseModel):
    source_code: str = Field(default="", description="源程序文本")


def lab7_compile(req: Lab7Request) -> dict:
    return run_compile(req.source_code)
