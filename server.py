from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.lab1_dfa import (
    DFAFromGrammarRequest,
    DFAFromTextRequest,
    DFARunRequest,
    DFAEnumRequest,
    dfa_from_grammar,
    dfa_from_text,
    dfa_run,
    dfa_enum,
    dfa_minimize,
)
from backend.lab2_lexer import LexRequest, lex_run
from backend.lab3_lr0 import LR0Request, lr0_run
from backend.lab4_slr1 import SLR1Request, slr1_run

app = FastAPI(title="Compiler Lab Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/api/lab1/dfa/from-grammar")
def api_dfa_from_grammar(req: DFAFromGrammarRequest):
    return dfa_from_grammar(req)


@app.post("/api/lab1/dfa/from-text")
def api_dfa_from_text(req: DFAFromTextRequest):
    return dfa_from_text(req)


@app.post("/api/lab1/dfa/run")
def api_dfa_run(req: DFARunRequest):
    return dfa_run(req)


@app.post("/api/lab1/dfa/enum")
def api_dfa_enum(req: DFAEnumRequest):
    return dfa_enum(req)


@app.post("/api/lab1/dfa/minimize")
def api_dfa_minimize(req: DFAFromTextRequest):
    return dfa_minimize(req)


@app.post("/api/lab2/lex")
def api_lex(req: LexRequest):
    return lex_run(req)


@app.post("/api/lab3/lr0")
def api_lr0(req: LR0Request):
    return lr0_run(req)


@app.post("/api/lab4/slr1")
def api_slr1(req: SLR1Request):
    return slr1_run(req)


app.mount("/static", StaticFiles(directory="."), name="static")

