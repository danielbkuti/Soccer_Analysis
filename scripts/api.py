"""
FastAPI service exposing the RAG router as an HTTP endpoint.

Wraps router.route() — no new RAG logic lives here, this is purely a new way
to reach the existing system (over HTTP) instead of only from a Python
script run by hand.

Run it:
    uvicorn api:app --reload --port 8000        (from inside scripts/)

Then either open http://127.0.0.1:8000/docs for the interactive Swagger UI,
or:
    curl -X POST http://127.0.0.1:8000/ask \\
        -H "Content-Type: application/json" \\
        -d '{"question": "What was City'"'"'s average corners at home in 2024/25?"}'
"""

from contextlib import asynccontextmanager
from typing import Any

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from build_vector_store import STORE_PATH, MODEL_NAME, COLLECTION_NAME
from router import route, format_structured_answer, format_semantic_answer

# Loaded once at startup, not per-request — embedding the model / opening the
# vector store on every single question would be needlessly slow.
_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _resources["model"] = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    _resources["collection"] = client.get_collection(COLLECTION_NAME)
    yield
    _resources.clear()


app = FastAPI(
    title="Soccer Analysis RAG API",
    description="Ask questions about Man City's corners, in plain English.",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    question: str
    answer: str
    path: str
    mode: str
    raw: Any


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    result = route(payload.question, model=_resources["model"], collection=_resources["collection"])

    if result["path"] == "structured":
        text = format_structured_answer(result["mode"], result.get("params", {}), result["result"])
    else:
        text = format_semantic_answer(result["result"])

    return Answer(
        question=payload.question,
        answer=text,
        path=result["path"],
        mode=result["mode"],
        raw=result["result"],
    )
