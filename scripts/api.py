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
from fastapi.responses import HTMLResponse
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


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Man City Corners Q&amp;A</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 640px;
         margin: 60px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  p.sub { color: #666; margin-top: 0; }
  input { width: 100%; padding: 10px; font-size: 15px; box-sizing: border-box;
          border: 1px solid #ccc; border-radius: 6px; }
  button { margin-top: 10px; padding: 10px 18px; font-size: 15px; background: #00968a;
           color: white; border: none; border-radius: 6px; cursor: pointer; }
  button:disabled { opacity: 0.6; cursor: default; }
  #result { margin-top: 24px; white-space: pre-wrap; line-height: 1.5; }
  .badge { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
           background: #eee; padding: 2px 8px; border-radius: 10px; margin-bottom: 8px; color: #555; }
  .examples { margin-top: 16px; font-size: 13px; color: #777; }
  .examples span { cursor: pointer; text-decoration: underline; margin-right: 12px; }
</style>
</head>
<body>
  <h1>Man City Corners Q&amp;A</h1>
  <p class="sub">Ask about corners in Man City's Premier League matches.</p>
  <input id="q" placeholder="e.g. Who does City average the least corners against?" autofocus />
  <button id="ask">Ask</button>
  <div class="examples">
    Try: <span onclick="setQ(this)">What was City's average corners at home in 2024/25?</span>
    <span onclick="setQ(this)">Why did City struggle for corners against Chelsea?</span>
  </div>
  <div id="result"></div>

<script>
const input = document.getElementById('q');
const btn = document.getElementById('ask');
const result = document.getElementById('result');

function setQ(el) { input.value = el.textContent; ask(); }

async function ask() {
  const question = input.value.trim();
  if (!question) return;
  btn.disabled = true;
  result.textContent = 'Thinking...';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const data = await res.json();
    result.innerHTML = '<div class="badge">' + data.path + ' / ' + data.mode + '</div><div>' +
      data.answer.replace(/\\n/g, '<br>') + '</div>';
  } catch (e) {
    result.textContent = 'Error: ' + e;
  }
  btn.disabled = false;
}

btn.addEventListener('click', ask);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


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
