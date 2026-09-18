# Premier League Analytics — Corners Prediction & RAG Q&A

Two complete, independent projects built over the same Manchester City match data:

1. **A corners prediction model** — Poisson regression that predicts how many corners City will win in a fixture, and, more importantly, knows how much to trust its own output.
2. **A RAG Q&A system** — a hybrid retrieval service that answers natural-language questions about that data, routing exact-number questions to real pandas queries so it *can't* hallucinate a statistic.

The first is applied ML on a genuinely small, noisy dataset. The second is AI engineering — retrieval, routing, eval design, and a service around it.

**Stack:** Python · pandas · PyTorch · ChromaDB · sentence-transformers · FastAPI · Jupyter

| | Project | What it demonstrates | Headline result |
|---|---|---|---|
| **1** | [Corners prediction model](#1--corners-prediction-model) | Leakage-safe feature engineering, hypothesis-driven debugging, calibrated uncertainty | R² 0.026 → 0.064; instability traced to its real cause and shipped as an honest range |
| **2** | [RAG Q&A system](#2--rag-qa-system) | Hybrid retrieval architecture, router design, automated eval | **18/20 (90%)** — 14/14 exact-number, 4/6 semantic |

> **If you only read one section**, read [Stress-testing, not just reporting a number](#stress-testing-not-just-reporting-a-number). A model output that looked wrong got investigated with two competing hypotheses — one rejected with evidence, one confirmed by measurement — instead of being accepted or quietly discarded.

---

## 1 · Corners Prediction Model

Predicts Manchester City's corners won in an upcoming Premier League fixture, from three-plus seasons of match data.

**The honest framing:** corners are high-variance (std ≈ 3.5 across City's matches) and low-signal relative to what standard match-result data exposes. The goal was never a highly accurate predictor — it was a well-validated one that reports its own limitations.

### Data & features

Three seasons from [football-data.co.uk](https://www.football-data.co.uk) (2023/24–2025/26) plus 6 manually-sourced current-season (2026/27) matches, collected when the primary source went down mid-project.

Four features, all built with `.shift(1).rolling(5, min_periods=1)` so **no match ever sees its own outcome**:

| Feature | What it captures |
|---|---|
| `city_rolling_corners` | City's own 5-match rolling average, corners won |
| `rolling_conceded` | Opponent's 5-match rolling average, corners conceded — built league-wide, not just from City fixtures |
| `rolling_shots_conceded` | Same idea, for shots |
| `is_home` | Venue — City average 7.4 corners at home vs. 6.3 away |

One feature was deliberately **not** built: half-time score. It looks like a natural proxy for match state driving second-half corner patterns, but a match's own `HTHG`/`HTAG` doesn't exist at prediction time — a fixture predictor runs pre-kickoff. Scoped, reasoned about, and dropped rather than silently used.

### Model

```python
nn.Sequential(nn.Linear(4, 1), nn.Softplus())   # softplus keeps the Poisson rate positive
criterion = nn.PoissonNLLLoss(log_input=False)
optimizer = Adam(lr=0.01)
```

Deliberately minimal — appropriate for ~114 training rows, not a place to reach for a deeper network.

### Results

| Feature set | Test R² |
|---|---|
| 3 features (baseline) | 0.026 |
| + opponent shots-conceded | 0.062 |
| + current-season data | 0.064 |

Test MSE beat the "always predict the training mean" baseline at every stage (9.5–9.7 vs. 10.4). R² stayed low throughout — expected given the target's natural variance, not a failure of the features.

### Stress-testing, not just reporting a number

The substantive part of this project. A first real-world prediction — City's corners away at Manchester United — returned **11.86**, which looked implausibly high next to the bookmaker's *total match* corners line of 10.5. Two competing explanations were tested directly:

**Hypothesis 1 — stale or regime-mixed data.** Both clubs had recent managerial and squad changes; rolling features computed across a blended pre/post-change window could plausibly mislead. Tested by filtering Manchester United's rolling window to strictly post-change matches — which produced **identical** numbers, because the change (13 Jan 2026) was old enough that the existing 5-match window was already entirely post-change. City's side showed the same pattern. **Rejected**, with the reasoning kept on record.

**Hypothesis 2 — model instability.** Retrained the identical architecture 10 times, different random seeds, same data, same fixture:

```
mean = 11.48    std = 2.43    range = (6.85, 13.64)
```

**Confirmed.** A ~7-corner spread from nothing but weight initialization, on ~114 training rows. Not stale data — pure small-sample training variance.

**What shipped:** not a point estimate, but *City in the **7–14** range*, with the uncertainty attributed to its actual source. A more useful and more defensible finding than either accepting 11.86 or dismissing it.

📄 Full write-up: **[docs/corners-model-writeup.md](docs/corners-model-writeup.md)** · Notebook: [notebooks/corners_model.ipynb](notebooks/corners_model.ipynb)

---

## 2 · RAG Q&A System

A question-answering service over the same match data. Ask it plain English; it decides how to answer.

```
data/raw/*.csv  (4 seasons)
        │
        ▼
scripts/data_pipeline.py ················ leakage-safe rolling features (single source of truth)
        │
        ├──────────────────────────▶ scripts/structured_query.py ──────┐  exact pandas queries
        │                                                               │
        └──▶ scripts/generate_match_notes.py                            ├──▶ router.py ──▶ api.py
             114 natural-language "analyst notes"                       │    (classify)    FastAPI + web UI
                        │                                               │
                        ▼                                               │
             scripts/build_vector_store.py ─────────────────────────────┘  semantic search
             ChromaDB + all-MiniLM-L6-v2 (local embeddings)
```

### The core design decision: hybrid routing

A pure vector-search RAG will happily return a *plausible-sounding wrong number*. So questions with one precise, checkable answer never touch the embeddings — [`router.py`](scripts/router.py) classifies each question and dispatches it:

| Path | Question shape | Answered by |
|---|---|---|
| `aggregate` | "City's average corners at home in 2024/25?" | Real pandas query, filtered |
| `which_match` | "Which game did City win the most corners?" | Same, returning the specific match |
| `rank` | "Who does City concede the most corners to?" | Ranked across all opponents — the opponent *is* the answer |
| `semantic` | "Why did City struggle for corners against Chelsea?" | ChromaDB vector search over the notes corpus |

The router is **rule-based on purpose** — no LLM call, no API key, fully deterministic and free to run. Its known limitation (keyword heuristics are brittle to unusual phrasing) is documented rather than hidden, with LLM tool-calling scoped as the eventual fix.

### Evaluation

Scored by [`run_eval.py`](scripts/run_eval.py) against 20 questions with independently-verified ground truth — automated comparison, not a human eyeballing printed output. Structured questions score on exact match of result fields (with float tolerance); semantic questions score on **retrieval hit-rate** (was the correct document actually in the top 3), not on whether the top result merely looks plausible.

```
Overall:    18/20 (90%)
Structured: 14/14 (100%)
Semantic:    4/6  (67%)  [retrieval hit-rate]
```

**Both semantic misses are the same known limitation, diagnosed rather than papered over:** dense embeddings don't do numeric or superlative reasoning. Asked for the match where City "got shut out on corners" (i.e. zero), retrieval returns thematically similar match notes but can't rank on the quantity; asked for a match where City were "dominant with plenty of corners" against Chelsea, it correctly narrows to Chelsea fixtures but can't order them by magnitude. That's a property of the embedding model, not a bug in the pipeline — and the fix is a routing change, not a retrieval tweak.

### Running it

```bash
cd scripts && uvicorn api:app --reload --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Minimal web UI — ask a question, see the answer plus a badge showing *which* retrieval path served it |
| `GET /docs` | Interactive Swagger UI |
| `GET /health` | Health check |
| `POST /ask` | `{"question": "..."}` → answer + path + mode + raw result fields |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many corners did Man City average at home?"}'
# {"answer":"City averaged 7.53 corners won at home (57 matches).","path":"structured","mode":"aggregate",...}
```

---

## Quick start

```bash
conda env create -f environment.yml
conda activate soccer-analysis
```

Or with a plain virtualenv — same pins, no conda required:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Raw match data is **not** committed (see [Data sources](#data-sources--attribution)). To reproduce from scratch:

```bash
# 1. download the England PL CSVs from football-data.co.uk into data/raw/
# 2. regenerate the RAG corpus and vector store
python scripts/generate_match_notes.py -o data/processed/match_notes.jsonl
python scripts/build_vector_store.py
# 3. verify everything works
python scripts/run_eval.py
```

---

## Repository structure

```
├── notebooks/
│   ├── corners_model.ipynb        # corners model + fixture prediction interface
│   └── ederson-eda.ipynb          # earlier first-shot analysis
├── scripts/
│   ├── data_pipeline.py           # shared loader + rolling features
│   ├── generate_match_notes.py    # RAG corpus generation
│   ├── build_vector_store.py      # embeddings → ChromaDB
│   ├── structured_query.py        # exact-answer pandas tool
│   ├── router.py                  # question classification + dispatch
│   ├── run_eval.py                # automated eval scoring
│   ├── api.py                     # FastAPI service + web UI
│   └── convert_sofascore_json.py  # manual-export → CSV schema
├── data/processed/
│   ├── match_notes.jsonl          # 114-note RAG corpus (committed)
│   └── eval_set.jsonl             # 20-question eval set
├── docs/
│   ├── corners-model-writeup.md   # full modeling narrative
│   └── model-expansion-handoff.md # scoping for the next phase
└── outputs/                       # figures + reports from the Ederson analysis
```

---

## Earlier work — Ederson first-shot concession analysis

The project this repo started as, and the foundation the corners work grew out of. Kept because the analysis is sound and the question is a good one; no longer the headline.

**The question:** in matches where a dominant team suppresses opponent shot volume, aggregate save percentage can hide *when* a goalkeeper is vulnerable. Does the **first shot faced** carry disproportionate risk?

**Scope:** Ederson's Premier League matches, 2017-18 through 2024-25 — 276 appearances across 8 seasons.

**Findings** (2017-18 detail, the season examined most closely):

- Matches with ≥1 shot faced: **29**
- Matches where the first shot became a goal: **9** → **31.0% first-shot concession rate**
- Among matches where he conceded at all, share where the *first* shot produced the goal: **52.9%**

The multi-season extension adds season-over-season concession rate, expected-vs-actual goals conceded (shot-stopping over/under-performance), and a logistic regression on first-shot goal probability — feature importances in [outputs/figures/](outputs/figures/), full match-by-match results in [outputs/reports/](outputs/reports/).

**Pipeline:** match data via the `soccerdata` library plus a custom scraper for shot-level events from [UnderStat](https://understat.com), standardized and validated to CSV before analysis. Stack: pandas, matplotlib, seaborn, PostgreSQL, Jupyter.

---

## Data sources & attribution

This repo contains **code that processes data, not redistributed data**. `data/raw/*` is gitignored.

- **[football-data.co.uk](https://www.football-data.co.uk)** — Premier League match data, 2023/24–2025/26. Free for personal/non-commercial use. Download the England Premier League CSVs directly to reproduce.
- **[Sofascore](https://www.sofascore.com)** — a small number of 2026/27 matches, collected **manually**, one page at a time via a personal Tampermonkey export script — not automated bulk scraping — and converted to the same schema by [scripts/convert_sofascore_json.py](scripts/convert_sofascore_json.py).
- **[UnderStat](https://understat.com)** / `soccerdata` — shot-level events for the earlier Ederson analysis.

[data/processed/match_notes.jsonl](data/processed/match_notes.jsonl) **is** committed — it's original generated text (one analyst note per match, produced from engineered features and real recorded outcomes), not a redistribution of any source's data.

---

## Roadmap

Scoped in detail in [docs/model-expansion-handoff.md](docs/model-expansion-handoff.md) — deliberately reasoned through before starting, not sprint-planned:

- **Expand from City-only to all 20 PL teams** — roughly 20× the training data, the single highest-leverage fix for the instability measured above. Requires generalizing the City-hardcoded pipeline and adding a team-identity signal.
- **Richer features** — league position, head-to-head history, referee, and a signal for managerial/tactical regime changes (a demonstrated error source).
- **A different model once data volume justifies it** — gradient boosting or a proper Poisson GLM, with time-based cross-validation.
- **Replace the router's keyword heuristics with LLM tool-calling** — fixes the *class* of brittleness rather than patching individual phrasings.
- **Non-PL competitions** (UCL, FA Cup, EFL Cup) — a separate data-sourcing problem; lowest priority.

---

## Author

**Tirenioluwa Daniel Biodun-Kuti**
[LinkedIn](https://www.linkedin.com/in/tbk022/)
