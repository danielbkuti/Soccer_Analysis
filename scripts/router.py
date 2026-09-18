"""
Keyword-heuristic router for the RAG system.

Deliberately not an LLM call — plain rule-based classification and parameter
extraction. Takes a raw English question and decides which of three
structured-query shapes applies, or falls back to semantic search:

    - "aggregate"   -> query_corners_stats(): a filtered average/total/etc.
    - "which_match" -> query_corners_stats(..., include_match=True): a
      min/max question where the answer is a specific game, not just a value.
    - "rank"        -> query_corners_rank(): "who" questions, where an
      opponent is the answer rather than a filter.
    - "semantic"    -> the persisted Chroma vector store, for narrative
      questions that don't reduce to a single number.

Usage:
    python scripts/router.py
"""

import re

import chromadb
from sentence_transformers import SentenceTransformer

from data_pipeline import load_city_matches
from structured_query import query_corners_stats, query_corners_rank
from build_vector_store import STORE_PATH, MODEL_NAME, COLLECTION_NAME

METRIC_CUES = {
    "average": "average", "avg": "average", "mean": "average",
    "total": "total",
    "most": "max", "highest": "max", "max": "max",
    "least": "min", "fewest": "min", "lowest": "min",
    "how many matches": "count", "count": "count",
}
# "min" deliberately excluded — as a bare 3-letter substring it false-matches
# inside ordinary words like "dominant" ("do-MIN-ant"). "least"/"fewest"/
# "lowest" already cover the same meaning without that collision risk.
DIRECTION_CUES = {
    "most": "max", "highest": "max", "max": "max",
    "least": "min", "fewest": "min", "lowest": "min",
}
RANK_TRIGGERS = ["who ", "which opponent", "which team", "any team", "any opponent"]
SEMANTIC_CUES = ["why", "explain", "describe", "what happened", "struggled", "dominant", "how did"]
SEASON_PATTERN = re.compile(r"\d{4}/\d{2}")
STAT_LABELS = {
    "corners_won": "corners won",
    "corners_conceded": "corners conceded",
    "total_match_corners": "total match corners",
}

_KNOWN_OPPONENTS = None  # cached on first use, not at import time


def known_opponents():
    global _KNOWN_OPPONENTS
    if _KNOWN_OPPONENTS is None:
        _KNOWN_OPPONENTS = sorted(load_city_matches()['opponent'].unique(), key=len, reverse=True)
    return _KNOWN_OPPONENTS


def detect_stat(q):
    if "concede" in q or "conceded" in q or "conceding" in q:
        return "corners_conceded"
    if "involving" in q or "combined" in q or "between both teams" in q:
        return "total_match_corners"
    return "corners_won"


def extract_season(question):
    m = SEASON_PATTERN.search(question)
    return m.group(0) if m else None


def extract_venue(q):
    if "home" in q:
        return "home"
    if "away" in q:
        return "away"
    return None


def extract_opponent(q):
    # Longest names first, so "Nott'm Forest" matches before a shorter
    # accidental substring would.
    for name in known_opponents():
        if name.lower() in q:
            return name
    return None


def classify(question):
    q = question.lower()
    if "which game" in q or "which match" in q:
        return "which_match"
    if "the match where" in q and any(w in q for w in DIRECTION_CUES):
        return "which_match"
    if any(t in q for t in RANK_TRIGGERS) and any(w in q for w in DIRECTION_CUES):
        return "rank"
    if any(cue in q for cue in METRIC_CUES):
        return "aggregate"
    if any(cue in q for cue in SEMANTIC_CUES):
        return "semantic"
    return "semantic"  # default: don't force a number out of a question that isn't asking for one


def extract_aggregate_params(question):
    q = question.lower()
    metric = next((v for k, v in METRIC_CUES.items() if k in q), "average")
    return {
        "metric": metric,
        "stat": detect_stat(q),
        "opponent": extract_opponent(q),
        "season": extract_season(question),
        "venue": extract_venue(q),
    }


def extract_which_match_params(question):
    q = question.lower()
    direction = next((v for k, v in DIRECTION_CUES.items() if k in q), "max")
    return {
        "metric": direction,
        "stat": detect_stat(q),
        "opponent": extract_opponent(q),
        "season": extract_season(question),
        "venue": extract_venue(q),
        "include_match": True,
    }


def extract_rank_params(question):
    q = question.lower()
    direction = next((v for k, v in DIRECTION_CUES.items() if k in q), "max")
    agg = "average" if any(w in q for w in ("average", "avg", "mean")) else "total"
    return {
        "agg": agg,
        "stat": detect_stat(q),
        "direction": direction,
        "season": extract_season(question),
        "venue": extract_venue(q),
    }


def semantic_search(question, model, collection, n_results=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return [
        {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def route(question, model=None, collection=None):
    mode = classify(question)

    if mode == "aggregate":
        params = extract_aggregate_params(question)
        return {"path": "structured", "mode": mode, "params": params, "result": query_corners_stats(**params)}

    if mode == "which_match":
        params = extract_which_match_params(question)
        return {"path": "structured", "mode": mode, "params": params, "result": query_corners_stats(**params)}

    if mode == "rank":
        params = extract_rank_params(question)
        return {"path": "structured", "mode": mode, "params": params, "result": query_corners_rank(**params)}

    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    if collection is None:
        client = chromadb.PersistentClient(path=str(STORE_PATH))
        collection = client.get_collection(COLLECTION_NAME)

    return {"path": "semantic", "mode": "semantic", "result": semantic_search(question, model, collection)}


def format_structured_answer(mode, params, result):
    if result.get('result') is None:
        return result.get('note', 'No matches found for those filters.')

    stat_label = STAT_LABELS[result['stat']]

    if mode == 'rank':
        return (f"{result['opponent']} — {result['agg']} of {result['result']} {stat_label} "
                f"({result['n_matches']} matches), the {result['direction']} among opponents faced.")

    if mode == 'which_match':
        m = result['match']
        return f"{m['score']} on {m['date']} — {result['result']} {stat_label}."

    clauses = []
    if params.get('opponent'):
        clauses.append(f"against {params['opponent']}")
    if params.get('season'):
        clauses.append(f"in {params['season']}")
    if params.get('venue'):
        clauses.append(f"at {params['venue']}")
    context = (" " + " ".join(clauses)) if clauses else ""

    val, n = result['result'], result['n_matches']
    plural = "es" if n != 1 else ""
    templates = {
        "average": f"City averaged {val} {stat_label}{context} ({n} match{plural}).",
        "total": f"City's total {stat_label}{context} was {val} ({n} match{plural}).",
        "max": f"The most {stat_label} in a single match{context} is {val} (across {n} match{plural}).",
        "min": f"The fewest {stat_label} in a single match{context} is {val} (across {n} match{plural}).",
        "count": f"City have played {val} match{'es' if val != 1 else ''}{context}.",
    }
    return templates[result['metric']]


def format_semantic_answer(results):
    lines = ["Here's what the corpus has on that:"]
    for r in results:
        lines.append(f"\n- ({r['metadata']['date']} vs {r['metadata']['opponent']}) {r['text']}")
    return "\n".join(lines)


def answer(question, model=None, collection=None):
    """The plain-English version of route() — same routing/retrieval underneath,
    formatted as something a person would actually want to read."""
    result = route(question, model=model, collection=collection)
    if result['path'] == 'structured':
        return format_structured_answer(result['mode'], result.get('params', {}), result['result'])
    return format_semantic_answer(result['result'])


def print_result(q, result):
    print(f"\nQ: {q}")
    print(f"  path: {result['path']} ({result['mode']})")
    if result['path'] == 'structured':
        print(f"  {format_structured_answer(result['mode'], result.get('params', {}), result['result'])}")
    else:
        print(f"  {format_semantic_answer(result['result'])}")


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print("Fixed regression checks (original 4):")
    for q in [
        "What was City's average corners at home in 2024/25?",
        "What's the most corners City have won against Arsenal?",
        "Why did City struggle for corners against Chelsea?",
        "How many matches has City played against Man United?",
    ]:
        print_result(q, route(q, model=model, collection=collection))

    print("\nNew mode checks (the 10 drafted eval questions):")
    for q in [
        "What was City's total corners in 2023/24?",
        "Who did City concede the most corners to overall?",
        "Most corners City have conceded against any team in total",
        "Which game did the most corners happen involving City?",
        "Which game did the least corners happen involving City?",
        "Which game did City have the most corners?",
        "Which game did City have the least corners?",
        "What was City's total corners in 2024/25?",
        "What was City's total corners in 2025/26?",
        "Who does City average the least corners against?",
    ]:
        print_result(q, route(q, model=model, collection=collection))

    print("\n" + "=" * 60)
    print("Type your own question (blank line to quit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print_result(q, route(q, model=model, collection=collection))
