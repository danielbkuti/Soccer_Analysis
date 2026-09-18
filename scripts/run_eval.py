"""
Automated scoring for the RAG system's eval set.

Loads data/processed/eval_set.jsonl (question + known-correct ground truth,
verified independently before being added there) and runs every question
through the real router, comparing actual output to expected — rather than a
human reading printed results and eyeballing them.

Structured questions are scored by exact match on the relevant result
fields (with float tolerance). Semantic questions are scored by retrieval
hit rate: was the expected match's document ID actually among the top
results, not just "does the top result look plausible."

Usage:
    python scripts/run_eval.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from router import route
from build_vector_store import STORE_PATH, MODEL_NAME, COLLECTION_NAME

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval_set.jsonl"


def load_eval_set():
    with EVAL_PATH.open() as f:
        return [json.loads(line) for line in f]


def values_match(expected, actual):
    if isinstance(expected, float) or isinstance(actual, float):
        if actual is None:
            return False
        return abs(float(expected) - float(actual)) < 0.01
    return expected == actual


def score_structured(case, result):
    if result['path'] != 'structured':
        return False, f"expected structured path, got {result['path']}"

    actual = result['result']
    for key, expected_value in case['expected'].items():
        if key == 'match':
            actual_match = actual.get('match') or {}
            for mk, mv in expected_value.items():
                if not values_match(mv, actual_match.get(mk)):
                    return False, f"match.{mk}: expected {mv!r}, got {actual_match.get(mk)!r}"
        elif not values_match(expected_value, actual.get(key)):
            return False, f"{key}: expected {expected_value!r}, got {actual.get(key)!r}"
    return True, "ok"


def score_semantic(case, result):
    if result['path'] != 'semantic':
        return False, f"expected semantic path, got {result['path']}"

    retrieved_ids = [r['id'] for r in result['result']]
    if case['expected_match_id'] in retrieved_ids:
        rank = retrieved_ids.index(case['expected_match_id']) + 1
        return True, f"found at rank {rank} of {len(retrieved_ids)}"
    return False, f"expected {case['expected_match_id']!r} not in top {len(retrieved_ids)}: {retrieved_ids}"


def main():
    cases = load_eval_set()
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    results = []
    for case in cases:
        result = route(case['question'], model=model, collection=collection)
        scorer = score_structured if case['path'] == 'structured' else score_semantic
        passed, detail = scorer(case, result)
        results.append((case, passed, detail))

    print("Per-question results:\n")
    for case, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] ({case['path']}) {case['question']}")
        if not passed:
            print(f"       {detail}")

    n_total = len(results)
    n_passed = sum(1 for _, p, _ in results if p)
    structured = [(c, p) for c, p, _ in results if c['path'] == 'structured']
    semantic = [(c, p) for c, p, _ in results if c['path'] == 'semantic']

    print(f"\n{'=' * 50}")
    print(f"Overall:    {n_passed}/{n_total} ({n_passed / n_total:.0%})")
    if structured:
        n_s = sum(1 for _, p in structured if p)
        print(f"Structured: {n_s}/{len(structured)} ({n_s / len(structured):.0%})")
    if semantic:
        n_sem = sum(1 for _, p in semantic if p)
        print(f"Semantic:   {n_sem}/{len(semantic)} ({n_sem / len(semantic):.0%}) [retrieval hit-rate]")


if __name__ == "__main__":
    main()
