"""
Structured-query tool for the RAG system's exact-number retrieval path.

For questions with one precise, checkable answer, this runs a real pandas
query against the real match data — never a semantic guess, so it can't
hallucinate a number. This is the counterpart to the vector-store search in
build_vector_store.py, which handles narrative "why" questions instead.

Two functions, covering two different question shapes:
    - query_corners_stats: an aggregate over City's matches, optionally
      filtered to one named opponent/season/venue ("what was City's average
      corners at home in 2024/25?"), with an option to also return *which*
      match produced a min/max value.
    - query_corners_rank: which opponent ranks highest/lowest on some stat,
      across all opponents ("who does City concede the most corners to?") —
      a different shape of question, since there's no single opponent to
      filter to; the opponent itself is what's being asked for.

Usage:
    python scripts/structured_query.py
"""

from data_pipeline import load_city_matches

VALID_METRICS = {"average", "total", "min", "max", "count"}
VALID_VENUES = {"home", "away"}
# Sstructured-query stat name -> the actual column in load_city_matches()
STAT_COLUMNS = {
    "corners_won": "corners",
    "corners_conceded": "conceded",
    "total_match_corners": "total_match_corners",
}


def _apply_filters(df, opponent, season, venue):
    if opponent is not None:
        df = df[df['opponent'] == opponent]
    if season is not None:
        df = df[df['Season'] == season]
    if venue is not None:
        if venue not in VALID_VENUES:
            raise ValueError(f"venue must be one of {VALID_VENUES}, got {venue!r}")
        df = df[df['is_home'] == (1 if venue == 'home' else 0)]
    return df


def query_corners_stats(metric, stat="corners_won", opponent=None, season=None, venue=None, include_match=False):
    """Aggregate a per-match stat over City's matches.

    Args:
        metric: one of "average", "total", "min", "max", "count".
        stat: which per-match number to aggregate — "corners_won" (City's
            own, the default and only option before this extension),
            "corners_conceded" (the flip side, by venue), or
            "total_match_corners" (both teams combined).
        opponent / season / venue: optional filters, as before.
        include_match: if True and metric is "max"/"min", also return which
            specific match produced that value (date, opponent, score) —
            without this, a max/min query only ever answers "what's the
            value," not "which game."
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
    if stat not in STAT_COLUMNS:
        raise ValueError(f"stat must be one of {set(STAT_COLUMNS)}, got {stat!r}")

    df = load_city_matches()
    df = _apply_filters(df, opponent, season, venue)

    if df.empty:
        return {
            "metric": metric, "stat": stat, "opponent": opponent, "season": season, "venue": venue,
            "result": None, "n_matches": 0,
            "note": "No matches found for these filters.",
        }

    col = STAT_COLUMNS[stat]
    values = df[col]
    result = {
        "average": lambda: round(values.mean(), 2),
        "total": lambda: int(values.sum()),
        "min": lambda: int(values.min()),
        "max": lambda: int(values.max()),
        "count": lambda: int(len(values)),
    }[metric]()

    out = {
        "metric": metric, "stat": stat, "opponent": opponent, "season": season, "venue": venue,
        "result": result, "n_matches": int(len(df)),
    }

    if include_match and metric in ("max", "min"):
        idx = values.idxmax() if metric == "max" else values.idxmin()
        row = df.loc[idx]
        out["match"] = {
            "date": row['Date'].strftime('%Y-%m-%d'),
            "opponent": row['opponent'],
            "is_home": bool(row['is_home']),
            "score": f"City {row['corners']}-{row['conceded']} {row['opponent']}",  # always City-first
        }

    return out


def query_corners_rank(agg, stat="corners_won", direction="max", season=None, venue=None, min_matches=1):
    """Find which opponent ranks highest/lowest on some stat, across all
    opponents — for questions like "who does City concede the most corners
    to?" where the opponent is the thing being asked for, not a filter.

    Args:
        agg: "total" or "average" — how to combine each opponent's matches.
        stat: "corners_won", "corners_conceded", or "total_match_corners".
        direction: "max" or "min".
        season / venue: optional filters.
        min_matches: exclude opponents with fewer than this many matches, so
            a single fluke match doesn't dominate an "average" ranking.
    """
    if agg not in {"total", "average"}:
        raise ValueError(f"agg must be 'total' or 'average', got {agg!r}")
    if direction not in {"max", "min"}:
        raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
    if stat not in STAT_COLUMNS:
        raise ValueError(f"stat must be one of {set(STAT_COLUMNS)}, got {stat!r}")

    df = load_city_matches()
    if season is not None:
        df = df[df['Season'] == season]
    if venue is not None:
        df = df[df['is_home'] == (1 if venue == 'home' else 0)]

    col = STAT_COLUMNS[stat]
    grouped = df.groupby('opponent')[col].agg(['sum', 'mean', 'count'])
    grouped = grouped[grouped['count'] >= min_matches]

    if grouped.empty:
        return {
            "agg": agg, "stat": stat, "direction": direction, "season": season, "venue": venue,
            "opponent": None, "result": None, "n_matches": 0,
            "note": "No opponents found meeting the filters/min_matches threshold.",
        }

    value_col = 'sum' if agg == 'total' else 'mean'
    idx = grouped[value_col].idxmax() if direction == 'max' else grouped[value_col].idxmin()
    row = grouped.loc[idx]

    return {
        "agg": agg, "stat": stat, "direction": direction, "season": season, "venue": venue,
        "opponent": idx,
        "result": round(row[value_col], 2) if agg == 'average' else int(row[value_col]),
        "n_matches": int(row['count']),
    }


if __name__ == "__main__":
    # The 10 drafted eval questions, verified end-to-end through the actual
    # tool functions rather than by hand — this is the real proof the
    # extension covers what it was built for.
    checks = [
        ("Total corners 2023/24", lambda: query_corners_stats("total", season="2023/24")),
        ("Concede most corners to (opponent)", lambda: query_corners_rank("total", stat="corners_conceded", direction="max")),
        ("Most conceded to any team, total", lambda: query_corners_rank("total", stat="corners_conceded", direction="max")),
        ("Most total corners in a match", lambda: query_corners_stats("max", stat="total_match_corners", include_match=True)),
        ("Least total corners in a match", lambda: query_corners_stats("min", stat="total_match_corners", include_match=True)),
        ("Match where City had most corners", lambda: query_corners_stats("max", stat="corners_won", include_match=True)),
        ("Match where City had fewest corners", lambda: query_corners_stats("min", stat="corners_won", include_match=True)),
        ("Total corners 2024/25", lambda: query_corners_stats("total", season="2024/25")),
        ("Total corners 2025/26", lambda: query_corners_stats("total", season="2025/26")),
        ("Opponent City average fewest corners against", lambda: query_corners_rank("average", stat="corners_won", direction="min")),
    ]

    for label, fn in checks:
        print(f"{label}: {fn()}")
