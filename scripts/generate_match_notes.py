"""
Generate one natural-language "analyst note" per Man City match, built from the
already-engineered rolling features (city_rolling_corners, rolling_conceded,
rolling_shots_conceded) plus each match's real recorded outcome.

This is the RAG project's corpus: no model inference happens here, only
historical facts and features that already exist and have been validated in
notebooks/corners_model.ipynb.

Every match gets the same core sentences (date, venue, opponent, rolling form,
actual corners result — uniformly available for all matches, old and new).
Richer sentences (shots, cards, fouls, half-time/full-time score, referee) are
added only when that match's data actually has them — the 6 manually-sourced
2026/27 matches are missing final score, HT score, and referee, so those
matches get shorter, plainer notes rather than fabricated detail.

Usage:
    python scripts/generate_match_notes.py -o data/processed/match_notes.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline import load_city_matches


def city_and_opponent(row, home_col, away_col):
    """For a home/away-indexed column pair, return (city's value, opponent's value).

    Returns (None, None) if either side is missing for this match — this is how
    the 6 manually-sourced 2026/27 matches (no final score, HT score, referee)
    fall back to shorter notes instead of the script guessing or crashing.
    """
    if home_col not in row or away_col not in row:
        return None, None
    if pd.isna(row[home_col]) or pd.isna(row[away_col]):
        return None, None
    if row['is_home']:
        return row[home_col], row[away_col]
    return row[away_col], row[home_col]


def build_note(row):
    venue = "hosted" if row['is_home'] else "traveled to"
    date_str = row['Date'].strftime('%d %B %Y')

    diff = row['corners'] - row['city_rolling_corners']
    if diff > 0.5:
        comparison = "above their recent average"
    elif diff < -0.5:
        comparison = "below their recent average"
    else:
        comparison = "in line with their recent average"

    sentences = [
        f"On {date_str}, Man City {venue} {row['opponent']}.",
        f"City's rolling 5-match average coming in was {row['city_rolling_corners']:.2f} corners won.",
        f"{row['opponent']} had been conceding an average of {row['rolling_conceded']:.2f} corners "
        f"and {row['rolling_shots_conceded']:.2f} shots per game recently.",
        f"City won {int(row['corners'])} corners — {comparison}.",
    ]

    city_shots, opp_shots = city_and_opponent(row, 'HS', 'AS')
    if city_shots is not None:
        sentences.append(f"City had {int(city_shots)} total shots to {row['opponent']}'s {int(opp_shots)}.")

    city_sot, opp_sot = city_and_opponent(row, 'HST', 'AST')
    if city_sot is not None:
        sentences.append(f"{int(city_sot)} of City's shots were on target, against {int(opp_sot)} for {row['opponent']}.")

    city_fouls, opp_fouls = city_and_opponent(row, 'HF', 'AF')
    if city_fouls is not None:
        sentences.append(f"City committed {int(city_fouls)} fouls to {row['opponent']}'s {int(opp_fouls)}.")

    city_yellow, opp_yellow = city_and_opponent(row, 'HY', 'AY')
    if city_yellow is not None:
        sentences.append(f"City picked up {int(city_yellow)} yellow card(s); {row['opponent']} picked up {int(opp_yellow)}.")

    city_red, opp_red = city_and_opponent(row, 'HR', 'AR')
    if city_red is not None and (city_red > 0 or opp_red > 0):
        sentences.append(f"City had {int(city_red)} red card(s); {row['opponent']} had {int(opp_red)}.")

    city_ht, opp_ht = city_and_opponent(row, 'HTHG', 'HTAG')
    if city_ht is not None:
        sentences.append(f"The score at half time was City {int(city_ht)}-{int(opp_ht)} {row['opponent']}.")

    city_ft, opp_ft = city_and_opponent(row, 'FTHG', 'FTAG')
    if city_ft is not None:
        result = "won" if city_ft > opp_ft else ("lost" if city_ft < opp_ft else "drew")
        sentences.append(f"City {result} the match {int(city_ft)}-{int(opp_ft)}.")

    if pd.notna(row.get('Referee')):
        sentences.append(f"{row['Referee']} was the referee.")

    return " ".join(sentences)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed/match_notes.jsonl"))
    args = parser.parse_args()

    city_all = load_city_matches()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        for _, row in city_all.iterrows():
            doc = {
                "id": f"match_{row['Date'].strftime('%Y%m%d')}_{row['opponent'].replace(' ', '_')}",
                "date": row['Date'].strftime('%Y-%m-%d'),
                "season": row['Season'],
                "opponent": row['opponent'],
                "is_home": bool(row['is_home']),
                "corners": int(row['corners']),
                "text": build_note(row),
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(city_all)} match notes to {args.output}")


if __name__ == "__main__":
    main()
