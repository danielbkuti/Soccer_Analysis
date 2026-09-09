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

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_city_matches():
    df1 = pd.read_csv(RAW / "pl_25-26.csv"); df1['Season'] = '2025/26'
    df2 = pd.read_csv(RAW / "pl_24-25.csv"); df2['Season'] = '2024/25'
    df3 = pd.read_csv(RAW / "pl_23-24.csv"); df3['Season'] = '2023/24'
    df4 = pd.read_csv(RAW / "pl_26-27_manual.csv"); df4['Season'] = '2026/27'
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    city_home = df[df['HomeTeam'] == 'Man City'].copy()
    city_away = df[df['AwayTeam'] == 'Man City'].copy()
    city_home['corners'] = city_home['HC']
    city_away['corners'] = city_away['AC']
    city_home['opponent'] = city_home['AwayTeam']
    city_away['opponent'] = city_away['HomeTeam']

    city_home['Date'] = pd.to_datetime(city_home['Date'], dayfirst=True)
    city_away['Date'] = pd.to_datetime(city_away['Date'], dayfirst=True)
    city_home = city_home.sort_values('Date').copy()
    city_away = city_away.sort_values('Date').copy()
    city_home['city_rolling_corners'] = city_home['HC'].shift(1).rolling(5, min_periods=1).mean()
    city_away['city_rolling_corners'] = city_away['AC'].shift(1).rolling(5, min_periods=1).mean()

    city_all = pd.concat([city_home, city_away], ignore_index=True)
    city_all['Date'] = pd.to_datetime(city_all['Date'], dayfirst=True)
    city_all['is_home'] = (city_all['HomeTeam'] == 'Man City').astype(int)

    # league-wide rolling conceded corners/shots, so the opponent's defensive
    # form can be described too — same shift(1)+groupby().transform() pattern
    # as the model's own features.
    home_conceded = df[['Date', 'HomeTeam', 'AC']].copy()
    home_conceded.columns = ['date', 'team', 'corners_conceded']
    away_conceded = df[['Date', 'AwayTeam', 'HC']].copy()
    away_conceded.columns = ['date', 'team', 'corners_conceded']
    all_conceded = pd.concat([home_conceded, away_conceded], ignore_index=True)
    all_conceded['date'] = pd.to_datetime(all_conceded['date'], dayfirst=True)
    all_conceded = all_conceded.sort_values(['team', 'date']).reset_index(drop=True)
    all_conceded['rolling_conceded'] = (
        all_conceded.groupby('team')['corners_conceded']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    home_shots = df[['Date', 'HomeTeam', 'AS']].copy()
    home_shots.columns = ['date', 'team', 'shots_conceded']
    away_shots = df[['Date', 'AwayTeam', 'HS']].copy()
    away_shots.columns = ['date', 'team', 'shots_conceded']
    all_shots = pd.concat([home_shots, away_shots], ignore_index=True)
    all_shots['date'] = pd.to_datetime(all_shots['date'], dayfirst=True)
    all_shots = all_shots.sort_values(['team', 'date']).reset_index(drop=True)
    all_shots['rolling_shots_conceded'] = (
        all_shots.groupby('team')['shots_conceded']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    ac_clean = all_conceded.dropna(subset=['rolling_conceded'])
    as_clean = all_shots.dropna(subset=['rolling_shots_conceded'])
    city_all = city_all.merge(ac_clean[['date', 'team', 'rolling_conceded']],
                               left_on=['Date', 'opponent'], right_on=['date', 'team'], how='left')
    city_all = city_all.merge(as_clean[['date', 'team', 'rolling_shots_conceded']],
                               left_on=['Date', 'opponent'], right_on=['date', 'team'], how='left')

    city_all = city_all.dropna(subset=['city_rolling_corners', 'rolling_conceded', 'rolling_shots_conceded'])
    return city_all.sort_values('Date').reset_index(drop=True)


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
