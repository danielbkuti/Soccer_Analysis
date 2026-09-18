"""
Shared data-loading pipeline for the corners project's scripts.

Reads the 3 football-data.co.uk season files plus the manually-sourced
2026/27 CSV, and rebuilds Man City's full match history with the same
leakage-safe rolling features used in notebooks/corners_model.ipynb
(city_rolling_corners, rolling_conceded, rolling_shots_conceded).

Single source of truth for this pipeline — generate_match_notes.py,
structured_query.py, and anything else that needs City's engineered match
data should import load_city_matches() from here rather than re-deriving it.
"""

from pathlib import Path

import numpy as np
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

    # league-wide rolling conceded corners/shots, so opponents' defensive
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

    # Per-match derived stats, for structured queries beyond "corners City
    # won" — corners City conceded (the flip side of HC/AC depending on
    # venue), and the match's combined total.
    city_all['conceded'] = np.where(city_all['is_home'] == 1, city_all['AC'], city_all['HC'])
    city_all['total_match_corners'] = city_all['HC'] + city_all['AC']

    city_all = city_all.dropna(subset=['city_rolling_corners', 'rolling_conceded', 'rolling_shots_conceded'])
    return city_all.sort_values('Date').reset_index(drop=True)
