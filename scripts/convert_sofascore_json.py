"""
Convert Sofascore match-stat JSON exports (from the Tampermonkey export script)
into a single CSV matching the column schema used by the existing
football-data.co.uk season files: Date, HomeTeam, AwayTeam, HC, AC, HS, AS.

Usage:
    python scripts/convert_sofascore_json.py data/raw/sofascore_exports/ -o data/raw/pl_26-27_manual.csv

Each input .json file is one match export from the Tampermonkey script.
Name your files with a leading ISO date (e.g. 2026-09-05_city-coventry.json)
so the date gets picked up automatically — otherwise you'll need to fill in
the Date column by hand afterward.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# Sofascore's full team names -> the abbreviated names used in the existing CSVs.
# Extend this as you pull data for teams not yet covered.
TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Coventry City": "Coventry",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}

# Sofascore stat label -> (home column, away column) in the output CSV
STAT_FIELDS = {
    "Corner kicks": ("HC", "AC"),
    "Total shots": ("HS", "AS"),
    "Shots on target": ("HST", "AST"),
    "Fouls": ("HF", "AF"),
    "Yellow cards": ("HY", "AY"),
    "Red cards": ("HR", "AR"),
}

DATE_IN_FILENAME = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_DDMMYYYY = re.compile(r"(\d{2}/\d{2}/\d{4})")


def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name)


def extract_date(payload, path):
    """Prefer a 'date' key in the JSON itself; fall back to a date in the filename.

    Whatever raw string we get (from the page, possibly with a time-of-day and/or
    stray whitespace/newlines attached) is reduced to a bare dd/mm/yyyy, matching
    the date-only format the existing CSVs use — no time, no extra characters.
    """
    raw = (payload.get("date") or "").strip()
    match = DATE_DDMMYYYY.search(raw)
    if match:
        return match.group(1)

    match = DATE_IN_FILENAME.search(path.stem)
    if match:
        year, month, day = match.group(1).split("-")
        return f"{day}/{month}/{year}"  # dd/mm/yyyy, matching the existing CSVs
    return ""


def convert_file(path):
    payload = json.loads(path.read_text())
    stats = payload.get("stats", {})

    row = {
        "Date": extract_date(payload, path),
        "HomeTeam": normalize_team(payload.get("home", "")),
        "AwayTeam": normalize_team(payload.get("away", "")),
    }

    missing = []
    for label, (home_col, away_col) in STAT_FIELDS.items():
        stat = stats.get(label)
        if stat is None:
            missing.append(label)
            row[home_col] = ""
            row[away_col] = ""
        else:
            row[home_col] = stat.get("home", "")
            row[away_col] = stat.get("away", "")

    if not row["Date"]:
        print(f"  ! {path.name}: no date found — fill in manually before using this row")
    if missing:
        print(f"  ! {path.name}: missing stat(s) {missing}")

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder containing exported .json files")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/raw/pl_26-27_manual.csv"))
    args = parser.parse_args()

    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {args.input_dir}")
        return

    print(f"Converting {len(json_files)} file(s):")
    rows = []
    for path in json_files:
        print(f"- {path.name}")
        rows.append(convert_file(path))

    fieldnames = ["Date", "HomeTeam", "AwayTeam"]
    for _, (home_col, away_col) in STAT_FIELDS.items():
        fieldnames += [home_col, away_col]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
