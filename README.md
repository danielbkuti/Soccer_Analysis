# Ederson First-Shot Concession Analysis

## Overview

This project analyzes Ederson's shot-stopping exposure during Manchester City's 2017–18 Premier League season, with a focus on **first-shot vulnerability** in low-shot environments.

Modern Manchester City matches often feature limited opponent shot volume. In these conditions, the **first shot on target** carries disproportionate match impact. This analysis builds a reproducible event-level pipeline to quantify that risk.

---

## Key Findings

- Matches with ≥1 shot faced: **29**
- Matches where first shot resulted in a goal: **9**
- **First-shot concession rate: 31.0%**

### Conditional Insight

Among matches in which Ederson conceded at least once:

- Share where the first shot produced the goal: **52.9%**

This highlights the high leverage of early defensive breakdowns in otherwise low-exposure matches.

---
## Data Sources & Collection

Match event data was obtained using the `soccerdata` Python library and supplemented with a custom web scraping pipeline built to extract shot-level information for Ederson's 2017–18 Premier League matches from [UnderStat](understat.com).

The scraping and preprocessing pipeline:

- programmatically retrieved match data  
- standardized event fields  
- exported structured CSV files for analysis  
- validated data integrity prior to EDA  

This end-to-end workflow ensures the analysis is fully reproducible from raw source data.

---
## Methodology

The analysis pipeline:

1. Clean and validate event-level shot data  
2. Isolate goalkeeper-relevant events  
3. Enforce minute-level filtering (shots while Ederson on pitch)  
4. Aggregate to match level  
5. Identify first on-target shot chronologically  
6. Compute concession metrics  

All steps are fully reproducible in the notebook.

---

## Tech Stack

- Python (pandas, matplotlib)
- Jupyter Notebook
- PostgreSQL (data ingestion and SQL validation)
- Git/GitHub

---

## Repository Structure
- ederson-first-shot-analysis/
- ├── data/raw/
- ├── notebooks/
- ├── sql/
- ├── outputs/
- └── README.md


---

## Why This Matters

In dominant teams that suppress shot volume, goalkeeper evaluation based solely on aggregate save percentage can miss **high-leverage early-match risk**.

This project demonstrates how event-level analysis can surface timing-based vulnerabilities that aggregate metrics may obscure.

---

## Next Steps

Planned extensions include:

- time-to-first-shot analysis  
- feature engineering for match context  
- logistic regression baseline  
- multi-season comparison  

---

## Corners Prediction Model & RAG System (separate sub-project)

A second, independent project in this repo: a Poisson regression model predicting Man City's corners won per match (`notebooks/corners_model.ipynb`), and a RAG (retrieval-augmented generation) Q&A system built on top of it — demonstrating general AI-engineering skill (retrieval, eval design) alongside the sports-domain ML work above.

### Results

| Feature set | Test R² |
|---|---|
| 3 features (baseline) | 0.026 |
| + opponent shots-conceded | 0.062 |
| + current-season data | 0.064 |

Test MSE beat the "always predict the mean" baseline at every stage (9.5–9.7 vs. 10.4). More notably: a first real-world prediction (11.86 corners for an upcoming fixture) looked implausible against the bookmaker's line, which led to directly testing two competing explanations — stale/regime-mixed data (ruled out) vs. training instability from a small dataset (confirmed: **std 2.43 across 10 random seeds** on the same fixture). The model ships with an honest range, not a false-precision point estimate. Full writeup — including the hypothesis-testing, not just the numbers — in [docs/corners-model-writeup.md](docs/corners-model-writeup.md).

The RAG system (corpus generation → embeddings/vector store → hybrid structured+semantic router → eval set → FastAPI service with a minimal UI) is scored against a 20-question eval set at **18/20 (90%)** — 14/14 on the exact-number structured path, 4/6 on semantic retrieval, with both misses diagnosed as genuine embedding-model limitations rather than left unexplained.

### Data sources

- **[football-data.co.uk](https://www.football-data.co.uk)** — Premier League match data for the 2023/24, 2024/25, and 2025/26 seasons. Free for personal/non-commercial use. Not committed to this repo (`data/raw/*` is gitignored); download the England Premier League CSVs directly from their site to reproduce.
- **[Sofascore](https://www.sofascore.com)** — a small number of current-season (2026/27) matches, collected manually (one page at a time, not automated bulk scraping) via a personal Tampermonkey export script and converted to the same CSV schema with [scripts/convert_sofascore_json.py](scripts/convert_sofascore_json.py). Also not committed, for the same reason as above — this repo doesn't redistribute either source's underlying data, only the code that processes it.

### RAG corpus

[data/processed/match_notes.jsonl](data/processed/match_notes.jsonl) **is** committed — it's original generated text (one natural-language "analyst note" per historical match, produced by [scripts/generate_match_notes.py](scripts/generate_match_notes.py) from the model's engineered features and each match's real recorded outcome), not a redistribution of either raw data source.

---

## Author

Tirenioluwa Daniel Biodun-Kuti

LinkedIn Profile : https://www.linkedin.com/in/tbk022/
