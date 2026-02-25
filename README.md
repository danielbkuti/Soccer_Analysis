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

## Author

Tirenioluwa Daniel Biodun-Kuti

LinkedIn Profile : https://www.linkedin.com/in/tbk022/
