# Predicting Man City's Corners: A Small-Data Case Study

A Poisson regression model predicting how many corners Manchester City will win in a Premier League match, built from three seasons of historical data plus a small manually-sourced supplement for the current season. The interesting part isn't the model architecture — it's simple by design — it's the process of stress-testing a small-data model rather than trusting its first output.

## The problem

Corners are high-variance (std ≈ 3.5 across City's matches) and low-signal relative to the features available in standard match-result data. The goal wasn't to build a highly accurate predictor — it's an honest, well-validated one that knows its own limitations.

## Data & features

Three seasons of Premier League data (2023/24–2025/26) from football-data.co.uk, supplemented with 6 current-season (2026/27) matches collected manually when the primary data source went down mid-project (via a personal Tampermonkey export tool, not a bulk scraper — see [README](../README.md) for the sourcing details).

Four leakage-safe features, all built with `.shift(1).rolling(5, min_periods=1)` so no match ever sees its own outcome:

- `city_rolling_corners` — City's own rolling 5-match average, corners won
- `rolling_conceded` — the upcoming opponent's rolling 5-match average, corners conceded (built league-wide, not just from City's own fixture history)
- `rolling_shots_conceded` — same idea, for shots
- `is_home` — venue, since City average 7.4 corners at home vs. 6.3 away

One feature was deliberately **not** built: half-time score. It looked like a natural proxy for "match state driving second-half corner patterns," but a match's own `HTHG`/`HTAG` doesn't exist at prediction time — a fixture predictor has to run pre-kickoff. The historical-rolling alternative was scoped but dropped in favor of the simpler, more directly-justified shots-conceded feature instead.

## Model

```
nn.Sequential(nn.Linear(4, 1), nn.Softplus())
criterion = nn.PoissonNLLLoss(log_input=False)
optimizer = Adam(lr=0.01)
```

A single linear layer with a softplus output to keep the Poisson rate positive. Deliberately minimal — appropriate for ~114 training rows, not a place to reach for a deeper network.

## Results

| Feature set | Test R² |
|---|---|
| 3 features (baseline) | 0.026 |
| + shots-conceded (4 features) | 0.062 |
| + current-season data | 0.064 |

Test MSE beat the "always predict the training mean" baseline (9.5–9.7 vs. 10.4) at every stage. R² stayed low throughout — expected given the target's natural variance, not a failure of the features.

## Stress-testing, not just reporting a number

The most substantive part of this project was refusing to trust a single prediction at face value. A first real-world test — predicting City's corners for their next fixture (away at Manchester United) — returned **11.86**, which looked implausibly high next to the bookmaker's *total match* corners line of 10.5. Rather than accept or dismiss it, two competing hypotheses were tested directly:

**Hypothesis 1 — stale or regime-mixed data.** Both teams had recent managerial/squad changes. Historical rolling features computed from a blended pre/post-change window could plausibly be misleading. Tested explicitly: filtering Man United's rolling window to strictly post-change matches produced **identical** numbers to the unfiltered version — their change (confirmed 13 Jan 2026) was long enough ago that the existing 5-match window was already entirely post-change. City's own side showed the same pattern. **Hypothesis rejected**, with the specific reasoning kept on record rather than quietly dropped.

**Hypothesis 2 — model instability.** Retrained the identical architecture 10 times with different random seeds, same data, same fixture:

```
mean = 11.48   std = 2.43   range = (6.85, 13.64)
```

**Confirmed.** A ~7-corner spread from nothing but weight initialization, on ~114 training rows. This is the actual explanation — not stale data, pure small-sample training variance — and it's a materially different, more useful finding than either accepting 11.86 or rejecting it outright.

## Honest conclusion

The defensible pre-match estimate for that fixture isn't a single number — it's **City in the 7–14 range**, with the real uncertainty coming from training variance on a small dataset, not from anything wrong with the features or the leakage handling. That range, and the reasoning behind it, is what actually got shipped — not a false-precision point estimate.

## What actually happened

The fixture was played on 13/09/2026: **Man United 0–1 Man City**, corners **6–7**. City's real total, **7**, sits at the exact lower bound of the predicted range — not near the central estimate of 11.48. The range held, but the point estimate would have missed by roughly 4.5 corners. One match is too small a sample to call this systematic bias, but it's a real, concrete illustration of why the honest range was worth reporting instead of the first clean-looking number the model produced.

## Tooling built alongside the model

- A Tampermonkey userscript + [conversion script](../scripts/convert_sofascore_json.py) to source current-season data manually when the primary source went down, with correct attribution and no bulk automated extraction
- A [fixture prediction interface](../notebooks/corners_model.ipynb) taking an upcoming opponent and venue, pulling live rolling stats, and returning a prediction — tested against a real fixture and a real bookmaker line, not a synthetic example

## Stack

Python, pandas, PyTorch, Jupyter — no external ML platform, no pretrained weights beyond what the (separate) RAG project layers on top of this later.
