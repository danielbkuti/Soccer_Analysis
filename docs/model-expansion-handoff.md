# Handoff — Model Accuracy & Expansion Plan

**Status:** Scoping only — not started. This is deliberately a "what it takes" reference for whenever this work actually begins, not a sprint plan.

**Repo:** [danielbkuti/Soccer_Analysis](https://github.com/danielbkuti/Soccer_Analysis)

---

## The ask

Three bundled expansions to the corners model:
1. Improve prediction accuracy
2. Expand from "Man City only" to all 20 Premier League teams
3. Extend to Man City's non-Premier-League matches (Champions League, FA Cup, EFL Cup, etc.)

---

## 1. Improving accuracy

The honest diagnosis from the Sep 8 session's investigation: low R² (~0.03–0.06) and the prediction instability we measured directly (std 2.43 across 10 seeds on one fixture) both trace back to one root cause — **~114 training rows is not much data** for any model, however small.

- **More data — the same lift as #2 below.** Pooling all 20 teams' matches (target = the home/away team's own corners, not specifically City's) multiplies the effective training set by roughly 20x. This is the single highest-leverage fix, and it's the same underlying work as the team expansion.
- **Richer features.** Currently just 3 rolling stats + venue. Worth adding: league table position (quality-gap proxy), head-to-head history, referee identity, and some signal for managerial/tactical regime changes — a real, already-demonstrated source of error from this session's investigation.
- **A different model once data volume justifies it.** A single `Linear` layer suits ~100 rows; at thousands of rows, a gradient-boosted model (XGBoost/LightGBM) or a proper Poisson GLM (`statsmodels`) may outperform and be far more stable than a small hand-rolled PyTorch net.
- **A real train/val/test protocol.** Time-based cross-validation (train on seasons 1-3, validate on 4, repeat) instead of one single split, once there's enough data to support it.

## 2. Expanding to all Premier League teams

A real architectural rewrite, not a config flag:

- **Generalize `data_pipeline.py`.** It's currently hard-coded around Man City (`city_home`, `city_away`, `'corners' = HC if City home else AC`). Needs to become "for any team, build their own home/away rolling history" — this already partially exists as a pattern, since `all_conceded`/`all_shots` are *already* built team-agnostically; the City-specific piece is what needs rewriting.
- **Add a team-identity signal.** A shared model needs to know *which* team it's predicting for, since teams differ systematically. Options: one-hot encode team identity (~20 features), or use *style* features (possession %, pass completion) if sourceable — the current data source (football-data.co.uk) doesn't include those, which is a real data-source gap, not just a modeling choice.
- **Data volume becomes the real bottleneck.** football-data.co.uk being down only cost us City + Man United's current-season data; if it stays down, manually collecting current-season data for all 20 teams via the Tampermonkey tool isn't remotely practical the way it was for 6 matches. This expansion basically requires football-data.co.uk (or a proper paid API) to be reliable — the manual tool was a stopgap for a narrow gap, not a scaling strategy.
- **Ripple effect on the RAG project.** The corpus generator, structured tool, and router are all currently City-only too. Generalizing the model doesn't automatically generalize the RAG system on top of it — that's additional, separate work if wanted later.

## 3. Man City in non-Premier-League competitions

The most separate of the three, and likely the most work relative to payoff:

- **Needs an entirely different data source.** football-data.co.uk is Premier-League-only. City's actual UCL fixtures this season (vs. Porto, PSG, Leipzig, Napoli, Barcelona — seen directly in this session's Sofascore browsing) aren't in the pipeline at all, nor is FA Cup/EFL Cup history.
- **A `competition` feature, and a pool-vs-separate judgment call.** Cup and European matches have different dynamics — squad rotation, quality mismatches (PL side vs. lower-league FA Cup opponent), different referees. A `competition` feature rather than fully separate per-competition models keeps the pooled sample usable.
- **Domain knowledge matters most here.** Knowing when City rotates their squad for a cup tie, or how seriously a given competition is treated in a given season, is exactly the kind of intuition a generic pipeline can't infer from the numbers alone — this is the piece most worth applying real football knowledge to, more than either of the other two expansions.

## Suggested priority order, whenever this starts

1. Generalize the pipeline to all PL teams first — biggest accuracy payoff, reuses existing patterns (`all_conceded`/`all_shots` already show the way)
2. Re-run the stability/variance investigation at the new scale — more data should shrink the std=2.43 problem, worth confirming rather than assuming
3. Only then tackle non-PL competitions — separate data-sourcing problem, smaller and more isolated payoff

---

## Completed alongside this handoff — Minimal Web UI

Not part of the expansion plan above — a small, immediate piece of work done in the same session: a minimal frontend for the existing RAG API, since the only prior "UI" was the developer-facing Swagger docs page.

- **Added to [scripts/api.py](../scripts/api.py):** a `GET /` route serving a single self-contained HTML page (inline, no separate static-file setup) — a text input, an "Ask" button, two clickable example questions, and a result area showing the answer plus a `path / mode` badge (surfacing which retrieval path answered the question, in keeping with this being a hybrid structured/semantic system).
- **Verified working end-to-end in the browser**, not just visually: clicked an example question, confirmed the real request round-tripped to `/ask` and rendered the correct answer ("City averaged 7.0 corners won in 2024/25 at home (19 matches)"), matching the eval set's known ground truth for that question.
- No new dependencies — plain HTML/CSS/JS returned as a string from the existing FastAPI app.
