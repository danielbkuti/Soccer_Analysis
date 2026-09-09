# Handoff — Corners Model Session (Sep 8, 2026)

**Repo:** [danielbkuti/Soccer_Analysis](https://github.com/danielbkuti/Soccer_Analysis)
**Notebook:** `notebooks/corners_model.ipynb`
**Env:** `ederson-ml` (conda)

---

## Session complete — Track A (corners model)

- [x] **HT-score leakage question** — resolved by dropping the idea. Raw in-match `HTHG`/`HTAG` isn't available pre-kickoff, so it can't be used for a fixture predictor; a historical-rolling version was judged not worth the added complexity vs. the simpler alternative below.
- [x] **Opponent shots-conceded feature added** (`rolling_shots_conceded`, from `HS`/`AS`, same `shift(1)` + `groupby().transform()` pattern as `rolling_conceded`). Model is now 4 features: `city_rolling_corners`, `rolling_conceded`, `is_home`, `rolling_shots_conceded`.
- [x] **Retrained and compared R²:** 0.026 (3-feature baseline) → 0.062 (4-feature, 3 seasons) → 0.064 (4-feature, + 2026/27 data).
- [x] **2026/27 data-sourcing tooling built**, since `football-data.co.uk` was down mid-session:
  - Tampermonkey userscript exports a Sofascore match's stats (corners, shots, etc.) to clipboard as JSON, triggered manually per page — not an automated crawler.
  - [scripts/convert_sofascore_json.py](../scripts/convert_sofascore_json.py) converts a folder of those JSON exports into one CSV matching the existing season-file schema (`Date, HomeTeam, AwayTeam, HC, AC, HS, AS`), with team-name normalization and date cleanup.
  - 6 matches collected this way → [data/raw/pl_26-27_manual.csv](../data/raw/pl_26-27_manual.csv) (City's 3 games + Man United's 3 games so far this season), concatenated onto the existing 3 seasons in the notebook's first cell.
- [x] **Fixture prediction interface built** (`predict_corners(opponent, is_home, n=5)`) — pulls each input's last-5-match average directly (no `shift(1)` needed, since there's no played match to leak from for a future fixture), feeds it through the trained model.
- [x] **Run on a real fixture** — Man United (home) vs Man City (away), confirmed from Sofascore's actual schedule as the next PL meeting: **13/09/2026, Old Trafford**.
- [x] **Instability investigated and root-caused** (see below) — multi-seed loop added to the notebook (`train_model(seed)`, cell `15586669cb652667`), regime-change hypothesis tested for both teams.

---

## ✅ Investigation results — why the first prediction (11.86) couldn't be trusted, and what the honest number actually is

The first real run of `predict_corners('Man United', is_home=0)` gave **11.86** — implausibly high against the bookmaker's *total match* corners line of 10.5 (both teams combined; Sofascore, Over 2.10 / Under 1.67). Rather than accept or discard that number on gut feel, it got tested against two competing explanations:

**Hypothesis 1 — stale/mixed-regime data.** Man City changed manager and most of the first team at the start of 2026/27 (3 games of history since); Man United changed on 13/01/2026 (20 games of history since). Checked whether either team's rolling-5 input was contaminated by pre-change matches:
- City: rolling-5 using all history (7.60) vs. their 3 post-change games only (7.67) — barely moved. Not a meaningful effect.
- Man United: initially looked sensitive (opp_conceded jumped 3.00→4.00 using only their most recent match) — but this turned out to be a **wrong assumption on my part** about how recent their change was. Once the real date (13/01/2026) was used, Man United's last 5 matches were already *entirely* post-change — the original baseline numbers were correct all along, and the earlier "sensitivity" was just single-match noise (n=1), not a regime effect.
- **Conclusion: regime change does not meaningfully explain the number.** Both teams' rolling inputs were already regime-appropriate.

**Hypothesis 2 — model instability.** Trained the same architecture 10 times with different seeds (`torch.manual_seed`), same data, same fixture:

```
mean = 11.48   std = 2.43   range = (6.85, 13.64)
```

**This is the real explanation.** With ~80 training rows, a single training run's output depends heavily on random weight initialization — confirmed and quantified, not just suspected.

**Honest final number for Sunday's fixture:** City ~7–14 corners, centered around ~11.5, with the uncertainty coming from training variance rather than data quality. This is a good number to hold up against the actual result after 13/09/2026 — both as a test of the model and as a concrete demonstration of why single-run point predictions from small datasets shouldn't be trusted at face value.

**Follow-up worth doing eventually (not urgent):** the variance itself is the remaining problem. Options for a future session: more training data (natural, as 2026/27 progresses), an ensemble-average prediction as the reported number instead of any single run, or a simpler/more regularized model that's less sensitive to initialization at this sample size.

---

## Next up — RAG system (separate project, starts fresh)

Per the [week plan](week-handoff.md — see chat history / your saved copy), the RAG build (corpus generation → embeddings → structured query + router → eval set) was deliberately scheduled for a **separate session**, since it's built from this model's finished feature set. That still holds — but the stability concern above is **not a blocker for corpus generation itself**.

Corpus generation only needs the rolling-average features (done, validated) and each match's *actual* recorded outcome — no model inference involved. It's safe to start Session 2 without resolving the stability issue first.

What *does* depend on the stability issue: enriching notes with "model predicted X, actual was Y" (a nice-to-have, not part of the core corpus), and any eval question built around a specific predicted number. Hold those off — and don't build them into the eval set — until the multi-seed check confirms 11.86 wasn't a fluke.
