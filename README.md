# DSA 210 — WallStreetBets attention vs meme-stock activity

## Project overview

This repository is the term project for **DSA 210 – Introduction to Data Science** (Sabancı University, Spring 2025–2026). It builds a full **data science pipeline**: collect and enrich data, explore it, run **statistical hypothesis tests**, then extend the work with **supervised machine learning** and clear interpretation—similar in spirit to course repos that keep one main notebook (e.g. [Arda Kalyoncu’s project layout](https://github.com/kalyoncuarda/Arda_Kalyoncu_Dsa_210_Term_Project/blob/main/README.md)) while the ML logic can also live in importable Python modules (cf. [Nil Kadakal’s separate `ml_salary_models.ipynb`](https://github.com/NilKadakal/DSA210_NilKadakal/blob/main/ml_salary_models.ipynb); here ML is **Section 10** inside `dsa210_analysis.ipynb` so one file shows the whole story).

**Working title:** *The Pulse of WallStreetBets: Analyzing the Impact of Social Media Discussion Volume on Stock Market Volatility and Trading Volume.*

**Repository:** [github.com/beyriahmetturan-rgb/Dsa-210](https://github.com/beyriahmetturan-rgb/Dsa-210)

---

## Research questions

1. **Association:** Is same-day **r/wallstreetbets** activity (posts, scores, ticker mentions) associated with **next-day** trading volume and volatility proxies for meme names (GME, AMC, BB)?
2. **Hypotheses:** Do **high-mention** days (top quartile) show systematically higher **next-day** volume or absolute returns than other days (Mann–Whitney)? Do Spearman correlations support a monotonic link?
3. **Prediction (ML extension):** Can we **predict** `log1p(next-day volume)` and a **high vs low volume** label from Reddit + same-day market features using several models, evaluated with **time-ordered cross-validation** (no random shuffling of trading days)?

---

## Datasets

| Source | Role | Notes |
|--------|------|--------|
| `reddit_wsb.csv` | Raw Reddit posts (title, body, score, timestamps) | Kaggle / G. Preda–style format; large file may stay local—pipeline expects it in repo root for full rerun. |
| `yfinance` | Daily OHLCV for GME, AMC, BB | Downloaded for the Reddit date span; **cached** as `data/market_<TICKER>.csv` to avoid repeated API calls. |
| Joined daily panels | One row per calendar day per ticker | Built in the notebook / `src/run_eda_hypothesis.py`; saved as `outputs/tables/joined_daily_<TICKER>.csv`. |

---

## Project structure

```
DSA210_project/
├── dsa210_analysis.ipynb    # Sections 1–9: EDA + hypothesis tests; Section 10: detailed ML milestone
├── src/
│   ├── __init__.py
│   ├── run_eda_hypothesis.py   # CLI: EDA + figures + hypothesis_tests.csv
│   └── run_ml.py               # CLI + importable run_ml_milestone() (same logic as notebook §10)
├── data/
│   └── market_<TICKER>.csv     # Cached OHLCV
├── outputs/
│   ├── figures/                # EDA plots + ML comparison plots (after §10)
│   └── tables/                 # daily_reddit_aggregates, joined_daily_*, hypothesis_tests, ml_*
├── requirements.txt
├── DSA-210-Project-Proposal.pdf   # if present
└── README.md
```

---

## How to run

1. **Clone** this repository and open a terminal in the project root (the folder that contains `src/`).

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **EDA + hypothesis tests (script)**

   ```bash
   python -m src.run_eda_hypothesis --reddit_csv reddit_wsb.csv --tickers GME AMC BB
   ```

4. **Machine learning (script)** — after joined tables exist:

   ```bash
   python -m src.run_ml --reddit_csv reddit_wsb.csv --tickers GME AMC BB
   # or, from cached joins only:
   python -m src.run_ml --from_tables outputs/tables --tickers GME AMC BB
   ```

5. **Notebook (full narrative, recommended for grading)**

   Open `dsa210_analysis.ipynb` in Jupyter / VS Code and **Run All** from top to bottom. Section **10** walks through the ML milestone with extra explanation and tables, then saves the same CSVs/figures as the CLI.

---

## Methodology

1. **Data cleaning** — Parse `created` (Unix epoch) to UTC; fallback to `timestamp`. Build a daily key and combine `title` + `body` for text mining.

2. **Mention extraction** — Regex tokenization for `$TICKER` / `TICKER` style mentions; daily counts per GME, AMC, BB.

3. **Aggregation** — Daily Reddit metrics: `posts`, `score_sum`, `score_mean`, `mentions_<TICKER>`.

4. **Enrichment** — Align with Yahoo Finance daily bars; compute returns and intraday range proxies.

5. **Join & targets** — Merge Reddit and market on `date`; define **next-day** targets (`vol_next`, `absret_next`, `hl_next`) for lead/lag style questions.

6. **EDA** — Time series of posts/mentions; scatter + regression: mentions vs next-day volume / |return|; figures under `outputs/figures/`.

7. **Hypothesis testing** — Mann–Whitney U (high vs low mention days); Spearman correlation (mentions vs next-day outcomes); results in `outputs/tables/hypothesis_tests.csv`.

8. **Machine learning extension** — Supervised models on engineered daily features (`src/run_ml.py`, **Section 10** in the notebook):
   - **Regression:** Ridge, Random Forest, Gradient Boosting → target `log1p(vol_next)`.
   - **Classification:** logistic regression, random forest → “high next-day volume” vs rest, where the volume median is computed **inside each training fold** only (no label leakage from the future).
   - **Validation:** `TimeSeriesSplit` (chronological blocks).
   - **Outputs:** `ml_regression_metrics.csv`, `ml_classification_metrics.csv`, `ml_feature_importance_<TICKER>.csv`, `ml_run_notes.json`, plus after notebook §10: `ml_mae_by_ticker.png`, `ml_r2_by_ticker.png`, `ml_roc_auc_by_ticker.png`, `ml_top_feature_importance_GME.png`.

---

## Machine learning extension (summary)

The ML stage asks a **prediction** question on top of the **association** tests from April: given WSB activity and same-day market state, how well can we forecast **next-day** liquidity (volume)?

- **Baseline for regression:** mean of `log1p(vol_next)` on the training slice of each CV fold (reported as `naive_mae_*` in `ml_regression_metrics.csv`). Models should beat this MAE if they add signal.
- **Why multiple models:** Ridge gives a **linear, interpretable** baseline; **Random Forest** and **Gradient Boosting** capture **non-linear** interactions (e.g. mentions × volatility). **Logistic regression** vs **random forest classifier** compare linear vs non-linear decision boundaries for the high-volume class.
- **How to read R²:** Under regime shifts (e.g. January 2021 meme spikes), sklearn **R² can be negative** on some folds even when **MAE improves** vs the naive baseline—use **`mae_improvement_vs_naive_pct`** alongside R².
- **Classification metrics:** Prefer **ROC-AUC** and **F1** for the minority “high volume” class; **accuracy alone** can look flattering when the positive class is rare.

---

## Findings

### Hypothesis tests (high-level)

| Hypothesis | GME | AMC | BB |
|------------|-----|-----|-----|
| H1: High mentions → higher **next-day volume** (Mann–Whitney) | Significant | Significant | Significant |
| H2: High mentions → higher **next-day \|return\|** | Significant | Significant | Not significant |
| H3: Spearman(mentions, next-day volume) | Strong positive ρ | Positive ρ | Positive ρ |
| H3b: Spearman(mentions, next-day \|return\|) | Significant | Not significant | Significant |

*(Exact p-values and statistics are in `outputs/tables/hypothesis_tests.csv` and in the notebook.)*

### ML insights (qualitative; refresh numbers after you rerun)

| Topic | Takeaway |
|--------|----------|
| Regression | Compare `mae_improvement_vs_naive_pct` per ticker in `ml_regression_metrics.csv`; Ridge is often competitive on short daily panels. |
| Trees vs linear | Random Forest / GBDT may win on some tickers; check per-ticker rows in the metrics table. |
| Feature importance | `ml_feature_importance_<TICKER>.csv` (RF refit on **all** days): often ranks `log_volume`, `score_sum`, and `mentions_<TICKER>` highly—liquidity and buzz move together in this sample. |
| Classification | Use ROC-AUC in `ml_classification_metrics.csv`; high accuracy with low F1 suggests majority-class dominance. |

All statistical and ML results are **correlational / predictive on a fixed sample**, not proof of causation.

---

## Limitations

- Regex mentions miss context and allow false positives.
- Daily bars hide intraday structure.
- Reddit CSV covers roughly **2020-09 – 2021-08**; conclusions may not generalize.
- ML uses a **small** number of trading days per ticker after joins—models are for coursework demonstration, not trading advice.

---

## Example figures (EDA)

![Posts per day](outputs/figures/posts_per_day.png)

![Mentions per day](outputs/figures/mentions_per_day.png)

---

## Notes on reproducibility

- Market data is **cached** under `data/` unless you pass `--refresh_market` to the EDA script.
- Re-running the notebook regenerates figures and tables under `outputs/`.

---

## Author

**GitHub:** [beyriahmetturan-rgb](https://github.com/beyriahmetturan-rgb) · **Course:** DSA 210 – Introduction to Data Science (Sabancı University, Spring 2025–2026)

---

## Project status

**Completed:** data collection & enrichment, cleaning, EDA, hypothesis tests, supervised ML (multi-model + time-series CV), integrated notebook Section 10, CLI scripts `run_eda_hypothesis.py` / `run_ml.py`.

**Upcoming:** final report & presentation (18 May per course calendar).

**Last updated:** May 2026

---

## Academic Integrity (AI usage disclosure)

This project is original course work for **DSA 210** (Sabancı University). **AI tools** (e.g. coding assistants) were used for:

- Organizing the repository and keeping a reproducible workflow  
- Implementing and debugging the EDA + hypothesis testing pipeline  
- Implementing the May ML milestone (`src/run_ml.py`, notebook Section 10), metrics tables, and figures  
- Restructuring the README (including alignment with common course README patterns) and improving clarity of explanations  

**Prompts (high level):** WSB + yfinance EDA/hypothesis deliverable; ML with time-series CV and multiple models; README in the style of peer examples; consolidate ML into `dsa210_analysis.ipynb` Section 10; document AI use per course policy. **All generated outputs were reviewed** and can be reproduced from this repository.

All sources (Kaggle Reddit data, yfinance) should be cited in the final report as required by the instructor.
