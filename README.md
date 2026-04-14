# DSA210 Project (Spring 2025-2026) — WSB vs Meme Stocks

This repo contains the **data collection (enrichment), EDA, and hypothesis tests** for the DSA210 term project:
**"The Pulse of WallStreetBets: Analyzing the Impact of Social Media Discussion Volume on Stock Market Volatility and Trading Volume"**.

## Motivation
The 2021 “meme stock” period raised a practical question: **can social-media attention act as an indicator of market activity?**
This project investigates whether **discussion volume on r/wallstreetbets** (mentions, post counts, and engagement) is associated with (and potentially leads) changes in:
- **Trading volume**
- **Volatility** (proxied with daily return-based measures)

## Data
- **Reddit posts**: `reddit_wsb.csv` (Kaggle / G. Preda dataset format)
- **Market enrichment**: downloaded via `yfinance` and cached under `data/market_<TICKER>.csv`

## Method (14 April checkpoint scope)
- **Cleaning / preparation**
  - Parse timestamps (`created` Unix epoch; fallback to `timestamp`)
  - Extract ticker mentions (regex on `title + body`)
  - Aggregate to **daily** time series (`posts`, `mentions_<TICKER>`, score aggregates)
- **Enrichment**
  - Download daily OHLCV using `yfinance` for selected tickers (default: GME/AMC/BB)
  - Join on date and create **next-day targets** (e.g., `vol_next`, `absret_next`)
- **EDA**
  - Time-series plots for posts/mentions
  - Scatter plots: mentions vs next-day volume / next-day |return|
- **Hypothesis testing**
  - Mann–Whitney U: compare **high-attention days** (top 25% mentions) vs others
  - Spearman correlation: mentions vs next-day targets

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (EDA + hypothesis tests)

```bash
python -m src.run_eda_hypothesis --reddit_csv reddit_wsb.csv --tickers GME AMC BB
```

Outputs:
- Figures saved under `outputs/figures/`
- Summary tables and test results saved under `outputs/tables/`

## Notebook (step-by-step)
For a cell-by-cell, “how it was done” view (imports → cleaning → enrichment → plots → tests), open:
- `dsa210_analysis.ipynb`

## Key findings (current checkpoint)
- **Next-day trading volume** tends to be higher after days with high WSB attention (mentions), across GME/AMC/BB (see `outputs/tables/hypothesis_tests.csv`).
- **Next-day volatility** (|return|) shows a weaker/mixed relationship: significant for some tickers, not for all.
- These results are **correlational**; they do not imply causality.

### Hypothesis test summary (p-values)
From `outputs/tables/hypothesis_tests.csv`:
- **H1 (High mentions → higher next-day volume)**: significant for **GME** (\(p=3.00\times 10^{-13}\)), **AMC** (\(p=4.76\times 10^{-6}\)), **BB** (\(p=2.39\times 10^{-11}\)).
- **H2 (High mentions → higher next-day volatility \(|r|\))**: significant for **GME** (\(p=0.0201\)) and **AMC** (\(p=0.0217\)); **BB** not significant (\(p=0.146\)).
- **H3 (Spearman: mentions vs next-day volume)**: significant positive correlation for **GME** (ρ=0.798, \(p=5.62\times 10^{-28}\)), **AMC** (ρ=0.485, \(p=1.74\times 10^{-8}\)), **BB** (ρ=0.702, \(p=3.07\times 10^{-19}\)).
- **H3b (Spearman: mentions vs next-day \(|r|\))**: significant for **GME** (ρ=0.354, \(p=6.95\times 10^{-5}\)) and **BB** (ρ=0.240, \(p=0.00793\)); **AMC** not significant (\(p=0.063\)).

## Example figures
![Posts per day](outputs/figures/posts_per_day.png)
![Mentions per day](outputs/figures/mentions_per_day.png)

## Notes on reproducibility
- Market data is **cached**. Re-running will reuse cached `data/market_<TICKER>.csv` unless `--refresh_market` is passed.
- The analysis aggregates Reddit posts to **daily** time series and joins with daily market data.

## Project structure
```
DSA210_project/
├── dsa210_analysis.ipynb
├── src/
│   └── run_eda_hypothesis.py
├── data/
│   └── market_<TICKER>.csv
├── outputs/
│   ├── figures/
│   └── tables/
├── REPORT_14APR_EDA_HYPOTHESIS.md
├── requirements.txt
└── README.md
```

## Academic Integrity (AI usage disclosure)
This project is an original work prepared for **DSA 210 – Introduction to Data Science** (Sabancı University).

AI tools were used as a **coding assistant** for:
- Drafting a reproducible project structure (`src/`, `outputs/`, `requirements.txt`, `README.md`)
- Implementing and debugging the EDA + hypothesis testing pipeline
- Converting the pipeline into a step-by-step Jupyter notebook

**Prompts used (high level):**
- “Implement the 14 April deliverable: data collection/enrichment, EDA, hypothesis tests for `reddit_wsb.csv` and yfinance market data (GME/AMC/BB).”
- “Create a cell-by-cell `ipynb` notebook showing imports, cleaning, mention extraction, joining, plots, and statistical tests.”
- “Add an Academic Integrity section and improve README; create `outputs/tables/` and save tables there.”

All results were reviewed and are reproducible by running the code in this repository.
