# DSA210 — 14 April Deliverable (Data Collection, EDA, Hypothesis Tests)

## Data sources
- **Reddit**: `reddit_wsb.csv` (WallStreetBets posts; columns: `title`, `body`, `score`, `created`, `timestamp`)
- **Market enrichment**: Daily OHLCV data for **GME**, **AMC**, **BB** downloaded with `yfinance` and cached under `data/market_<TICKER>.csv`.

## Data preparation
- Converted post time to UTC datetime using `created` (Unix epoch). If missing, used parsed `timestamp`.
- Extracted ticker mentions from `title + body` using a regex and counted daily mentions for each ticker.
- Aggregated Reddit posts to daily level:
  - `posts` (number of posts/day)
  - `score_sum`, `score_mean`
  - `mentions_GME`, `mentions_AMC`, `mentions_BB`
- Joined daily Reddit aggregates with daily market data by date.
- Created **next-day** market targets to test “leading indicator” effects:
  - `vol_next` = next-day trading volume
  - `absret_next` = next-day absolute close-to-close return
  - `hl_next` = next-day high-low range scaled by close

All generated tables are in `outputs/` and figures are in `outputs/figures/`.

## EDA (what was visualized)
- WSB activity over time: `outputs/figures/posts_per_day.png`
- Daily mentions for GME/AMC/BB: `outputs/figures/mentions_per_day.png`
- Scatter + regression:
  - `<TICKER>_mentions_vs_nextday_volume.png`
  - `<TICKER>_mentions_vs_nextday_abs_return.png`

## Hypotheses and statistical tests
Daily unit of analysis.

### H1 — High-mention days → higher next-day trading volume
- **Test**: Mann–Whitney U (non-parametric two-sample test)
- **Grouping**: “High attention” = top 25% days by `mentions_<TICKER>`; “Low attention” = remaining days
- **Effect size**: Cliff’s delta

**Results (p-values):**
- GME: \(p=3.00 \times 10^{-13}\)
- AMC: \(p=4.76 \times 10^{-6}\)
- BB: \(p=2.39 \times 10^{-11}\)

### H2 — High-mention days → higher next-day volatility
Volatility proxy: **next-day** \(|r|\) where \(r\) is close-to-close return.

- **Test**: Mann–Whitney U

**Results (p-values):**
- GME: \(p=0.0201\)
- AMC: \(p=0.0217\)
- BB: \(p=0.146\) (not significant at 0.05)

### H3 — Mentions correlate with next-day activity (monotonic)
- **Test**: Spearman correlation between `mentions_<TICKER>` and next-day target.

**Mentions vs next-day volume (Spearman ρ, p-value):**
- GME: ρ=0.798, \(p=5.62 \times 10^{-28}\)
- AMC: ρ=0.485, \(p=1.74 \times 10^{-8}\)
- BB: ρ=0.702, \(p=3.07 \times 10^{-19}\)

**Mentions vs next-day \(|r|\) (Spearman ρ, p-value):**
- GME: ρ=0.354, \(p=6.95 \times 10^{-5}\)
- AMC: ρ=0.169, \(p=0.063\) (not significant at 0.05)
- BB: ρ=0.240, \(p=0.00793\)

Full results table: `outputs/tables/hypothesis_tests.csv`.

## Limitations (for this stage)
- Mention extraction is regex-based and may include false positives / miss context (e.g., abbreviations).
- Using daily aggregation hides intraday dynamics.
- Next-day relationships are correlational; no causal claims.
- Only 2020-09 to 2021-08 is covered by this Reddit CSV.

## How to reproduce

```bash
pip install -r requirements.txt
python -m src.run_eda_hypothesis --reddit_csv reddit_wsb.csv --tickers GME AMC BB --refresh_market
```

