from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib import pyplot as plt
from scipy import stats


TICKER_RE = re.compile(r"(?<![A-Z0-9$])\$?([A-Z]{1,5})(?![A-Z0-9])")


@dataclass(frozen=True)
class HypothesisResult:
    name: str
    ticker: str
    n: int
    statistic: float
    p_value: float
    extra: dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA + hypothesis tests for WSB vs meme stocks")
    p.add_argument("--reddit_csv", type=str, default="reddit_wsb.csv")
    p.add_argument("--tickers", nargs="+", default=["GME", "AMC", "BB"])
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--refresh_market", action="store_true")
    p.add_argument("--max_rows", type=int, default=None, help="Optional: limit reddit rows for quick runs")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_reddit(csv_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    # Dataset is large; keep only columns we need.
    usecols = ["title", "score", "created", "body", "timestamp"]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows, low_memory=False)

    # Normalize types.
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["created"] = pd.to_numeric(df["created"], errors="coerce")

    # Prefer unix epoch if present; otherwise parse timestamp string.
    # Keep dt_utc consistently as tz-aware datetime64[ns, UTC] to avoid mixed dtypes.
    dt_created = pd.to_datetime(df["created"], unit="s", utc=True, errors="coerce")
    dt_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["dt_utc"] = dt_created.fillna(dt_ts)
    df = df[df["dt_utc"].notna()].copy()
    # Daily key for aggregation/merge: tz-naive midnight (datetime64[ns])
    df["date"] = df["dt_utc"].dt.floor("D").dt.tz_localize(None)

    # Combine text fields for mention extraction.
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["body"]).str.upper()

    return df


def extract_mentions(df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers]
    ticker_set = set(tickers)

    def _mentions(text: str) -> set[str]:
        found = set(m.group(1) for m in TICKER_RE.finditer(text))
        return found & ticker_set

    df = df.copy()
    df["mentioned"] = df["text"].map(_mentions)
    for t in tickers:
        df[f"m_{t}"] = df["mentioned"].map(lambda s, tt=t: int(tt in s))
    return df


def aggregate_daily(df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers]
    agg = {
        "score": ["count", "sum", "mean"],
    }
    for t in tickers:
        agg[f"m_{t}"] = ["sum"]

    daily = df.groupby("date", as_index=True).agg(agg)
    daily.columns = ["_".join(c).strip("_") for c in daily.columns.to_flat_index()]
    daily = daily.rename(
        columns={
            "score_count": "posts",
            "score_sum": "score_sum",
            "score_mean": "score_mean",
        }
    )
    for t in tickers:
        daily = daily.rename(columns={f"m_{t}_sum": f"mentions_{t}"})

    daily = daily.sort_index()
    return daily


def fetch_market(ticker: str, start: str, end: str, cache_path: Path, refresh: bool) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        # Cached files may have been written with MultiIndex headers (two rows).
        # If so, read both header rows and flatten.
        m = pd.read_csv(cache_path, header=[0, 1])
    else:
        data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if data.empty:
            raise RuntimeError(f"No market data returned for {ticker}.")
        m = data.reset_index()
        ensure_dir(cache_path.parent)
        m.to_csv(cache_path, index=False)

    if isinstance(m.columns, pd.MultiIndex):
        # Typical yfinance MultiIndex: (field, ticker). We keep only field names.
        cols = []
        for a, b in m.columns.to_list():
            if a in ("Date", "Datetime"):
                cols.append("Date")
            else:
                cols.append(a)
        m.columns = cols

    # Normalize schema.
    # Align to same daily key dtype used for Reddit: tz-naive midnight (datetime64[ns]).
    m["Date"] = pd.to_datetime(m["Date"], utc=True, errors="coerce").dt.floor("D").dt.tz_localize(None)
    m = m.rename(columns={"Date": "date"})

    # Volatility proxies.
    m["return_close"] = m["Close"].pct_change()
    m["abs_return"] = m["return_close"].abs()
    m["hl_range"] = (m["High"] - m["Low"]) / m["Close"]
    m = m[["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "return_close", "abs_return", "hl_range"]]
    m = m.sort_values("date")
    return m


def join_reddit_market(daily: pd.DataFrame, market: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = daily.reset_index().merge(market, on="date", how="inner")
    df = df.sort_values("date")

    # Next-day targets (leading indicator tests).
    df["vol_next"] = df["Volume"].shift(-1)
    df["absret_next"] = df["abs_return"].shift(-1)
    df["hl_next"] = df["hl_range"].shift(-1)
    df["ticker"] = ticker
    return df


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Effect size for two independent samples: P(X>Y) - P(X<Y)
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n*m) but small after daily aggregation.
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def hypothesis_tests(joined: pd.DataFrame, ticker: str) -> list[HypothesisResult]:
    results: list[HypothesisResult] = []

    mention_col = f"mentions_{ticker}"
    df = joined.dropna(subset=[mention_col, "Volume", "abs_return", "hl_range", "vol_next", "absret_next", "hl_next"])

    # Define "high attention" days as top quartile of mentions (within ticker).
    q75 = df[mention_col].quantile(0.75)
    high = df[df[mention_col] >= q75]
    low = df[df[mention_col] < q75]

    def mw(name: str, a: pd.Series, b: pd.Series) -> HypothesisResult:
        # Be robust for quick/small runs (e.g., --max_rows) where one group can be empty.
        if len(a) == 0 or len(b) == 0:
            return HypothesisResult(
                name=name,
                ticker=ticker,
                n=int(len(a) + len(b)),
                statistic=float("nan"),
                p_value=float("nan"),
                extra={"n_high": int(len(a)), "n_low": int(len(b)), "cliffs_delta": float("nan")},
            )

        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        cd = cliff_delta(a.to_numpy(), b.to_numpy())
        return HypothesisResult(
            name=name,
            ticker=ticker,
            n=len(a) + len(b),
            statistic=float(stat),
            p_value=float(p),
            extra={"n_high": int(len(a)), "n_low": int(len(b)), "cliffs_delta": float(cd)},
        )

    # H1: High-mention days have higher next-day trading volume.
    results.append(mw("H1_high_mentions_increase_nextday_volume", high["vol_next"], low["vol_next"]))

    # H2: High-mention days have higher next-day volatility (abs return).
    results.append(mw("H2_high_mentions_increase_nextday_abs_return", high["absret_next"], low["absret_next"]))

    # H3: Mentions correlate with next-day volume/volatility (Spearman).
    if df.shape[0] < 3:
        rho_v, p_v = (np.nan, np.nan)
    else:
        rho_v, p_v = stats.spearmanr(df[mention_col], df["vol_next"])
    results.append(
        HypothesisResult(
            name="H3_spearman_mentions_vs_nextday_volume",
            ticker=ticker,
            n=int(df.shape[0]),
            statistic=float(rho_v),
            p_value=float(p_v),
            extra={},
        )
    )
    if df.shape[0] < 3:
        rho_a, p_a = (np.nan, np.nan)
    else:
        rho_a, p_a = stats.spearmanr(df[mention_col], df["absret_next"])
    results.append(
        HypothesisResult(
            name="H3b_spearman_mentions_vs_nextday_abs_return",
            ticker=ticker,
            n=int(df.shape[0]),
            statistic=float(rho_a),
            p_value=float(p_a),
            extra={},
        )
    )

    return results


def plot_eda(daily: pd.DataFrame, joined_by_ticker: dict[str, pd.DataFrame], out_fig_dir: Path, tickers: list[str]) -> None:
    sns.set_theme(style="whitegrid")
    ensure_dir(out_fig_dir)

    # Overall posting activity.
    plt.figure(figsize=(12, 4))
    plt.plot(daily.index, daily["posts"], linewidth=1)
    plt.title("WSB posts per day (all posts)")
    plt.xlabel("Date")
    plt.ylabel("Posts")
    plt.tight_layout()
    plt.savefig(out_fig_dir / "posts_per_day.png", dpi=160)
    plt.close()

    # Mentions per ticker over time.
    plt.figure(figsize=(12, 5))
    for t in tickers:
        col = f"mentions_{t}"
        if col in daily.columns:
            plt.plot(daily.index, daily[col], linewidth=1, label=t)
    plt.title("Ticker mentions per day (from title+body)")
    plt.xlabel("Date")
    plt.ylabel("Mentions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig_dir / "mentions_per_day.png", dpi=160)
    plt.close()

    # For each ticker: mentions vs next-day volume scatter.
    for t, j in joined_by_ticker.items():
        mention_col = f"mentions_{t}"
        df = j.dropna(subset=[mention_col, "vol_next", "absret_next"])
        if df.empty:
            continue

        plt.figure(figsize=(6, 5))
        sns.regplot(data=df, x=mention_col, y="vol_next", scatter_kws={"s": 12, "alpha": 0.4}, line_kws={"color": "red"})
        plt.title(f"{t}: Mentions vs next-day Volume")
        plt.tight_layout()
        plt.savefig(out_fig_dir / f"{t}_mentions_vs_nextday_volume.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        sns.regplot(data=df, x=mention_col, y="absret_next", scatter_kws={"s": 12, "alpha": 0.4}, line_kws={"color": "red"})
        plt.title(f"{t}: Mentions vs next-day |return|")
        plt.tight_layout()
        plt.savefig(out_fig_dir / f"{t}_mentions_vs_nextday_abs_return.png", dpi=160)
        plt.close()


def save_results(results: list[HypothesisResult], out_path: Path) -> None:
    rows = []
    for r in results:
        row = {
            "hypothesis": r.name,
            "ticker": r.ticker,
            "n": r.n,
            "statistic": r.statistic,
            "p_value": r.p_value,
        }
        for k, v in (r.extra or {}).items():
            row[k] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    reddit_csv = Path(args.reddit_csv)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    ensure_dir(out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)

    tickers = [t.upper().strip() for t in args.tickers]

    print("Loading reddit CSV...")
    df = load_reddit(reddit_csv, max_rows=args.max_rows)
    print(f"Loaded {len(df):,} posts spanning {df['date'].min().date()} to {df['date'].max().date()}")

    print("Extracting ticker mentions...")
    df = extract_mentions(df, tickers=tickers)
    daily = aggregate_daily(df, tickers=tickers)
    daily.to_csv(table_dir / "daily_reddit_aggregates.csv")

    start = str(daily.index.min().date())
    # yfinance end is exclusive-ish; add one day buffer
    end = str((daily.index.max() + pd.Timedelta(days=2)).date())

    joined_by_ticker: dict[str, pd.DataFrame] = {}
    all_results: list[HypothesisResult] = []

    print("Fetching + joining market data...")
    for t in tickers:
        cache_path = data_dir / f"market_{t}.csv"
        market = fetch_market(t, start=start, end=end, cache_path=cache_path, refresh=args.refresh_market)
        joined = join_reddit_market(daily, market, ticker=t)
        joined_by_ticker[t] = joined
        joined.to_csv(table_dir / f"joined_daily_{t}.csv", index=False)

        res = hypothesis_tests(joined, ticker=t)
        all_results.extend(res)

    print("Creating EDA figures...")
    plot_eda(daily, joined_by_ticker, out_fig_dir=fig_dir, tickers=tickers)

    print("Saving hypothesis test results...")
    save_results(all_results, out_path=table_dir / "hypothesis_tests.csv")

    print("Done.")
    print(f"- Outputs: {out_dir.resolve()}")
    print(f"- Figures: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()

