"""
Machine-learning milestone (DSA 210): supervised models on daily joined WSB + market data.

We use time-ordered cross-validation (TimeSeriesSplit) so that the model never trains on
future days when scoring a past-held-out segment — appropriate for sequential market data.

Tasks:
  - Regression: predict log1p(next-day volume) from same-day Reddit activity + same-day market proxies.
  - Classification: predict whether next-day volume is at or above the training fold's median volume
    (threshold is recomputed inside each fold from training labels only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .run_eda_hypothesis import (
    aggregate_daily,
    extract_mentions,
    fetch_market,
    join_reddit_market,
    load_reddit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML models: WSB features vs next-day volume (time-series CV)")
    p.add_argument("--reddit_csv", type=str, default="reddit_wsb.csv")
    p.add_argument("--tickers", nargs="+", default=["GME", "AMC", "BB"])
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--refresh_market", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument(
        "--from_tables",
        type=str,
        default="",
        help="If non-empty, load joined_daily_<TICKER>.csv from this directory (skip Reddit rebuild).",
    )
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_joined_tables(table_dir: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load precomputed daily joins (e.g. after EDA run) when Reddit CSV is not available locally."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = table_dir / f"joined_daily_{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing joined table: {path}")
        out[t] = pd.read_csv(path, parse_dates=["date"])
    return out


def build_joined_from_reddit(args: argparse.Namespace, tickers: list[str]) -> dict[str, pd.DataFrame]:
    reddit_csv = Path(args.reddit_csv)
    data_dir = Path(args.data_dir)
    df = load_reddit(reddit_csv, max_rows=args.max_rows)
    df = extract_mentions(df, tickers=tickers)
    daily = aggregate_daily(df, tickers=tickers)
    start = str(daily.index.min().date())
    end = str((daily.index.max() + pd.Timedelta(days=2)).date())
    joined_by_ticker: dict[str, pd.DataFrame] = {}
    for t in tickers:
        cache_path = data_dir / f"market_{t}.csv"
        market = fetch_market(t, start=start, end=end, cache_path=cache_path, refresh=args.refresh_market)
        joined_by_ticker[t] = join_reddit_market(daily, market, ticker=t)
    return joined_by_ticker


def engineer_features(joined: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X and aligned targets. Rows without a realized next day (vol_next NA) are dropped.
    y_reg: log1p(vol_next). y_vol_raw: raw vol_next (for classification thresholds).
    """
    mcol = f"mentions_{ticker}"
    df = joined.sort_values("date").copy()
    df = df[df["vol_next"].notna()].copy()

    # Same-day market proxies; fill edge NA from pct_change windows.
    df["log_volume"] = np.log1p(df["Volume"].clip(lower=0))
    df["abs_return"] = df["abs_return"].fillna(0.0)
    df["hl_range"] = df["hl_range"].fillna(0.0)

    # Simple autoregressive social context: yesterday's buzz may carry into today.
    df["mentions_lag1"] = df[mcol].shift(1).fillna(0.0)
    df["posts_lag1"] = df["posts"].shift(1).fillna(0.0)

    feature_cols = [
        "posts",
        "score_sum",
        "score_mean",
        mcol,
        "log_volume",
        "abs_return",
        "hl_range",
        "mentions_lag1",
        "posts_lag1",
    ]
    X = df[feature_cols].copy()
    y_vol_raw = df["vol_next"].astype(float)
    y_reg = pd.Series(np.log1p(y_vol_raw.clip(lower=0).to_numpy()), index=y_vol_raw.index)
    return X, y_reg, y_vol_raw


def _n_splits_ts(n_samples: int) -> int:
    if n_samples < 40:
        return 3
    if n_samples < 80:
        return 4
    return 5


def evaluate_regression_cv(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> list[dict[str, Any]]:
    """Return one dict per model with mean/std of MAE, RMSE, R2 across TimeSeriesSplit folds."""
    n = len(X)
    n_splits = min(_n_splits_ts(n), max(2, n // 15))
    tsc = TimeSeriesSplit(n_splits=n_splits)

    models: dict[str, Any] = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=random_state,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=200,
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, est in models.items():
        maes: list[float] = []
        rmses: list[float] = []
        r2s: list[float] = []
        for train_idx, test_idx in tsc.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(est))])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            maes.append(mean_absolute_error(y_te, pred))
            rmses.append(float(np.sqrt(mean_squared_error(y_te, pred))))
            r2s.append(float(pipe.score(X_te, y_te)))

        rows.append(
            {
                "model": name,
                "cv_splits": n_splits,
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
            }
        )
    return rows


def evaluate_classification_cv(
    X: pd.DataFrame,
    vol_next: pd.Series,
    random_state: int,
) -> list[dict[str, Any]]:
    """
    Binary label: volume_next >= median(volume_next on the training slice of that fold).
    """
    n = len(X)
    n_splits = min(_n_splits_ts(n), max(2, n // 15))
    tsc = TimeSeriesSplit(n_splits=n_splits)

    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, est in models.items():
        accs: list[float] = []
        rocs: list[float] = []
        f1s: list[float] = []
        for train_idx, test_idx in tsc.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            v_tr = vol_next.iloc[train_idx]
            v_te = vol_next.iloc[test_idx]
            med = float(v_tr.median())
            y_tr = (v_tr >= med).astype(int)
            y_te = (v_te >= med).astype(int)
            if y_te.nunique() < 2:
                continue
            pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(est))])
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            accs.append(accuracy_score(y_te, pred))
            try:
                rocs.append(roc_auc_score(y_te, proba))
            except ValueError:
                rocs.append(float("nan"))
            f1s.append(f1_score(y_te, pred, zero_division=0))

        if not accs:
            rows.append(
                {
                    "model": name,
                    "cv_splits": n_splits,
                    "accuracy_mean": float("nan"),
                    "roc_auc_mean": float("nan"),
                    "f1_mean": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "model": name,
                "cv_splits": n_splits,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "roc_auc_mean": float(np.nanmean(rocs)),
                "roc_auc_std": float(np.nanstd(rocs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
            }
        )
    return rows


def naive_baseline_mae(y: pd.Series, n_splits: int) -> tuple[float, float]:
    """Predict the expanding / prior-only mean in log space (approximates 'no ML' rolling mean)."""
    tsc = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    for train_idx, test_idx in tsc.split(y):
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pred = np.full(shape=len(y_te), fill_value=float(y_tr.mean()))
        maes.append(mean_absolute_error(y_te, pred))
    return float(np.mean(maes)), float(np.std(maes))


def export_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    out_csv: Path,
    random_state: int,
) -> None:
    """Fit a single RF on all data for interpretability (not a CV estimate of generalization)."""
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", rf)])
    pipe.fit(X, y)
    imp = pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values(
        "importance", ascending=False
    ).to_csv(out_csv, index=False)


def run_ml_milestone(
    joined_by_ticker: dict[str, pd.DataFrame],
    table_dir: Path,
    tickers: list[str],
    *,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Run regression + classification CV, export `ml_*.csv` and `ml_run_notes.json` under ``table_dir``.

    Shared by ``python -m src.run_ml`` and ``dsa210_analysis.ipynb`` so the notebook and CLI stay aligned.
    """
    ensure_dir(table_dir)
    tickers = [t.upper().strip() for t in tickers]
    reg_rows: list[dict[str, Any]] = []
    cls_rows: list[dict[str, Any]] = []

    for t in tickers:
        j = joined_by_ticker[t]
        X, y_reg, y_vol = engineer_features(j, ticker=t)
        if len(X) < 25:
            if verbose:
                print(f"Skip {t}: too few rows after feature engineering ({len(X)}).")
            continue

        n_splits = min(_n_splits_ts(len(X)), max(2, len(X) // 15))
        base_m, base_s = naive_baseline_mae(y_reg, n_splits=n_splits)

        if verbose:
            print(f"\n=== {t} (n={len(X)}) ===")
            print(f"Naive train-mean baseline MAE (log1p vol_next): {base_m:.4f} ± {base_s:.4f}")

        for r in evaluate_regression_cv(X, y_reg, random_state=random_state):
            r["ticker"] = t
            r["naive_mae_mean"] = base_m
            r["naive_mae_std"] = base_s
            reg_rows.append(r)
            if verbose:
                print(
                    f"  {r['model']}: MAE {r['mae_mean']:.4f} ± {r['mae_std']:.4f} | "
                    f"RMSE {r['rmse_mean']:.4f} | R² {r['r2_mean']:.3f}"
                )

        for r in evaluate_classification_cv(X, y_vol, random_state=random_state):
            r["ticker"] = t
            cls_rows.append(r)
            if verbose:
                print(
                    f"  [{r['model']}] acc {r.get('accuracy_mean', float('nan')):.3f} | "
                    f"ROC-AUC {r.get('roc_auc_mean', float('nan')):.3f} | "
                    f"F1 {r.get('f1_mean', float('nan')):.3f}"
                )

        export_feature_importance(
            X,
            y_reg,
            table_dir / f"ml_feature_importance_{t}.csv",
            random_state=random_state,
        )

    df_reg: pd.DataFrame | None = None
    df_cls: pd.DataFrame | None = None
    if reg_rows:
        df_reg = pd.DataFrame(reg_rows)
        df_reg["mae_improvement_vs_naive_pct"] = (
            (df_reg["naive_mae_mean"] - df_reg["mae_mean"]) / df_reg["naive_mae_mean"] * 100.0
        )
        df_reg.to_csv(table_dir / "ml_regression_metrics.csv", index=False)
    if cls_rows:
        df_cls = pd.DataFrame(cls_rows)
        df_cls.to_csv(table_dir / "ml_classification_metrics.csv", index=False)

    notes = {
        "validation": "TimeSeriesSplit: each fold trains on an earlier time block and tests on the next block.",
        "regression_target": "log1p(next-day volume)",
        "classification_target": "1 if next-day volume >= median(next-day volume on the training slice of that CV fold)",
        "interpretation": (
            "Ridge is a regularized linear baseline; RandomForest and GradientBoosting capture non-linear "
            "interactions between WSB activity and same-day market proxies. "
            "Under TimeSeriesSplit, sklearn R² can be strongly negative when the test window differs from the "
            "training scale even if MAE beats a naive mean forecaster — report MAE improvement vs naive alongside R². "
            "For classification, prefer ROC-AUC over accuracy when classes are imbalanced; high accuracy with low F1 "
            "usually means the model predicts the majority class often."
        ),
    }
    (table_dir / "ml_run_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nSaved: {table_dir / 'ml_regression_metrics.csv'}")
        print(f"Saved: {table_dir / 'ml_classification_metrics.csv'}")
        print(f"Saved: {table_dir / 'ml_feature_importance_<TICKER>.csv'}")
        print(f"Saved: {table_dir / 'ml_run_notes.json'}")

    return df_reg, df_cls


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    ensure_dir(table_dir)
    ensure_dir(fig_dir)

    tickers = [t.upper().strip() for t in args.tickers]

    if args.from_tables:
        print(f"Loading joined tables from {args.from_tables!r} ...")
        joined_by_ticker = load_joined_tables(Path(args.from_tables), tickers)
    else:
        print("Building joined panels from Reddit + market (same pipeline as EDA)...")
        joined_by_ticker = build_joined_from_reddit(args, tickers)

    run_ml_milestone(
        joined_by_ticker,
        table_dir,
        tickers,
        random_state=args.random_state,
        verbose=True,
    )


if __name__ == "__main__":
    main()
