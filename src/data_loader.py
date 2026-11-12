"""
data_loader.py
--------------
Real financial market data ingestion via `yfinance`, missing-value auditing,
strict chronological train/test splitting (preventing look-ahead bias),
Engle-Granger cointegration / Augmented Dickey-Fuller stationarity testing,
and static vs. rolling-window spread construction.
"""

from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint


DEFAULT_ASSET_METADATA: Dict[str, Dict[str, str]] = {
    "GOOGL": {"name": "Alphabet Inc. Class A (Voting)", "type": "Equity (Dual-Class Common)"},
    "GOOG": {"name": "Alphabet Inc. Class C (Non-Voting)", "type": "Equity (Dual-Class Common)"},
    "V": {"name": "Visa Inc. Class A", "type": "Equity (Payment Network)"},
    "MA": {"name": "Mastercard Inc. Class A", "type": "Equity (Payment Network)"},
    "GLD": {"name": "SPDR Gold Shares", "type": "Commodity ETF (Gold Bullion)"},
    "GDX": {"name": "VanEck Gold Miners ETF", "type": "Equity ETF (Gold Mining)"},
    "AAPL": {"name": "Apple Inc.", "type": "Equity (Consumer Technology)"},
    "TSLA": {"name": "Tesla, Inc.", "type": "Equity (Automotive/Clean Tech)"},
    "IEF": {"name": "iShares 7-10 Year Treasury Bond ETF", "type": "Fixed Income ETF (7-10Y US Sovereign)"},
    "SHY": {"name": "iShares 1-3 Year Treasury Bond ETF", "type": "Fixed Income ETF (1-3Y US Sovereign Carry)"},
    "^TNX": {"name": "CBOE 10-Year US Treasury Yield Index", "type": "Interest Rate Index (% Yield)"},
    "^VIX": {"name": "CBOE Volatility Index", "type": "Implied Volatility Index (% Annualized Vol)"},
}


def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite tables and indices for financial time-series storage."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                adj_close REAL NOT NULL,
                PRIMARY KEY (date, ticker)
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date
            ON daily_prices (ticker, date);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_metadata (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                asset_type TEXT,
                first_date TEXT,
                last_date TEXT,
                total_records INTEGER
            );
        """)
    return conn


def save_prices_to_sqlite(
    df: pd.DataFrame,
    db_path: Path,
    metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """
    Persist wide daily price dataframe into normalized SQLite daily_prices table
    and update asset_metadata catalog.
    """
    conn = init_sqlite_db(db_path)
    meta_lookup = metadata or DEFAULT_ASSET_METADATA

    # Melt wide prices to long format (date, ticker, adj_close)
    reset_df = df.copy()
    reset_df.index = pd.to_datetime(reset_df.index).strftime("%Y-%m-%d")
    reset_df.index.name = "date"
    long_df = reset_df.reset_index().melt(id_vars=["date"], var_name="ticker", value_name="adj_close")
    long_df = long_df.dropna(subset=["adj_close"])

    with conn:
        # Upsert price records
        records = [
            (str(row.date), str(row.ticker), float(row.adj_close))
            for row in long_df.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_prices (date, ticker, adj_close)
            VALUES (?, ?, ?);
            """,
            records,
        )

        # Upsert metadata
        for ticker in df.columns:
            series = df[ticker].dropna()
            if len(series) == 0:
                continue
            first_d = str(pd.to_datetime(series.index[0]).date())
            last_d = str(pd.to_datetime(series.index[-1]).date())
            info = meta_lookup.get(ticker, {"name": ticker, "type": "Unknown"})
            conn.execute(
                """
                INSERT OR REPLACE INTO asset_metadata
                (ticker, name, asset_type, first_date, last_date, total_records)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (ticker, info.get("name", ticker), info.get("type", "Unknown"), first_d, last_d, len(series)),
            )

    conn.close()


def load_prices_from_sqlite(
    db_path: Path,
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query normalized SQLite daily_prices and pivot back into a wide DataFrame
    indexed by datetime.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    query = "SELECT date, ticker, adj_close FROM daily_prices WHERE 1=1"
    params: List[str] = []

    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    query += " ORDER BY date ASC"

    df_long = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df_long.empty:
        return pd.DataFrame()

    df_wide = df_long.pivot(index="date", columns="ticker", values="adj_close")
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide.sort_index(inplace=True)

    if tickers:
        avail = [t for t in tickers if t in df_wide.columns]
        df_wide = df_wide[avail]

    return df_wide


@dataclass
class MarketDataBundle:
    """Cleaned market price/yield dataset with missing-value audit metadata."""

    prices: pd.DataFrame
    missing_counts_raw: Dict[str, int]
    start_date: str
    end_date: str
    n_trading_days: int


def fetch_market_data(
    tickers: List[str],
    start: str = "2012-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> MarketDataBundle:
    """
    Download adjusted daily close prices / yield indices from Yahoo Finance
    or load from local SQLite / CSV cache, perform explicit missing-value diagnostics,
    forward-fill isolated holiday mismatches (e.g., US bond market holidays vs NYSE equity trading days),
    and drop any leading incomplete rows.
    """
    loaded_from_db = False
    raw_close: Optional[pd.DataFrame] = None
    missing_raw: Dict[str, int] = {}

    # 1. Attempt loading from local SQLite database first
    if db_path is not None and db_path.exists() and not force_refresh:
        try:
            cached_db = load_prices_from_sqlite(db_path, tickers=tickers, start=start, end=end)
            if not cached_db.empty and all(t in cached_db.columns for t in tickers):
                raw_close = cached_db
                missing_raw = {col: int(raw_close[col].isna().sum()) for col in raw_close.columns}
                loaded_from_db = True
        except Exception:
            loaded_from_db = False

    # 2. Fall back to CSV cache if SQLite not hit
    if not loaded_from_db:
        if cache_path is not None and cache_path.exists() and not force_refresh:
            raw_close = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            missing_raw = {col: int(raw_close[col].isna().sum()) for col in raw_close.columns}
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downloaded = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
            if isinstance(downloaded.columns, pd.MultiIndex):
                raw_close = downloaded["Close"].copy()
            else:
                raw_close = downloaded.copy()

            # Preserve requested column ordering
            available = [t for t in tickers if t in raw_close.columns]
            raw_close = raw_close[available].sort_index()
            missing_raw = {col: int(raw_close[col].isna().sum()) for col in raw_close.columns}

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                raw_close.to_csv(cache_path)

    # Clean missing values: forward fill holiday gaps, drop incomplete leading rows
    cleaned = raw_close.ffill().dropna()

    # 3. Persist to SQLite database if requested and not already loaded from it
    if db_path is not None and not loaded_from_db:
        try:
            save_prices_to_sqlite(cleaned, db_path)
        except Exception as e:
            warnings.warn(f"Failed to persist prices to SQLite at {db_path}: {e}")

    return MarketDataBundle(
        prices=cleaned,
        missing_counts_raw=missing_raw,
        start_date=str(cleaned.index[0].date()),
        end_date=str(cleaned.index[-1].date()),
        n_trading_days=len(cleaned),
    )


def test_pair_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
    use_log: bool = True,
) -> Dict[str, float]:
    """
    Perform Engle-Granger two-step cointegration test and Augmented Dickey-Fuller (ADF)
    unit-root test on the OLS spread between two asset price series.
    """
    ya = np.log(series_a.to_numpy(dtype=float)) if use_log else series_a.to_numpy(dtype=float)
    yb = np.log(series_b.to_numpy(dtype=float)) if use_log else series_b.to_numpy(dtype=float)

    X = sm.add_constant(yb)
    ols_res = sm.OLS(ya, X).fit()
    alpha_hat = float(ols_res.params[0])
    beta_hat = float(ols_res.params[1])
    spread = ya - (alpha_hat + beta_hat * yb)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg_tstat, eg_pval, eg_crit = coint(ya, yb)
        adf_out = adfuller(spread, autolag="AIC")

    return {
        "alpha": alpha_hat,
        "beta": beta_hat,
        "r_squared": float(ols_res.rsquared),
        "eg_tstat": float(eg_tstat),
        "eg_pvalue": float(eg_pval),
        "eg_crit_5pct": float(eg_crit[1]),
        "adf_tstat": float(adf_out[0]),
        "adf_pvalue": float(adf_out[1]),
    }


def construct_cointegrated_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    train_End_idx: int,
    use_log: bool = True,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Construct a cointegrated spread strictly fitting hedge parameters (alpha, beta)
    ONLY on the chronological training window [:train_End_idx] to eliminate look-ahead bias.
    """
    if train_End_idx < 30 or train_End_idx >= len(series_a):
        raise ValueError("Invalid chronological split index for training hedge ratio.")

    ya_full = np.log(series_a) if use_log else series_a.copy()
    yb_full = np.log(series_b) if use_log else series_b.copy()

    ya_train = ya_full.iloc[:train_End_idx]
    yb_train = yb_full.iloc[:train_End_idx]

    coint_stats = test_pair_cointegration(
        series_a.iloc[:train_End_idx],
        series_b.iloc[:train_End_idx],
        use_log=use_log,
    )
    spread = ya_full - (coint_stats["alpha"] + coint_stats["beta"] * yb_full)
    spread.name = f"Spread_{series_a.name}_{series_b.name}"
    return spread, coint_stats


def construct_rolling_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    lookback: int = 252,
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward (strictly causal) rolling OLS hedge ratio estimation on window [t - lookback + 1, t].
    At day t, hedge ratio beta_t and spread X_t use ONLY data available up to close of day t.
    """
    ya = np.log(series_a.to_numpy(dtype=float)) if use_log else series_a.to_numpy(dtype=float)
    yb = np.log(series_b.to_numpy(dtype=float)) if use_log else series_b.to_numpy(dtype=float)
    n = len(ya)

    alphas = np.full(n, np.nan, dtype=np.float64)
    betas = np.full(n, np.nan, dtype=np.float64)
    spreads = np.full(n, np.nan, dtype=np.float64)

    for t in range(lookback - 1, n):
        win_a = ya[t - lookback + 1 : t + 1]
        win_b = yb[t - lookback + 1 : t + 1]
        b_mean = float(np.mean(win_b))
        a_mean = float(np.mean(win_a))
        var_b = float(np.dot(win_b - b_mean, win_b - b_mean))
        cov_ab = float(np.dot(win_b - b_mean, win_a - a_mean))
        beta_t = cov_ab / var_b if var_b > 1e-14 else 1.0
        alpha_t = a_mean - beta_t * b_mean

        alphas[t] = alpha_t
        betas[t] = beta_t
        spreads[t] = ya[t] - (alpha_t + beta_t * yb[t])

    return pd.DataFrame(
        {
            "log_a": ya,
            "log_b": yb,
            "alpha": alphas,
            "beta": betas,
            "spread": spreads,
        },
        index=series_a.index,
    )
