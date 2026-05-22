"""
Monte Carlo Portfolio Engine
AI Industrial Revolution — Quarterly Sector Rotation Optimizer
"""
import json
import sqlite3
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import minimize
from typing import Optional

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "portfolio.db"
COMPANIES_PATH = BASE_DIR / "companies.json"

RISK_FREE_RATE = 0.045  # ~current T-bill rate annualized
TRADING_WEEKS = 52

# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quarter TEXT NOT NULL,
            ticker TEXT NOT NULL,
            weight REAL NOT NULL,
            sector TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quarter TEXT NOT NULL,
            best_sector_tilt TEXT,
            sharpe REAL,
            expected_return REAL,
            expected_vol REAL,
            p5_return REAL,
            p95_return REAL,
            n_valid_tickers INTEGER,
            run_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_current_quarter() -> str:
    now = datetime.now()
    q = (now.month - 1) // 3 + 1
    return f"{now.year}Q{q}"


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], period="2y", interval="1wk",
                 cache_hours=6) -> pd.DataFrame:
    """Fetch weekly closes for all tickers with SQLite cache."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now()
    results = {}
    to_fetch = []

    for t in tickers:
        row = conn.execute(
            "SELECT data, fetched_at FROM price_cache WHERE ticker=?", (t,)
        ).fetchone()
        if row:
            fetched_at = datetime.fromisoformat(row[1])
            age_h = (now - fetched_at).total_seconds() / 3600
            if age_h < cache_hours:
                try:
                    s = pd.Series(json.loads(row[0]))
                    s.index = pd.to_datetime(s.index)
                    results[t] = s
                    continue
                except Exception:
                    pass
        to_fetch.append(t)

    conn.close()

    if to_fetch:
        batch_size = 50
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i+batch_size]
            try:
                raw = yf.download(
                    batch, period=period, interval=interval,
                    auto_adjust=True, progress=False, threads=True
                )
                if raw.empty:
                    continue

                closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
                if isinstance(closes, pd.Series):
                    closes = closes.to_frame(name=batch[0])

                conn2 = sqlite3.connect(DB_PATH)
                for t in batch:
                    if t in closes.columns:
                        s = closes[t].dropna()
                        if len(s) >= 20:
                            results[t] = s
                            conn2.execute(
                                "INSERT OR REPLACE INTO price_cache VALUES (?,?,?)",
                                (t, json.dumps({str(k): v for k, v in s.items()}),
                                 now.isoformat())
                            )
                conn2.commit()
                conn2.close()
            except Exception as e:
                print(f"[fetch_prices] Error batch {i}: {e}")

    # Build DataFrame
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df.sort_index(inplace=True)
    return df


# ─── Return Calculations ──────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Weekly log returns, drop tickers with <40 weeks of data."""
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    # Keep only columns with enough history
    valid = returns.columns[returns.count() >= 40]
    return returns[valid].dropna()


def compute_expected_returns(returns: pd.DataFrame, momentum_weight=0.6) -> pd.Series:
    """
    Blend of recent momentum (last 13 weeks) and long-term mean.
    Annualized weekly log returns.
    """
    long_mean = returns.mean() * TRADING_WEEKS
    recent = returns.tail(13).mean() * TRADING_WEEKS
    blended = momentum_weight * recent + (1 - momentum_weight) * long_mean
    return blended


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def run_monte_carlo(
    returns: pd.DataFrame,
    n_simulations: int = 8000,
    sector_map: Optional[dict] = None,
) -> dict:
    """
    Run Monte Carlo simulation.
    Returns portfolio cloud + best unconstrained allocation.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    if n < 5:
        return {"error": "Not enough tickers for simulation"}

    mu = compute_expected_returns(returns)
    cov = returns.cov() * TRADING_WEEKS  # annualize

    port_returns, port_vols, port_sharpes = [], [], []
    weight_matrix = []

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        w = rng.dirichlet(np.ones(n))
        r = float(np.dot(w, mu))
        v = float(np.sqrt(w @ cov.values @ w))
        s = (r - RISK_FREE_RATE) / v if v > 0 else 0

        port_returns.append(r)
        port_vols.append(v)
        port_sharpes.append(s)
        weight_matrix.append(w)

    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
    port_sharpes = np.array(port_sharpes)
    weight_matrix = np.array(weight_matrix)

    # Best Sharpe portfolio
    best_idx = int(np.argmax(port_sharpes))
    best_weights = weight_matrix[best_idx]

    # Min volatility portfolio
    minvol_idx = int(np.argmin(port_vols))

    return {
        "tickers": tickers,
        "n_simulations": n_simulations,
        "cloud": {
            "returns": port_returns.tolist(),
            "vols": port_vols.tolist(),
            "sharpes": port_sharpes.tolist(),
        },
        "best_sharpe": {
            "weights": dict(zip(tickers, best_weights.tolist())),
            "return": float(port_returns[best_idx]),
            "vol": float(port_vols[best_idx]),
            "sharpe": float(port_sharpes[best_idx]),
        },
        "min_vol": {
            "weights": dict(zip(tickers, weight_matrix[minvol_idx].tolist())),
            "return": float(port_returns[minvol_idx]),
            "vol": float(port_vols[minvol_idx]),
            "sharpe": float(port_sharpes[minvol_idx]),
        },
        "percentiles": {
            "p5_return": float(np.percentile(port_returns, 5)),
            "p25_return": float(np.percentile(port_returns, 25)),
            "p50_return": float(np.percentile(port_returns, 50)),
            "p75_return": float(np.percentile(port_returns, 75)),
            "p95_return": float(np.percentile(port_returns, 95)),
        },
    }


# ─── Sector Tilt Optimizer ────────────────────────────────────────────────────

def sector_tilt_analysis(
    returns: pd.DataFrame,
    companies: list[dict],
    sector_min: float = 0.30,
    sector_max: float = 0.55,
) -> dict:
    """
    For each sector, optimize portfolio with that sector constrained to
    [sector_min, sector_max] allocation. Returns best sector + optimal weights.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    mu = compute_expected_returns(returns).values
    cov = (returns.cov() * TRADING_WEEKS).values

    # Build sector index mapping
    sector_tickers = {}
    ticker_sector = {}
    for c in companies:
        t = c["ticker"]
        s = c["sector"]
        if t in tickers:
            sector_tickers.setdefault(s, []).append(t)
            ticker_sector[t] = s

    sectors = list(sector_tickers.keys())
    results = {}

    def neg_sharpe(w):
        r = float(np.dot(w, mu))
        v = float(np.sqrt(w @ cov @ w))
        return -(r - RISK_FREE_RATE) / v if v > 1e-9 else 0

    base_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 0.15)] * n  # max 15% per position

    for sector in sectors:
        idx = [tickers.index(t) for t in sector_tickers.get(sector, []) if t in tickers]
        if not idx:
            continue

        def sector_sum_lb(w, idx=idx):
            return sum(w[i] for i in idx) - sector_min

        def sector_sum_ub(w, idx=idx):
            return sector_max - sum(w[i] for i in idx)

        constraints = base_constraints + [
            {"type": "ineq", "fun": sector_sum_lb},
            {"type": "ineq", "fun": sector_sum_ub},
        ]

        w0 = np.ones(n) / n
        try:
            res = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success or res.fun < 0:
                w = res.x
                w = np.clip(w, 0, None)
                w /= w.sum()
                r = float(np.dot(w, mu))
                v = float(np.sqrt(w @ cov @ w))
                sharpe = (r - RISK_FREE_RATE) / v if v > 1e-9 else 0
                sector_allocation = sum(w[i] for i in idx)
                results[sector] = {
                    "sharpe": sharpe,
                    "return": r,
                    "vol": v,
                    "sector_allocation": float(sector_allocation),
                    "weights": {tickers[i]: float(w[i]) for i in range(n) if w[i] > 0.002},
                }
        except Exception as e:
            print(f"[sector_tilt] {sector} optimization error: {e}")

    if not results:
        return {"best_sector": None, "tilts": {}}

    best_sector = max(results, key=lambda s: results[s]["sharpe"])
    return {"best_sector": best_sector, "tilts": results}


# ─── Full Quarterly Simulation ─────────────────────────────────────────────────

def run_full_simulation(companies: list[dict], progress_cb=None) -> dict:
    """
    Main entry point. Returns complete quarterly allocation.
    """
    init_db()
    quarter = get_current_quarter()

    # Filter tradeable tickers (skip pre-rev / no-ticker)
    tradeable = [
        c for c in companies
        if c["ticker"] and c["ticker"] != "—"
        and c["moat"] >= 2  # include all with any moat
    ]

    tickers = [c["ticker"] for c in tradeable]
    if progress_cb:
        progress_cb({"step": "fetch_prices", "n": len(tickers)})

    prices = fetch_prices(tickers)
    if progress_cb:
        progress_cb({"step": "compute_returns", "n_prices": len(prices.columns)})

    if prices.empty:
        return {"error": "Could not fetch price data"}

    returns = compute_returns(prices)
    valid_tickers = list(returns.columns)

    if progress_cb:
        progress_cb({"step": "monte_carlo", "n_valid": len(valid_tickers)})

    # Lookup company data for valid tickers
    valid_companies = [c for c in tradeable if c["ticker"] in valid_tickers]

    # Run MC
    mc_result = run_monte_carlo(returns)
    if "error" in mc_result:
        return mc_result

    if progress_cb:
        progress_cb({"step": "sector_tilt"})

    # Sector tilt analysis
    tilt_result = sector_tilt_analysis(returns, valid_companies)

    # Current quarter prices (last close) for position sizing
    latest_prices = {t: float(prices[t].iloc[-1]) for t in valid_tickers if t in prices.columns}

    # Build final allocation from best sector tilt or best MC Sharpe
    best_sector = tilt_result.get("best_sector")
    if best_sector and best_sector in tilt_result["tilts"]:
        final_weights = tilt_result["tilts"][best_sector]["weights"]
    else:
        final_weights = mc_result["best_sharpe"]["weights"]

    # Enrich with company metadata
    ticker_meta = {c["ticker"]: c for c in valid_companies}
    allocations = []
    for t, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        if w < 0.001:
            continue
        meta = ticker_meta.get(t, {})
        allocations.append({
            "ticker": t,
            "company": meta.get("company", t),
            "sector": meta.get("sector", ""),
            "layer": meta.get("layer", ""),
            "moat": meta.get("moat", 0),
            "valuation": meta.get("valuation", ""),
            "weight": round(w * 100, 2),
            "price": latest_prices.get(t),
        })

    # Sector summary
    sector_weights = {}
    for a in allocations:
        s = a["sector"]
        sector_weights[s] = sector_weights.get(s, 0) + a["weight"]

    # Tilt comparison table
    tilt_comparison = []
    for sector, data in tilt_result.get("tilts", {}).items():
        tilt_comparison.append({
            "sector": sector,
            "sharpe": round(data["sharpe"], 3),
            "expected_return_pct": round(data["return"] * 100, 1),
            "expected_vol_pct": round(data["vol"] * 100, 1),
            "sector_allocation_pct": round(data["sector_allocation"] * 100, 1),
            "is_best": sector == best_sector,
        })
    tilt_comparison.sort(key=lambda x: -x["sharpe"])

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM allocations WHERE quarter=?", (quarter,))
    for a in allocations:
        conn.execute(
            "INSERT INTO allocations (quarter, ticker, weight, sector) VALUES (?,?,?,?)",
            (quarter, a["ticker"], a["weight"], a["sector"])
        )
    mc = mc_result
    best = mc["best_sharpe"]
    conn.execute(
        "INSERT INTO simulations (quarter, best_sector_tilt, sharpe, expected_return, "
        "expected_vol, p5_return, p95_return, n_valid_tickers) VALUES (?,?,?,?,?,?,?,?)",
        (quarter, best_sector, best["sharpe"], best["return"], best["vol"],
         mc["percentiles"]["p5_return"], mc["percentiles"]["p95_return"], len(valid_tickers))
    )
    conn.commit()
    conn.close()

    return {
        "quarter": quarter,
        "n_companies_total": len(companies),
        "n_valid_tickers": len(valid_tickers),
        "best_sector_tilt": best_sector,
        "allocations": allocations,
        "sector_summary": sector_weights,
        "tilt_comparison": tilt_comparison,
        "mc_summary": {
            "n_simulations": mc["n_simulations"],
            "best_sharpe": round(best["sharpe"], 3),
            "expected_return_pct": round(best["return"] * 100, 1),
            "expected_vol_pct": round(best["vol"] * 100, 1),
            "p5_return_pct": round(mc["percentiles"]["p5_return"] * 100, 1),
            "p50_return_pct": round(mc["percentiles"]["p50_return"] * 100, 1),
            "p95_return_pct": round(mc["percentiles"]["p95_return"] * 100, 1),
        },
        "mc_cloud": {
            "returns": [round(r * 100, 2) for r in mc["cloud"]["returns"][::4]],  # downsample 4x
            "vols": [round(v * 100, 2) for v in mc["cloud"]["vols"][::4]],
            "sharpes": [round(s, 3) for s in mc["cloud"]["sharpes"][::4]],
        },
    }


# ─── History ──────────────────────────────────────────────────────────────────

def get_allocation_history() -> list[dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT quarter, ticker, weight, sector
        FROM allocations ORDER BY quarter DESC, weight DESC
    """).fetchall()
    sims = conn.execute("""
        SELECT quarter, best_sector_tilt, sharpe, expected_return, expected_vol,
               p5_return, p95_return, n_valid_tickers, run_at
        FROM simulations ORDER BY quarter DESC
    """).fetchall()
    conn.close()

    quarters = {}
    for q, t, w, s in rows:
        quarters.setdefault(q, {"quarter": q, "allocations": [], "simulation": None})
        quarters[q]["allocations"].append({"ticker": t, "weight": w, "sector": s})

    for q, tilt, sharpe, ret, vol, p5, p95, n, run_at in sims:
        if q in quarters:
            quarters[q]["simulation"] = {
                "best_sector_tilt": tilt, "sharpe": sharpe,
                "expected_return": ret, "expected_vol": vol,
                "p5_return": p5, "p95_return": p95,
                "n_valid_tickers": n, "run_at": run_at,
            }

    return list(quarters.values())
