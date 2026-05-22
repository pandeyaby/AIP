"""
AI Industrial Revolution Portfolio — FastAPI Server
Quarterly Monte Carlo Sector Rotation Engine
"""
import json
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from engine import (
    init_db, get_current_quarter, run_full_simulation,
    get_allocation_history, fetch_prices, compute_returns,
    compute_expected_returns, COMPANIES_PATH
)

BASE_DIR = Path(__file__).parent

# Load companies
with open(COMPANIES_PATH) as f:
    COMPANIES = json.load(f)

app = FastAPI(title="AI Industrial Revolution Portfolio", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simulation state
_sim_state = {"running": False, "progress": [], "result": None, "error": None}


def _progress_cb(msg: dict):
    _sim_state["progress"].append(msg)
    print(f"[SIM] {msg}")


# ─── Static ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    return FileResponse(html_path)


# ─── Companies ────────────────────────────────────────────────────────────────

@app.get("/api/companies")
async def get_companies(sector: Optional[str] = None, min_moat: int = 0):
    cos = COMPANIES
    if sector:
        cos = [c for c in cos if c["sector"] == sector]
    if min_moat:
        cos = [c for c in cos if c["moat"] >= min_moat]
    return {
        "total": len(cos),
        "companies": cos,
        "sectors": list({c["sector"] for c in COMPANIES}),
    }


@app.get("/api/companies/summary")
async def companies_summary():
    sectors = {}
    for c in COMPANIES:
        s = c["sector"]
        sectors.setdefault(s, {"count": 0, "avg_moat": 0, "moats": []})
        sectors[s]["count"] += 1
        sectors[s]["moats"].append(c["moat"])
    for s in sectors:
        m = sectors[s]["moats"]
        sectors[s]["avg_moat"] = round(sum(m) / len(m), 1)
        del sectors[s]["moats"]
    return {"total": len(COMPANIES), "by_sector": sectors}


# ─── Prices ───────────────────────────────────────────────────────────────────

@app.get("/api/prices")
async def get_prices(tickers: Optional[str] = None):
    """Fetch latest prices. tickers = comma-separated or all."""
    t_list = tickers.split(",") if tickers else [c["ticker"] for c in COMPANIES if c["ticker"] != "—"]
    prices = fetch_prices(t_list, period="3mo", interval="1wk", cache_hours=1)
    if prices.empty:
        raise HTTPException(status_code=503, detail="Could not fetch prices")
    latest = {t: round(float(prices[t].iloc[-1]), 2) for t in prices.columns if not prices[t].isna().all()}
    prev = {t: round(float(prices[t].iloc[-2]), 2) for t in prices.columns if len(prices[t].dropna()) >= 2}
    changes = {t: round((latest[t] / prev[t] - 1) * 100, 2) for t in latest if t in prev}
    return {"prices": latest, "prev": prev, "week_pct_change": changes, "n": len(latest)}


# ─── Simulation ───────────────────────────────────────────────────────────────

@app.post("/api/simulate")
async def start_simulation(background_tasks: BackgroundTasks):
    if _sim_state["running"]:
        return {"status": "already_running", "progress": _sim_state["progress"]}

    _sim_state.update({"running": True, "progress": [], "result": None, "error": None})

    def _run():
        try:
            result = run_full_simulation(COMPANIES, progress_cb=_progress_cb)
            _sim_state["result"] = result
        except Exception as e:
            _sim_state["error"] = str(e)
            print(f"[SIM ERROR] {e}")
        finally:
            _sim_state["running"] = False

    background_tasks.add_task(_run)
    return {"status": "started", "quarter": get_current_quarter()}


@app.get("/api/simulate/status")
async def sim_status():
    return {
        "running": _sim_state["running"],
        "progress": _sim_state["progress"],
        "done": _sim_state["result"] is not None,
        "error": _sim_state["error"],
    }


@app.get("/api/simulate/result")
async def sim_result():
    if _sim_state["running"]:
        return {"status": "running"}
    if _sim_state["error"]:
        raise HTTPException(status_code=500, detail=_sim_state["error"])
    if _sim_state["result"] is None:
        raise HTTPException(status_code=404, detail="No simulation run yet. POST /api/simulate first.")
    return _sim_state["result"]


# ─── Allocation ───────────────────────────────────────────────────────────────

@app.get("/api/allocation/current")
async def current_allocation():
    """Return latest saved allocation from DB (without re-running simulation)."""
    history = get_allocation_history()
    if not history:
        raise HTTPException(status_code=404, detail="No allocations yet. Run simulation first.")
    latest = history[0]
    # Enrich with company data
    ticker_meta = {c["ticker"]: c for c in COMPANIES}
    for a in latest["allocations"]:
        meta = ticker_meta.get(a["ticker"], {})
        a["company"] = meta.get("company", a["ticker"])
        a["moat"] = meta.get("moat", 0)
        a["valuation"] = meta.get("valuation", "")
        a["layer"] = meta.get("layer", "")
    return latest


@app.get("/api/allocation/history")
async def allocation_history():
    history = get_allocation_history()
    if not history:
        raise HTTPException(status_code=404, detail="No allocation history yet.")
    return {"quarters": history}


# ─── Returns Analysis ─────────────────────────────────────────────────────────

@app.get("/api/returns")
async def sector_returns():
    """Last quarter's actual returns by sector."""
    tickers = [c["ticker"] for c in COMPANIES if c["ticker"] != "—"]
    prices = fetch_prices(tickers, period="6mo", interval="1wk", cache_hours=4)
    if prices.empty:
        raise HTTPException(status_code=503, detail="Could not fetch prices")

    returns = compute_returns(prices)
    mu = compute_expected_returns(returns)

    ticker_meta = {c["ticker"]: c for c in COMPANIES}
    sector_returns = {}
    ticker_returns = []

    for t in returns.columns:
        meta = ticker_meta.get(t, {})
        s = meta.get("sector", "unknown")
        r_annual = float(mu.get(t, 0))
        # Last 13 weeks (1 quarter)
        r_q = float(returns[t].tail(13).sum()) if t in returns.columns else 0
        sector_returns.setdefault(s, []).append(r_q)
        ticker_returns.append({
            "ticker": t,
            "company": meta.get("company", t),
            "sector": s,
            "quarterly_return_pct": round(r_q * 100, 2),
            "annual_expected_pct": round(r_annual * 100, 1),
            "moat": meta.get("moat", 0),
            "valuation": meta.get("valuation", ""),
        })

    sector_avg = {
        s: round(sum(v) / len(v) * 100, 2)
        for s, v in sector_returns.items() if v
    }

    ticker_returns.sort(key=lambda x: -x["quarterly_return_pct"])

    return {
        "sector_avg_quarterly_pct": sector_avg,
        "top_performers": ticker_returns[:20],
        "bottom_performers": ticker_returns[-10:],
        "all": ticker_returns,
    }


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("🚀 AI Industrial Revolution Portfolio")
    print(f"   Companies: {len(COMPANIES)}")
    print(f"   Quarter:   {get_current_quarter()}")
    print(f"   URL:       http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
