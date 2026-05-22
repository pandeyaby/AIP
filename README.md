# 🤖 AI Industrial Revolution Portfolio

**145 companies across Core AI Stack, Optics & Photonics, Quantum Computing, and Utilities & Power.**

Monte Carlo-driven quarterly sector rotation engine. Identifies optimal sector tilts every quarter to maximize risk-adjusted returns.

## 🔴 Live Dashboard
👉 **[pandeyaby.github.io/AIP](https://pandeyaby.github.io/AIP)**

## Q2 2026 Recommendation
- **Best Sector Tilt:** Utilities & Power (Sharpe 7.62)
- **Expected Return:** 138.6% annualized
- **P5 → P95 Range:** 62% → 91%
- 8,000 Monte Carlo simulations across 142 valid tickers

## How It Works

1. **Data:** Pulls 2 years of weekly prices via `yfinance` for all 145 companies
2. **Returns:** Blended signal — 60% recent momentum (last quarter) + 40% long-term mean
3. **Monte Carlo:** 8,000 random portfolio simulations → efficient frontier
4. **Sector Tilt:** For each of 4 sectors, optimizes weights with that sector at 30–50% allocation
5. **Rebalance:** Run quarterly → new allocation weights for next 3 months

## Companies
| Sector | Count | Avg Moat |
|--------|-------|----------|
| Core AI Stack | 86 | 3.8 |
| Optics & Photonics | 21 | 2.9 |
| Quantum Computing | 6 | 2.0 |
| Utilities & Power | 32 | 3.3 |

Added beyond spreadsheet: ARM, TSM, ASML, SentinelOne, Rubrik, The Trade Desk, Axon, GitLab

## Run Locally
```bash
pip install fastapi uvicorn yfinance numpy pandas scipy
python app.py
# → http://localhost:8765
```

## Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---
*Not investment advice. Quantitative research tool only.*
