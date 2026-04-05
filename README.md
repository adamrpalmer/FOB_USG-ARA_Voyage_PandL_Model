# FOB USG–ARA Voyage P&L Model

A stochastic decision-support tool for evaluating the commercial viability of WTI crude arbitrage from the US Gulf Coast to Northwest Europe.

---

## Overview

Given live WTI Houston FOB prices and the prevailing WTI/Brent spread, the model simulates voyage P&L for a single Aframax shipment across thousands of historical market and operational scenarios. It outputs a P&L distribution, expected value, CVaR at the 5th percentile, and a risk-adjusted go/no-go recommendation based on an EV/CVaR₀.₀₅ ≥ 1.0 decision rule.

The model quantifies the stochastic arithmetic of an unhedged voyage. The trader overlays basis view, hedging strategy, and counterparty assessment on top. Basis and counterparty data are excluded due to proprietary data constraints; hedging is excluded because the model targets a risk warehousing trade structure.

---

## Trade Structure

| Parameter | Specification |
|---|---|
| Commodity | Crude Oil |
| Origin proxy | WTI Houston FOB |
| Destination proxy | Dated Brent |
| Route | USG → ARA |
| Vessel class | Aframax |
| Cargo size | 730,000 bbl / 95,000 mt |
| Incoterms | FOB Buy USG, CIF Sell ARA |
| Decision rule | EV/CVaR₀.₀₅ ≥ 1.0 |

**Timeline:** At *t* = 0, freight is fixed and WTI is purchased under a Sales and Purchase Agreement (SPA), with a Letter of Credit issued to the USG counterparty. The buy leg is priced as the 5-day average of WTI Houston FOB assessments around the Bill of Lading date. The vessel transits to ARA, where the sell leg floats to the 5-day average of Dated Brent assessments around discharge. Financing accrues from BL to settlement under SOFR plus a credit spread. Total exposure spans arbitrage decision to ARA counterparty settlement.

**P&L components:** Spread − Freight − Financing − Demurrage − Insurance − Port Fees

---

## Methodology

Non-deterministic voyage costs are simulated via regime-conditioned historical block bootstrap on a time-indexed market data matrix. Each simulation path is anchored to a historically similar WTI level and spread regime, so the model inherits the empirical joint distribution of market variables without parametric assumptions about their dependence structure.

The matrix traces six nodes through the trade timeline — from arbitrage decision to settlement — reading market and operational variables at each. Variables unavailable for bootstrapping due to confidentiality (demurrage, financing spread, port fees, cargo loss) are parametrised with triangular distributions under conservative assumptions.

Full methodology is documented separately.

**Market data inputs (Bloomberg):** Dated Brent, WTI Houston FOB, TD25 Flat Rate, WS Quote, SOFR, FX, AIS vessel tracking.

---

## Getting Started

```bash
git clone <repo-url>
cd FOB_USG-ARA_Voyage_PL_Model
python -m venv .venv && .venv\Scripts\activate  # Windows
source .venv/bin/activate                        # macOS/Linux
pip install -r requirements.txt
python main.py
```

Market inputs and simulation parameters are configured in `src/config.py`. Results are written to `outputs/`.

---

## Project Structure

```
src/
├── config.py       # Parameters and thresholds
├── pnl.py          # P&L sub-functions
├── simulate.py     # Monte-Carlo engine
└── report.py       # Output generation
data/
├── raw/            # Source market data
└── processed/      # Market data matrix
outputs/            # Simulation results
```

---

## Status

Active development. t1 selection method and data acquisition are the current workflows; core methodology is complete.

Second-year undergraduate research project — University of Nottingham.

---

## Author

**Adam Palmer** · University of Nottingham

---

## Licence

See [LICENSE](./LICENSE).