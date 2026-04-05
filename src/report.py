"""
Output generation for the FOB USG-NWE Voyage P&L Model.

Produces:
  - Console summary (EV, CVaR, EV/CVaR, component breakdown, trade decision)
  - P&L distribution histogram
  - Decision surface over a (WTI level, spread) grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .config import CVAR_ALPHA, DECISION_THRESHOLD


# ── Risk statistics ───────────────────────────────────────────────────────────

def compute_ev(pnl: pd.Series) -> float:
    """Expected profit across all simulations."""
    return float(pnl.mean())


def compute_cvar(pnl: pd.Series, alpha: float = CVAR_ALPHA) -> float:
    """
    CVaR_α (Expected Shortfall): mean of the α left-tail outcomes.
    Returned as a positive number representing the magnitude of expected loss.
    Returns NaN if the tail is empty.
    """
    threshold = pnl.quantile(alpha)
    tail = pnl[pnl <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def compute_decision_metric(ev: float, cvar: float) -> float:
    """
    EV / CVaR_α per the decision rule (§2).
    Returns inf when CVaR ≤ 0 (tail is profitable; no expected loss).
    """
    if np.isnan(cvar) or cvar <= 0.0:
        return float("inf")
    return ev / cvar


def trade_recommendation(ratio: float) -> str:
    """EXECUTE if EV/CVaR ≥ DECISION_THRESHOLD, else DO NOT EXECUTE."""
    return "EXECUTE" if ratio >= DECISION_THRESHOLD else "DO NOT EXECUTE"


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
) -> None:
    """Print a structured summary of simulation results to stdout."""
    pnl   = sim_df["pnl"]
    ev    = compute_ev(pnl)
    cvar  = compute_cvar(pnl)
    ratio = compute_decision_metric(ev, cvar)
    rec   = trade_recommendation(ratio)

    w = 62
    print("=" * w)
    print("  FOB USG-NWE Voyage P&L Model — Simulation Summary")
    print("=" * w)
    print(f"  Live inputs at t=0:")
    print(f"    WTI Houston FOB         : ${wti_level:>10.2f} /bbl")
    print(f"    Brent-WTI Spread        : ${spread:>10.2f} /bbl")
    print(f"  Bootstrapped from matrix (simulation means):")
    print(f"    WS Quote                : {sim_df['ws_quote'].mean():>10.1f} WS")
    print(f"    TD25 Flat Rate          : ${sim_df['td25_flat_rate'].mean():>10.2f} /mt")
    print(f"  Simulations               : {len(pnl):>10,}")
    print("-" * w)
    print(f"  Expected Profit  E[π]     : ${ev:>12,.0f}")
    print(f"  CVaR ({CVAR_ALPHA*100:.0f}%)                : ${-cvar:>12,.0f}"
          .replace("$-", "-$"))
    print(f"  EV / CVaR                 : {ratio:>12.3f}")
    print(f"  Decision threshold        : {DECISION_THRESHOLD:>12.3f}")
    print("-" * w)
    print(f"  Decision  →  {rec}")
    print("=" * w)
    print()
    print("  P&L Component Breakdown (simulation means):")
    components = ["spread", "freight", "financing", "demurrage", "insurance", "port_fees"]
    for col in components:
        sign = "+" if sim_df[col].mean() >= 0 else ""
        print(f"    {col:<20}: {sign}${sim_df[col].mean():>12,.0f}")
    print(f"    {'net P&L':<20}:  ${ev:>12,.0f}")
    print()
    print(f"  Avg total trade exposure  : {sim_df['total_exposure_days'].mean():.1f} days")
    print(f"  Avg financing exposure    : {sim_df['financing_exposure_days'].mean():.1f} days")
    print()


# ── P&L distribution plot ─────────────────────────────────────────────────────

def plot_pnl_distribution(
    sim_df: pd.DataFrame,
    wti_level: float,
    spread: float,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Histogram of simulated P&L with EV, VaR, and CVaR marked.
    Values displayed in $M for readability.
    """
    pnl_m    = sim_df["pnl"] / 1e6
    ev_m     = pnl_m.mean()
    var_m    = float(pnl_m.quantile(CVAR_ALPHA))
    tail     = pnl_m[pnl_m <= var_m]
    cvar_m   = float(-tail.mean()) if not tail.empty else float("nan")

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(pnl_m, bins=120, color="#2c7bb6", alpha=0.82, edgecolor="none",
            density=True, label="_nolegend_")

    ax.axvline(ev_m, color="#d7191c", lw=1.8,
               label=f"E[π] = ${ev_m:+.2f}M")
    ax.axvline(var_m, color="#f4a518", lw=1.5, ls="--",
               label=f"VaR {CVAR_ALPHA*100:.0f}% = ${var_m:.2f}M")
    ax.axvspan(pnl_m.min() - 0.05, var_m, alpha=0.18, color="#f4a518",
               label=f"CVaR {CVAR_ALPHA*100:.0f}% = ${cvar_m:.2f}M (expected loss)")
    ax.axvline(0, color="black", lw=0.8, ls=":")

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fM"))
    ax.set_xlabel("Voyage P&L")
    ax.set_ylabel("Density")
    ax.set_title(
        f"FOB USG-NWE Voyage P&L Distribution\n"
        f"WTI: ${wti_level:.2f}/bbl  |  Brent-WTI Spread: ${spread:.2f}/bbl  |  "
        f"n = {len(sim_df):,}"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ── Decision surface plot ─────────────────────────────────────────────────────

def plot_decision_surface(
    matrix: pd.DataFrame,
    t1_selector,
    wti_levels: np.ndarray | None = None,
    spreads: np.ndarray | None = None,
    n_sims: int = 2_000,
    seed: int | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Contour plot of EV/CVaR over a grid of (WTI level, Brent-WTI spread).
    The decision boundary (EV/CVaR = DECISION_THRESHOLD) is drawn as a dashed line.
    WS Quote and TD25 Flat Rate are bootstrapped from the matrix per simulation.

    Parameters
    ----------
    wti_levels  : 1-D array of WTI price points to evaluate ($/bbl).
    spreads     : 1-D array of Brent-WTI spread points to evaluate ($/bbl).
    n_sims      : Simulations per grid cell (lower for speed; 2,000 is sufficient
                  for surface shape at the cost of some noise).
    """
    from .simulate import outer_loop   # avoid circular import at module level

    if wti_levels is None:
        wti_levels = np.linspace(55, 95, 12)
    if spreads is None:
        spreads = np.linspace(-3.0, 10.0, 12)

    ratios = np.full((len(spreads), len(wti_levels)), np.nan)

    total = len(spreads) * len(wti_levels)
    done  = 0
    print(f"Computing decision surface ({total} grid cells, {n_sims:,} sims each)...")

    for i, sp in enumerate(spreads):
        for j, wti in enumerate(wti_levels):
            sim = outer_loop(
                matrix=matrix,
                wti_level=wti,
                spread=sp,
                t1_selector=t1_selector,
                n_sims=n_sims,
                seed=seed,
            )
            ev    = compute_ev(sim["pnl"])
            cvar  = compute_cvar(sim["pnl"])
            ratios[i, j] = compute_decision_metric(ev, cvar)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} cells complete", end="\r")

    print()

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(wti_levels, spreads, ratios, levels=20, cmap="RdYlGn",
                     vmin=0, vmax=max(2.0, float(np.nanmax(ratios))))
    ax.contour(wti_levels, spreads, ratios, levels=[DECISION_THRESHOLD],
               colors="black", linewidths=1.8, linestyles="--")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("EV / CVaR\u2080.\u2080\u2085", fontsize=10)

    ax.set_xlabel("WTI Houston FOB ($/bbl)")
    ax.set_ylabel("Brent-WTI Spread ($/bbl)")
    ax.set_title(
        "Decision Surface: EV / CVaR\u2080.\u2080\u2085\n"
        f"(dashed line = threshold {DECISION_THRESHOLD:.1f})"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
