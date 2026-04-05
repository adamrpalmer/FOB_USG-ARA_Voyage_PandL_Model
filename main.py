"""
FOB USG-NWE Voyage P&L Model — CLI entry point.

Usage
-----
    python main.py --matrix data/matrix.csv --wti 72.50 --spread 3.20

Required arguments
------------------
    --matrix  PATH   Path to market data matrix (CSV or Excel).
                     Must have a DatetimeIndex and columns matching MATRIX_COLS
                     in src/config.py. Financial columns (Dated Brent,
                     WTI Houston FOB, WS Quote, TD25 Flat Rate, SOFR, FX)
                     carry NaN on non-trading days. Vessel tracking columns
                     (t_*) carry NaN between observations.

    --wti     FLOAT  Current WTI Houston FOB price ($/bbl) at arbitrage decision.
    --spread  FLOAT  Current Brent-WTI spread ($/bbl). Positive = Brent > WTI.

Optional arguments
------------------
    --n       INT    Number of Monte-Carlo simulations (default 10,000).
    --seed    INT    Random seed for reproducibility.
    --surface        Also compute and display the decision surface.
    --save    DIR    Directory to write plots instead of displaying interactively.

t1 Selector
-----------
The block-start selection algorithm (t1) is not yet finalised. By default the
model uses RandomT1Selector, which picks t1 uniformly at random from eligible
matrix dates. This allows structural end-to-end testing but does not produce
regime-conditional results.

To plug in a finished selector, import it and instantiate it inside the
`_build_selector` function in this file.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from src.config import MATRIX_COLS, DEFAULT_N_SIMS
from src.simulate import outer_loop
from src.report import print_summary, plot_pnl_distribution, plot_decision_surface
from src.t1_selector import RandomT1Selector, T1Selector


# ── Matrix loading ────────────────────────────────────────────────────────────

def load_matrix(path: str) -> pd.DataFrame:
    """
    Load the market data matrix from a CSV or Excel file.

    Expected format
    ---------------
    - First column: date index (parsed as DatetimeIndex, ascending order).
    - Remaining columns: exactly the names listed in MATRIX_COLS (src/config.py).
    - Financial columns (Dated Brent, WTI Houston FOB, SOFR, FX): NaN on
      weekends and public holidays.
    - Vessel tracking columns (t_*): NaN between observations; values present
      when a voyage record is available for that date.

    Supported file types: .csv, .xlsx, .xls
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, index_col=0, parse_dates=True)
    else:
        sys.exit(f"[error] Unsupported matrix file format '{ext}'. Use CSV or Excel.")

    df.index = pd.to_datetime(df.index)
    df       = df.sort_index()

    missing = [c for c in MATRIX_COLS if c not in df.columns]
    if missing:
        sys.exit(
            f"[error] Matrix is missing required columns: {missing}\n"
            f"Expected columns: {MATRIX_COLS}"
        )

    return df[MATRIX_COLS]


# ── Selector construction ─────────────────────────────────────────────────────

def _build_selector() -> T1Selector:
    """
    Return the t1 selector to use.

    When the t1 selection methodology is finalised, replace the body of this
    function with instantiation of the concrete selector, e.g.:

        from src.t1_selector import MyRegimeSelector
        return MyRegimeSelector(n_neighbours=20)

    Until then, RandomT1Selector allows structural testing.
    """
    return RandomT1Selector()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FOB USG-NWE Voyage P&L Monte-Carlo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matrix",  required=True,
                   help="Path to market data matrix (CSV or Excel).")
    p.add_argument("--wti",     type=float, required=True,
                   help="Current WTI Houston FOB price ($/bbl).")
    p.add_argument("--spread",  type=float, required=True,
                   help="Current Brent-WTI spread ($/bbl).")
    p.add_argument("--n",       type=int, default=DEFAULT_N_SIMS,
                   help=f"Number of simulations (default {DEFAULT_N_SIMS:,}).")
    p.add_argument("--seed",    type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--surface", action="store_true",
                   help="Compute and display the EV/CVaR decision surface.")
    p.add_argument("--save",    type=str, default=None,
                   help="Directory path to save plots (skips interactive display).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load matrix ───────────────────────────────────────────────────────────
    print(f"Loading market data matrix: {args.matrix}")
    matrix = load_matrix(args.matrix)
    print(
        f"Matrix loaded — {len(matrix):,} rows  "
        f"({matrix.index[0].date()} → {matrix.index[-1].date()})"
    )
    print()

    # ── Build t1 selector ─────────────────────────────────────────────────────
    selector = _build_selector()
    print(f"t1 selector : {type(selector).__name__}")
    print()

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"Running {args.n:,} simulations...")
    sim_df = outer_loop(
        matrix=matrix,
        wti_level=args.wti,
        spread=args.spread,
        t1_selector=selector,
        n_sims=args.n,
        seed=args.seed,
    )
    print(f"Complete — {len(sim_df):,} scenarios generated.")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(sim_df, args.wti, args.spread)

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_dist = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_dist = os.path.join(args.save, "pnl_distribution.png")

    plot_pnl_distribution(
        sim_df,
        wti_level=args.wti,
        spread=args.spread,
        show=(args.save is None),
        save_path=save_dist,
    )

    if args.surface:
        save_surf = None
        if args.save:
            save_surf = os.path.join(args.save, "decision_surface.png")
        plot_decision_surface(
            matrix=matrix,
            t1_selector=selector,
            show=(args.save is None),
            save_path=save_surf,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
