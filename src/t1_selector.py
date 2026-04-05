"""
t1 Selection Interface for the FOB USG-NWE Voyage P&L Model.

The market data matrix and the t1 selection algorithm are the two components
that are plugged in when data and methodology become available. This module
defines the interface that any t1 selector must satisfy, and provides a
random fallback for end-to-end testing of the rest of the simulation system.

Usage
-----
Implement a subclass of T1Selector and pass an instance to outer_loop():

    class MySelector(T1Selector):
        def select(self, matrix, wti_level, spread, rng):
            # ... regime-matching logic here ...
            return chosen_date

    sim_df = outer_loop(matrix, wti, spread, ws, td25,
                        t1_selector=MySelector())

The random fallback is used automatically when no selector is supplied and
the matrix is present, allowing structural testing before the methodology
is finalised.
"""

from __future__ import annotations

import abc
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import COL_DATED_BRENT, COL_WTI_HOUSTON, MAX_BLOCK_DAYS


class T1Selector(abc.ABC):
    """
    Abstract base for all t1 selection strategies.

    A selector receives the market data matrix and the live inputs at the
    arbitrage decision (t=0), and returns a single historical date t1 from
    which the inner loop will begin tracing the matrix block.

    Contract
    --------
    - The returned date must be a valid index entry in `matrix`.
    - The returned date must be early enough that the full block (up to
      MAX_BLOCK_DAYS ahead) remains within the matrix.
    - The selector may use `rng` for any stochastic choices, so that results
      are reproducible when the outer loop seeds the generator.
    """

    @abc.abstractmethod
    def select(
        self,
        matrix: pd.DataFrame,
        wti_level: float,
        spread: float,
        rng: np.random.Generator,
    ) -> pd.Timestamp:
        """
        Choose a historical block-start date t1.

        Parameters
        ----------
        matrix    : Market data matrix (DatetimeIndex, ascending).
        wti_level : Current WTI Houston FOB price ($/bbl) at t=0.
        spread    : Current Brent–WTI spread ($/bbl) at t=0.
        rng       : NumPy random generator (seeded by outer loop).

        Returns
        -------
        pd.Timestamp — a date present in matrix.index.
        """

    # ── Shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _eligible_dates(matrix: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Dates that have both WTI and Brent values and leave enough room for a
        full block (MAX_BLOCK_DAYS) before the end of the matrix.
        """
        cutoff = matrix.index[-1] - timedelta(days=MAX_BLOCK_DAYS)
        valid_mask = (
            matrix[COL_WTI_HOUSTON].notna()
            & matrix[COL_DATED_BRENT].notna()
            & (matrix.index <= cutoff)
        )
        return matrix.index[valid_mask]


# ── Stub selector ─────────────────────────────────────────────────────────────

class UnimplementedT1Selector(T1Selector):
    """
    Placeholder raised when no selector has been assigned.

    Replace this with a concrete T1Selector subclass once the t1 selection
    methodology is decided. The error message documents exactly what the
    implementation must provide.
    """

    def select(
        self,
        matrix: pd.DataFrame,
        wti_level: float,
        spread: float,
        rng: np.random.Generator,
    ) -> pd.Timestamp:
        raise NotImplementedError(
            "t1 selection methodology has not yet been implemented.\n\n"
            "Implement a subclass of T1Selector and pass it to outer_loop() "
            "via the `t1_selector` argument.\n\n"
            "The selector must:\n"
            "  1. Accept (matrix, wti_level, spread, rng).\n"
            "  2. Return a pd.Timestamp present in matrix.index.\n"
            "  3. Only select dates within T1Selector._eligible_dates(matrix)\n"
            "     (i.e. both WTI and Brent non-NaN, and >= MAX_BLOCK_DAYS "
            "before the end of the matrix).\n\n"
            "For structural testing without a methodology, use RandomT1Selector."
        )


# ── Test-only fallback ────────────────────────────────────────────────────────

class RandomT1Selector(T1Selector):
    """
    Selects t1 uniformly at random from all eligible dates.

    Not a valid regime-conditional selector — exists only to allow end-to-end
    testing of the simulation infrastructure before the t1 methodology is
    finalised. Do not use for real trade decisions.
    """

    def select(
        self,
        matrix: pd.DataFrame,
        wti_level: float,
        spread: float,
        rng: np.random.Generator,
    ) -> pd.Timestamp:
        eligible = self._eligible_dates(matrix)
        if eligible.empty:
            raise ValueError(
                "No eligible t1 dates in matrix (need both WTI and Brent "
                f"non-NaN, at least {MAX_BLOCK_DAYS} days before matrix end)."
            )
        idx = rng.integers(0, len(eligible))
        return eligible[idx]
