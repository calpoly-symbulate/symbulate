"""
Table-based probability spaces for Symbulate.

Defines:
- Empirical: a discrete distribution from an explicit (outcomes, probabilities) table
- LifeTable: curtate future lifetime K_x built from an SOA-style mortality table
             (columns: age, qx), layered on top of Empirical
"""
import sys
sys.path.insert(0, "/home/claude/symbulate")

import numpy as np
import pandas as pd
from scipy import stats

from symbulate.distributions import Distribution


class Empirical(Distribution):
    """Discrete distribution defined by an explicit table of outcomes and probabilities.

    Parameters
    ----------
    outcomes : array_like of int
        The possible outcomes. Must be integer-valued (a requirement of
        scipy's underlying rv_discrete machinery).
    probabilities : array_like of float
        Probability associated with each outcome, in the same order as
        `outcomes`. Renormalized to sum to 1 if they don't already
        (e.g., due to floating point rounding in a source table).

    Examples
    --------
    >>> X = Empirical([1, 2, 3], [0.2, 0.3, 0.5])
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, outcomes, probabilities):
        outcomes = np.asarray(outcomes)
        probabilities = np.asarray(probabilities, dtype=float)

        if len(outcomes) != len(probabilities):
            raise ValueError(
                f"outcomes and probabilities must be the same length "
                f"(got {len(outcomes)} and {len(probabilities)})"
            )
        if np.any(probabilities < 0):
            raise ValueError("probabilities cannot be negative")
        if not np.issubdtype(outcomes.dtype, np.integer):
            # rv_discrete requires integer support; try a clean cast,
            # but fail loudly rather than silently truncating real values
            if np.allclose(outcomes, np.round(outcomes)):
                outcomes = outcomes.astype(int)
            else:
                raise ValueError(
                    "Empirical currently requires integer-valued outcomes "
                    "(a limitation of scipy's rv_discrete). Map non-integer "
                    "outcomes to integer codes and keep a lookup table on "
                    "the side if you need the original values."
                )

        total = probabilities.sum()
        if total <= 0:
            raise ValueError("probabilities must sum to a positive number")
        probabilities = probabilities / total  # normalize defensively

        self.outcomes_ = outcomes
        self.probabilities_ = probabilities

        custom = stats.rv_discrete(name="empirical", values=(outcomes, probabilities))
        super().__init__({}, custom, discrete=True)


class LifeTable(Empirical):
    """Curtate future lifetime distribution K_x from an SOA-style mortality table.

    Given a mortality table with columns for age and q_x (probability of
    death within one year, given alive at the start of the year), this
    builds the distribution of K_x = curtate future lifetime for a life
    currently age `age`: the number of *complete* future years lived.

    P(K_x = k) = (l_{x+k} - l_{x+k+1}) / l_x

    Parameters
    ----------
    table : str or pandas.DataFrame
        Path to a CSV file, or an already-loaded DataFrame, with at
        least an age column and a qx column.
    age : int
        The current age of the life being modeled. Must be present in
        the table and less than the table's final (omega) age.
    qx_column : str, optional
        Name of the qx column. Default 'qx'.
    age_column : str, optional
        Name of the age column. Default 'age'.

    Attributes
    ----------
    age : int
        The entry age used to build this distribution.
    lx_ : numpy.ndarray
        Survivorship column computed from qx, starting at a radix of
        100,000 at `age`.

    Examples
    --------
    >>> X = LifeTable("mortality_table.csv", age=45)  # doctest: +SKIP
    >>> X.draw()  # doctest: +SKIP
    38
    """

    def __init__(self, table, age, qx_column="qx", age_column="age"):
        df = table if isinstance(table, pd.DataFrame) else pd.read_csv(table)

        for col in (age_column, qx_column):
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in table. "
                    f"Available columns: {list(df.columns)}"
                )

        df = df.sort_values(age_column).reset_index(drop=True)

        if age not in set(df[age_column]):
            raise ValueError(
                f"age={age} not found in the table's age column "
                f"(range: {df[age_column].min()}-{df[age_column].max()})"
            )

        # restrict to ages >= entry age
        df = df[df[age_column] >= age].reset_index(drop=True)

        qx = df[qx_column].to_numpy(dtype=float)
        if np.any((qx < 0) | (qx > 1)):
            raise ValueError("qx values must be between 0 and 1")
        if qx[-1] != 1.0:
            raise ValueError(
                "The table's final qx must be 1.0 (everyone dies by the "
                "table's last age, i.e. the table must reach its omega age)."
            )

        # Build l_x via a radix of 100,000 at the entry age. l_x must have
        # one MORE entry than qx: l_x[i] is survivors at the start of the
        # i-th table row, and l_x[i+1] = l_x[i] * (1 - qx[i]) applies every
        # row's qx, including the terminal row (qx = 1.0), which is what
        # forces l_x down to exactly 0 at the end of the table.
        lx = np.empty(len(qx) + 1)
        lx[0] = 100_000.0
        for i in range(len(qx)):
            lx[i + 1] = lx[i] * (1 - qx[i])

        k = np.arange(len(qx))
        probs = (lx[:-1] - lx[1:]) / lx[0]

        self.age = age
        self.lx_ = lx

        super().__init__(k, probs)
