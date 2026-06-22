"""Data structure for storing tabulated results.

This module defines a data structure, Table, that stores the
output of a .tabulate() operation.  Typically, Table stores
the possible outcomes and their counts or relative frequencies.
"""
from .base import Arithmetic


TABLE_TEMPLATE = '''
<table>
  <thead>
    <th width="80%">{outcome_column}</th>
    <th width="20%">{value_column}</th>
  </thead>
  <tbody>
    {table_body}
  </tbody>
</table>
'''


def _get_row_html(outcome, count):
    return "<tr><td>%s</td><td>%s</td></tr>" % (outcome, count)


class Table(dict, Arithmetic):
    """A tabulated summary of simulation outcomes.

    Stores outcome-count or outcome-relative-frequency pairs, typically
    produced by calling ``.tabulate()`` on a simulation result. Inherits
    from ``dict`` so standard dictionary operations work directly on
    outcomes and their values.

    Parameters
    ----------
    hash_map : dict
        Mapping from outcomes to their counts or frequencies.
    outcomes : list, optional
        Ordered list of all possible outcomes. Outcomes absent from
        ``hash_map`` are assigned a value of 0, and the table preserves
        this ordering. Defaults to ``None``.
    normalize : bool, optional
        If ``True``, divide each count by the total so entries become
        relative frequencies. Defaults to ``False``.
    outcome_column : str, optional
        Label for the outcomes column in the display.
        Defaults to ``"Outcome"``.

    Attributes
    ----------
    outcomes : list or None
        Ordered list of possible outcomes, if provided.
    outcome_column : str
        Label for the outcomes column.
    value_column : str
        ``"Relative Frequency"`` when ``normalize=True``,
        otherwise ``"Frequency"``.

    Examples
    --------
    >>> from symbulate import *
    >>> t = Table({'heads': 55, 'tails': 45})
    >>> t['heads']
    55

    >>> t_rel = Table({'heads': 55, 'tails': 45}, normalize=True)
    >>> round(t_rel['heads'], 2)
    0.55

    >>> t2 = Table({'A': 3}, outcomes=['A', 'B', 'C'])
    >>> t2['B']
    0

    >>> RV(Bernoulli(0.5)).sim(100).tabulate()  # doctest: +SKIP
    Outcome  Frequency
          0  48
          1  52
      Total  100
    """

    def __init__(self, hash_map, outcomes=None, normalize=False,
                 outcome_column="Outcome"):
        """Initialize a Table from a mapping of outcomes to counts."""
        self.outcomes = outcomes
        self.outcome_column = outcome_column
        if outcomes is None:
            for outcome, count in hash_map.items():
                self[outcome] = count
        else:
            for outcome in outcomes:
                self[outcome] = (
                    hash_map[outcome] if outcome in hash_map
                    else 0
                )
                
        if normalize:
            for key in self.ordered_keys():
                self[key] /= sum(hash_map.values())
            self.value_column = 'Relative Frequency'
        else:
            self.value_column = 'Frequency'
                
    def ordered_keys(self):
        """Return outcome keys in sorted or declaration order.

        Returns
        -------
        list
            Keys sorted in ascending order when ``outcomes`` was not
            specified, or in the original ``outcomes`` order otherwise.

        Examples
        --------
        >>> from symbulate import *
        >>> t = Table({'B': 2, 'A': 3})
        >>> t.ordered_keys()
        ['A', 'B']

        >>> t2 = Table({'B': 2, 'A': 3}, outcomes=['B', 'A'])
        >>> t2.ordered_keys()
        ['B', 'A']
        """
        # get keys in order
        if self.outcomes is None:
            keys = list(self.keys())
            try:
                keys.sort()
            except Exception:
                pass
        else:
            # preserve ordering of outcomes, if specified
            keys = self.outcomes

        return keys
    
    def __repr__(self):
        """Return a plain-text table representation."""
        keys = self.ordered_keys()
        keys_strings = [str(x) for x in keys]
        max_key_length = len(max(keys_strings, key=len))

        table_rows = []

        for i, key in enumerate(keys):
            if len(str(key)) <= len(self.outcome_column):
                outcome_space = ' ' * (len(self.outcome_column) - len(str(key)))
            else:
                outcome_space = ' ' * (max_key_length - len(str(key)))
            table_rows.append(f"{key}{outcome_space} {self[key]}")

            if i >= 18:
                last_outcome = str(keys[-1])
                last_value = str(self[keys[-1]])
                table_rows.append(f"{'.' * len(last_outcome)}{outcome_space} "
                                  f"{'.' * len(last_value)}")
                table_rows.append(f"{last_outcome}{outcome_space} "
                                  f"{last_value}")
                break

        if max_key_length <= len(self.outcome_column):
            outcome_header_space = ' '
            total_row_space = ' ' * (len(self.outcome_column) - len('Total'))
        else:
            outcome_header_space = ' ' * (max_key_length -
                                          len(self.outcome_column) + 1)
            total_row_space = ' ' * (max_key_length - len('Total'))

        total = str(sum(self.values()))
        table_rows.append(f"{total_row_space}Total {total}")
        table_rows.insert(0, f"{self.outcome_column}{outcome_header_space}"
                             f"{self.value_column}")

        return '\n'.join(table_rows)

    def _repr_html_(self):
        """Return an HTML table representation for Jupyter notebooks."""
        keys = self.ordered_keys()

        # get HTML for table body
        table_body = ""
        for i, key in enumerate(keys):
            table_body += _get_row_html(key, self[key])
            # if we've already printed 19 rows, skip to end
            if i >= 18:
                table_body += _get_row_html("...", "...")
                table_body += _get_row_html(keys[-1], self[keys[-1]])
                break
        total = str(sum(self.values()))
        table_body += _get_row_html("<b>Total</b>", "<b>%s</b>" % total)

        # return HTML for entire table
        return TABLE_TEMPLATE.format(outcome_column = self.outcome_column,
                                     value_column = self.value_column,
                                     table_body=table_body)

    def _operation_factory(self, op):
        """Return a function that applies ``op`` element-wise to all values.

        Used by the ``Arithmetic`` superclass to define standard arithmetic
        operations (``+``, ``-``, ``*``, ``/``, ``**``, etc.) on a Table.

        Parameters
        ----------
        op : callable
            A binary operator, e.g. ``operator.add``.

        Returns
        -------
        callable
            A function ``_op_func(self, other)`` that returns a new
            ``Table`` with each value replaced by ``op(value, other)``.
        """

        def _op_func(self, other):
            return Table(
                {outcome: op(count, other) for outcome, count in self.items()},
                self.outcomes
            )

        return _op_func
