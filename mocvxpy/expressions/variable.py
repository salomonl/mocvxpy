import cvxpy as cp
import numpy as np

from typing import Any, Iterable, Optional


class Variable(cp.Variable):
    """The optimization variables in a problem.

    Overloads the cvxpy.Variable class.
    """

    def __init__(
        self,
        shape: int | Iterable[int] = (),
        name: str | None = None,
        var_id: int | None = None,
        **kwargs: Any,
    ):
        self._values = None
        super(Variable, self).__init__(shape, name, var_id, **kwargs)

    @property
    def values(self) -> Optional[np.ndarray]:
        """Returns: the numeric values of the variable.

        Each value corresponds to an optimal value of a multiobjective problem.
        """
        return self._values

    @values.setter
    def values(self, vals: np.ndarray) -> None:
        """Setter method for variable values.

        Arguments
        ---------
        vals: np.ndarray
            The optimal values corresponding to the variable.
        """
        self._values = vals
