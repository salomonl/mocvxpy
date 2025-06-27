import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.subproblem import Subproblem
from mocvxpy.expressions.order_cone import OrderCone
from typing import List, Optional, Union


class WeightedSumSubproblem(Subproblem):
    """Solve the weighted subproblem.

    Solve min sum wi fi(x)
          x in Omega

    and its dual.

    Arguments
    ---------
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.
    """

    def __init__(
        self,
        objectives: List[Union[cp.Minimize, cp.Maximize]],
        constraints: Optional[List[cp.Constraint]] = None,
    ) -> None:
        super().__init__(objectives, constraints, None)

    def create_subproblem(self):
        nobj = len(self._objectives)

        self._weights = np.ones(nobj)

        # TODO: take into consideration min and max
        weighted_sum_objective = cp.Minimize(
            sum(
                self._weights[obj] * objective.expr
                for (obj, objective) in enumerate(self._objectives)
            )
        )

        return cp.Problem(weighted_sum_objective, self._constraints)

    @property
    def parameters(self) -> np.ndarray:
        """Accessor method for parameters values.

        Returns
        -------
        np.ndarray
            The values of the outer vertex
        """
        return self._weights

    @parameters.setter
    def parameters(self, param_values) -> None:
        """Setter method for parameters values.

        Arguments
        -------
        param_values: np.ndarray
            The weights
        """
        nobj = len(self._objectives)
        if np.size(self._weights) != nobj:
            raise ValueError(
                "The number of weights |W| = ",
                np.size(self._weights),
                "is not compatible with the number of objectives",
                nobj,
            )
        self._weights = param_values

        # Reload the problem
        weighted_sum_objective = cp.Minimize(
            sum(
                self._weights[obj] * objective.expr
                for (obj, objective) in enumerate(self._objectives)
            )
        )
        self._pb = cp.Problem(weighted_sum_objective, self._constraints)

    def dual_objective_values(self) -> np.ndarray:
        """Returns the dual values associated to the ``objective constraints''
           of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        # There is no dual values in this case
        return np.array([])
