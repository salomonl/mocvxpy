import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.subproblem import Subproblem
from typing import List, Optional, Tuple, Union


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

    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Returns the dual values associated to the constraints of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        dual_constraint_values = []
        if not self._constraints:
            return None
        for constraint in self._constraints:
            if isinstance(constraint.dual_value, float):
                dual_constraint_values += [constraint.dual_value]
            else:
                if constraint.dual_value.ndim == 0:
                    # In some circumstances, the dual_value method can return
                    # a 0-dimensional array. We need to deal with this case
                    dual_constraint_values += [constraint.dual_value.tolist()]
                else:
                    dual_constraint_values += constraint.dual_value.tolist()
        return np.asarray(dual_constraint_values)


def solve_weighted_sum_subproblem(
    weights: np.ndarray,
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: Optional[List[cp.Constraint]] = None,
    solver: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solve the weighted subproblem.

    Solve min sum wi fi(x)
          x in Omega

    and its dual.

    NB: Create and solve the corresponding subproblem, which involves a cost.
    This function is used for parallelism, since it guarantees the independence of
    the created subproblems.

    Arguments
    ---------
    weights: np.ndarray
        The weights.
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.
    solver: optional[str]
        The solver to use.
    solver_path: list of (str, dict) tuples or strings, optional
        The solvers to use with optional arguments. The method tries the solvers
        in the given order and returns the first solver's solution that succeeds.
    verbose: optional[bool]
        If True, displays the outputs of the solver.
    **kwargs
        Additional keyword arguments specifying solver specific options.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray, np.ndarray, optional[np.ndarray]]
        The status of the optimization, the optimal solution values, the optimal objective values,
        the dual objective values, and the dual constraint values (if they exist).
    """
    weighted_sum_pb = WeightedSumSubproblem(objectives, constraints)
    weighted_sum_pb.parameters = weights
    weighted_sum_status = weighted_sum_pb.solve(
        solver=solver, verbose=verbose, **kwargs
    )
    if weighted_sum_status not in ["infeasible", "unbounded", "unsolved"]:
        return (
            weighted_sum_status,
            weighted_sum_pb.solution(),
            weighted_sum_pb.objective_values(),
            weights,
            weighted_sum_pb.dual_constraint_values(),
        )
    else:
        return weighted_sum_status, np.array([]), np.array([]), np.array([]), None
