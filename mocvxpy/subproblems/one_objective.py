import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.subproblem import Subproblem
from typing import List, Optional, Tuple, Union


class OneObjectiveSubproblem(Subproblem):
    """Solve the one objective subproblem.

    Solve min sum fi(x)
          x in Omega

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
        self._obj_ind = -1
        # TODO: take into consideration min and max
        sum_objective = cp.Minimize(
            sum(objective.expr for (obj, objective) in enumerate(self._objectives))
        )

        return cp.Problem(sum_objective, self._constraints)

    @property
    def parameters(self) -> np.ndarray:
        """Accessor method for parameters values.

        Returns
        -------
        np.ndarray
            The objective index
        """
        return np.array([self._obj_ind])

    @parameters.setter
    def parameters(self, param_values) -> None:
        """Setter method for parameters values.

        Arguments
        -------
        obj_ind: int
            The objective index.
        """
        nobj = len(self._objectives)
        if param_values >= nobj:
            raise ValueError(
                f"The objective index {param_values} must be comprised between 0 and {nobj -1}"
            )
        self._obj_ind = param_values

        # Reload the problem
        self._pb = cp.Problem(self._objectives[param_values], self._constraints)

    def dual_objective_values(self) -> np.ndarray:
        """Returns the dual values associated to the ``objective constraints''
           of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        # There is no dual values in this case
        return np.array([])


def solve_one_objective_subproblem(
    obj: int,
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: Optional[List[cp.Constraint]] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Solve the one objective subproblem.

    Solve min sum fi(x)
          x in Omega

    NB: Create and solve the corresponding subproblem, which involves a cost.
    This function is used for parallelism, since it guarantees the independence of
    the created subproblems.

    Arguments
    ---------
    obj: int
        The index of the objective to optimize.
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray]
       The status of the optimization, the optimal solution values and the optimal objective values.
    """
    single_obj_pb = OneObjectiveSubproblem(objectives, constraints)
    single_obj_pb.parameters = obj
    single_obj_status = single_obj_pb.solve()
    if single_obj_status not in ["infeasible", "unbounded", "unsolved"]:
        return (
            single_obj_status,
            single_obj_pb.solution(),
            single_obj_pb.objective_values(),
        )
    else:
        return single_obj_status, np.ndarray([]), np.ndarray([])
