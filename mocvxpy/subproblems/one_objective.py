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
            sum(objective.expr for objective in self._objectives)
        )

        return cp.Problem(sum_objective, self._constraints)

    def create_backup_subproblem(self) -> cp.Problem:
        return self.create_subproblem()

    def allow_backup_subproblem_optimization(self) -> bool:
        return False

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

        Warning! It is the responsibility of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        # There is no dual values in this case
        return np.array([])

    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Returns the dual values associated to the constraints of the subproblem.

        Warning! It is the responsibility of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        dual_constraint_values = []
        if not self._constraints:
            return None
        for constraint in self._constraints:
            if isinstance(constraint.dual_value, float) or isinstance(
                constraint.dual_value, complex
            ):
                dual_constraint_values += [constraint.dual_value]
            else:
                if constraint.dual_value.ndim == 0:
                    # In some circumstances, the dual_value method can return
                    # a 0-dimensional array. We need to deal with this case
                    dual_constraint_values += [constraint.dual_value.tolist()]
                else:
                    dual_constraint_values += constraint.dual_value.flatten().tolist()
        return np.asarray(dual_constraint_values)


def solve_one_objective_subproblem(
    obj: int,
    single_obj_pb: OneObjectiveSubproblem,
    solver: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solve the one objective subproblem.

    Solve min sum fi(x)
          x in Omega

    NB: Solve the corresponding subproblem. This function is specifically used
    for parallelism. It is the responsibility of the user to be sure that each
    subproblem instance is independent of each other.

    Arguments
    ---------
    obj: int
        The index of the objective to optimize.
    single_obj_pb: OneObjectiveSubproblem
        The subproblem instance to solve.
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
    single_obj_pb.parameters = obj
    single_obj_status = single_obj_pb.solve(solver=solver, verbose=verbose, **kwargs)
    if single_obj_status not in ["infeasible", "unbounded", "unsolved"]:
        return (
            single_obj_status,
            single_obj_pb.solution(),
            single_obj_pb.objective_values(),
            np.array(
                [
                    1.0 if i == obj else 0.0
                    for i in range(len(single_obj_pb._objectives))
                ]
            ),
            single_obj_pb.dual_constraint_values(),
        )
    else:
        return single_obj_status, np.array([]), np.array([]), np.array([]), None
