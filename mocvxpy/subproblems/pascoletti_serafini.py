import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.subproblem import Subproblem
from mocvxpy.expressions.order_cone import OrderCone
from typing import List, Optional, Union


class PascolettiSerafiniSubproblem(Subproblem):
    """Solve the Pascoletti-Serafini subproblem.

    Solve: min z
           s.t. Z (f(x) -vref - z * dir) <= 0
           x in Omega
    where Z defines the ordering cone
    C = {y : Z y >= 0}
    of the multiobjective optimization problem.

    and its dual.

    Arguments
    ---------
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.
    order_cone: Optional[OrderCone]
        The order cone of the problem
    """

    def __init__(
        self,
        objectives: List[Union[cp.Minimize, cp.Maximize]],
        constraints: Optional[List[cp.Constraint]] = None,
        order_cone: Optional[OrderCone] = None,
    ) -> None:
        super().__init__(objectives, constraints, order_cone)

    def create_subproblem(self) -> cp.Problem:
        nobj = len(self._objectives)

        # Create problem parameters
        # 1- The outer vertex target.
        self._vref = cp.Parameter(nobj)
        # 2- The direction to reach the outer vertex target
        self._dir = cp.Parameter(nobj)

        z = cp.Variable()

        # Add constraints: Z(f(x) - vref - z * dir) <= 0
        Z = None if self._order_cone is None else self._order_cone.inequalities
        if Z is None:
            # Use the Pareto dominance cone
            for obj, objective in enumerate(self._objectives):
                self._constraints.append(
                    objective.expr <= self._vref[obj] + z * self._dir[obj]
                )
        else:
            for zrow in Z:
                self._constraints.append(
                    sum(
                        zrow[obj]
                        * (objective.expr - self._vref[obj] - z * self._dir[obj])
                        for obj, objective in enumerate(self._objectives)
                    )
                    <= 0
                )

        return cp.Problem(cp.Minimize(z), self._constraints)

    @property
    def parameters(self) -> np.ndarray:
        """Accessor method for parameters values.

        Returns
        -------
        np.ndarray
            The first nobj values correspond to the values of the outer vertex
            and the last nobj values correspond to the values of the vector direction.
        """
        return np.concatenate((self._vref.value, self._dir.value))

    @parameters.setter
    def parameters(self, param_values) -> None:
        """Setter method for parameters values.

        Arguments
        -------
        param_values: np.ndarray
            The parameters values.
            The first nobj values correspond to the values of the outer vertex
            and the last nobj values correspond to the values of the vector direction.
        """
        nobj = len(self._objectives)
        self._vref.value = param_values[:nobj]
        self._dir.value = param_values[nobj:]

    def dual_objective_values(self) -> np.ndarray:
        """Returns the dual values associated to the ``objective constraints''
           of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        # Collect dual values associated to the rays of the order cone
        nobj = len(self._objectives)
        Z = None if self._order_cone is None else self._order_cone.inequalities

        dual_obj_constraints_vals = (
            np.zeros(nobj) if Z is None else np.zeros(Z.shape[0])
        )
        if Z is None:
            for obj in range(nobj):
                dual_obj_constraints_vals[obj] = self._constraints[
                    -nobj + obj
                ].dual_value
        else:
            for ind in range(Z.shape[0]):
                dual_obj_constraints_vals[ind] = self._constraints[
                    -Z.shape[0] + ind
                ].dual_value

        return dual_obj_constraints_vals
