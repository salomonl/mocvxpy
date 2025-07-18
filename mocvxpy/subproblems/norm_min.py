import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.subproblem import Subproblem
from mocvxpy.expressions.order_cone import OrderCone
from typing import List, Optional, Union, Tuple


class NormMinSubproblem(Subproblem):
    """Solve the norm minimization subproblem.

    Solve min || z ||
          s.t. Z f(x) <= Z (z + vref)
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

    def create_subproblem(self):
        nobj = len(self._objectives)

        # Create the outer vertex target.
        self._vref = cp.Parameter(nobj)

        z = cp.Variable(nobj)

        # Add constraints: Z f(x) <= Z (v + z)
        Z = None if self._order_cone is None else self._order_cone.inequalities
        if Z is None:
            # Use the Pareto dominance cone
            for obj, objective in enumerate(self._objectives):
                self._constraints.append(objective.expr <= self._vref[obj] + z[obj])
        else:
            for zrow in Z:
                self._constraints.append(
                    sum(
                        zrow[obj] * (objective.expr - self._vref[obj] - z[obj])
                        for obj, objective in enumerate(self._objectives)
                    )
                    <= 0
                )

        return cp.Problem(cp.Minimize(cp.norm(z)), self._constraints)

    @property
    def parameters(self) -> np.ndarray:
        """Accessor method for parameters values.

        Returns
        -------
        np.ndarray
            The values of the outer vertex
        """
        return self._vref.value

    @parameters.setter
    def parameters(self, param_values) -> None:
        """Setter method for parameters values.

        Arguments
        -------
        param_values: np.ndarray
            The values of the outer vertex.
        """
        nobj = len(self._objectives)
        self._vref.value = param_values[:nobj]

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

    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Returns the dual values associated to the constraints of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        Z = None if self._order_cone is None else self._order_cone.inequalities
        if Z is None:
            nobj = len(self._objectives)
            noriginal_cons = len(self._constraints) - nobj
        else:
            noriginal_cons = len(self._constraints) - Z.shape[0]
        if noriginal_cons == 0:
            return None

        dual_constraint_values = []
        for constraint in self._constraints[:noriginal_cons]:
            if isinstance(constraint.dual_value, float):
                dual_constraint_values += [constraint.dual_value]
            else:
                dual_constraint_values += constraint.dual_value.tolist()
        return np.asarray(dual_constraint_values)


def solve_norm_min_subproblem(
    outer_vertex: np.ndarray,
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: Optional[List[cp.Constraint]] = None,
    order_cone: Optional[OrderCone] = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """Solve the norm minimization subproblem.

    Solve min || z ||
          s.t. Z f(x) <= Z (z + vref)
          x in Omega
    where Z defines the ordering cone
    C = {y : Z y >= 0}
    of the multiobjective optimization problem.

    and its dual.

    NB: Create and solve the corresponding subproblem, which involves a cost.
    This function is used for parallelism, since it guarantees the independence of
    the created subproblems.

    Arguments
    ---------
    outer_vertex: np.ndarray
        The outer vertex (vref).
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.
    order_cone: Optional[OrderCone]
        The order cone of the problem.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray, np.ndarray, float, optional[np.ndarray]]
        The status of the optimization, the optimal solution values, the optimal objective values,
        the dual objective values, the optimal value of the norm min subproblem and the dual
        values of the constraints (if they exist).
    """
    norm_min_pb = NormMinSubproblem(objectives, constraints, order_cone)
    norm_min_pb.parameters = outer_vertex
    norm_min_status = norm_min_pb.solve()
    if norm_min_status not in ["infeasible", "unbounded", "unsolved"]:
        return (
            norm_min_status,
            norm_min_pb.solution(),
            norm_min_pb.objective_values(),
            norm_min_pb.dual_objective_values(),
            norm_min_pb.value(),
            norm_min_pb.dual_constraint_values(),
        )
    else:
        return (
            norm_min_status,
            np.ndarray([]),
            np.ndarray([]),
            np.ndarray([]),
            -1.0,
            np.ndarray([]),
        )
