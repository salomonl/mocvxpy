import abc
import cvxpy as cp
import numpy as np

from copy import copy, deepcopy
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.problems.utilities import (
    extract_variables_from_problem,
)
from typing import List, Optional, Union


class Subproblem(metaclass=abc.ABCMeta):
    """An abstract base class for single-objective subproblems.

    The multiobjective optimization methods involve the optimization
    of multiple parameterized single-objective subproblems. The Subproblem class aims:
    1- to allocate once the optimization structures used to solve similar subproblems;
    2- isolate different instances of subproblems such that they can be solved in parallel.

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
    ):
        nobj = len(objectives)
        if nobj <= 1:
            raise ValueError("The number of objectives must be superior to 1", nobj)
        self._objectives = copy(objectives)

        if constraints is None:
            self._constraints = []
        else:
            self._constraints = copy(constraints)

        self._order_cone = order_cone

        self._vars = extract_variables_from_problem(self._objectives, self._constraints)

        self._pb = self.create_subproblem()

    @abc.abstractmethod
    def create_subproblem(self) -> cp.Problem:
        """Create subproblem.

        Must be implemented by all subclasses of Subproblem.

        Returns
        -------
        the created subproblem
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def parameters(self) -> np.ndarray:
        """Accessor method for parameters values.

        Returns
        -------
        np.ndarray
           A vector of parameter values.
        """
        raise NotImplementedError()

    @parameters.setter
    @abc.abstractmethod
    def parameters(self, param_values: np.ndarray) -> None:
        """Setter method for parameters values.

        Arguments
        -------
        param_values: np.ndarray
           The parameters values.
        """
        raise NotImplementedError()

    def solution(self) -> np.ndarray:
        """Returns the optimal decision values that correspond to the
        original multiobjective problem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the optimization has worked.
        Otherwise, the values are likely to be wrong.
        """
        opt_values = []
        for var in self._vars:
            opt_values += [val for val in var.value]
        return np.asarray(opt_values)

    def objective_values(self) -> np.ndarray:
        """Returns the optimal objective values that correspond to the
           original multiobjective problem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        return np.array([objective.expr.value for objective in self._objectives])

    @abc.abstractmethod
    def dual_objective_values(self) -> np.ndarray:
        """Returns the dual values associated to the ``objective constraints''
        of the subproblem.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Returns the dual values associated to the constraints of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        raise NotImplementedError()

    def value(self) -> float:
        """Returns the optimal objective value of the subproblem.

        Warning! It is the responsability of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the value is likely to be wrong.
        """
        return self._pb.value

    def solve(self) -> str:
        """Solve the problem.

        Returns
        -------
        str
           The status of the resolution
        """
        try:
            self._pb.solve(solver=cp.MOSEK)
        except:
            return "unsolved"

        if self._pb.status not in ["infeasible", "unbounded"]:
            return "solved"

        return "unsolved"
