import abc
import cvxpy as cp
import numpy as np

from copy import deepcopy
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
        # When doing a deepcopy, we need to copy both objectives and constraints
        # together to keep relations between variables
        if constraints is None:
            self._objectives, self._constraints = deepcopy(objectives), []
        else:
            self._objectives, self._constraints = deepcopy((objectives, constraints))

        self._order_cone = order_cone

        self._vars = extract_variables_from_problem(self._objectives, self._constraints)

        self._pb = self.create_subproblem()
        self._backup_pb = None

    @abc.abstractmethod
    def create_subproblem(self) -> cp.Problem:
        """Create subproblem.

        Must be implemented by all subclasses of Subproblem.

        Returns
        -------
        the created subproblem
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def create_backup_subproblem(self) -> cp.Problem:
        """Create (backup) subproblem.

        Must be implemented by all subclasses of Subproblem.

        NB: A backup subproblem is not implemented as a DPP problem.
        It is always recomputed from scratch. The backup subproblem is
        used instead of the original problem when the corresponding
        DPP problem cannot be solved due to a DCP error. This method
        is always called AFTER create_subproblem().

        Returns
        -------
        the created subproblem
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def allow_backup_subproblem_optimization(self) -> bool:
        """Indicates if one can use the backup subproblem
        in case of DCP error.

        Returns
        -------
        True if the solve() method can use the backup subproblem
        if a DCP error occurs, false otherwise.
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

        Warning! It is the responsibility of the user to call the solve() method
        before calling this method and to check the optimization has worked.
        Otherwise, the values are likely to be wrong.
        """
        opt_values = []
        for var in self._vars:
            opt_values += [val for val in var.value.flatten()]
        return np.asarray(opt_values)

    def objective_values(self) -> np.ndarray:
        """Returns the optimal objective values that correspond to the
           original multiobjective problem.

        Warning! It is the responsibility of the user to call the solve() method
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

        Warning! It is the responsibility of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the values are likely to be wrong.
        """
        raise NotImplementedError()

    def value(self) -> float:
        """Returns the optimal objective value of the subproblem.

        Warning! It is the responsibility of the user to call the solve() method
        before calling this method and to check the resolution has worked. Otherwise,
        the value is likely to be wrong.
        """
        if self._backup_pb is not None:
            return self._backup_pb.value
        return self._pb.value

    def solve(
        self, solver: Optional[str] = None, verbose: bool = False, **kwargs
    ) -> str:
        """Solve the problem.

        Arguments
        ---------
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
        str
           The status of the resolution
        """
        # Always try to compute the original subproblem
        self._backup_pb = None
        try:
            self._pb.solve(solver=solver, verbose=verbose, **kwargs)
        except cp.DCPError:
            if not self.allow_backup_subproblem_optimization():
                return "unsolved"

            # Compute the backup problem and solve it
            self._backup_pb = self.create_backup_subproblem()
            try:
                self._backup_pb.solve(solver=solver, verbose=verbose, **kwargs)
            except:
                # Nothing can be done anymore
                return "unsolved"
        except:
            return "unsolved"

        pb_status = (
            self._pb.status if self._backup_pb is None else self._backup_pb.status
        )
        if pb_status not in ["infeasible", "unbounded"]:
            return "solved"

        return "unsolved"
