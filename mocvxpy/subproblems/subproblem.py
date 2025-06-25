import abc
import numpy as np


class Subproblem(metaclass=abc.ABCMeta):
    """An abstract base class for single-objective subproblems.

    The multiobjective optimization methods involve the optimization
    of multiple parameterized single-objective subproblems. The Subproblem class aims:
    1- to allocate once the optimization structures used to solve similar subproblems;
    2- isolate different instances of subproblems such that they can be solved in parallel.
    """

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

    @abc.abstractmethod
    def solution(self) -> np.ndarray:
        """Returns the optimal decision values that correspond to the
        original multiobjective problem.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def objective_values(self) -> np.ndarray:
        """Returns the optimal objective values that correspond to the
        original multiobjective problem.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def dual_objective_values(self) -> np.ndarray:
        """Returns the dual values associated to the ``objective constraints''
        of the subproblem.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def value(self) -> float:
        """Returns the optimal objective value of the subproblem."""
        raise NotImplementedError()

    @abc.abstractmethod
    def solve(self) -> str:
        """Solve the problem.

        Returns
        -------
        str
           The status of the resolution
        """
        raise NotImplementedError()
