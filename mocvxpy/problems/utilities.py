import cvxpy as cp
import numpy as np

from cvxpy.utilities.deterministic import unique_list
from typing import List, Union


def extract_variables_from_problem(
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: List[cp.Constraint],
) -> List[cp.Variable]:
    """
    Extract variables from problem defined by some objectives
    and constraints.

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.

    constraints : list
        The constraints on the problem variables (can be empty).

    Returns
    -------
    List[cp.Variable]
        The list of variables of the problem.
    """
    vars_ = []
    for objective in objectives:
        vars_ += objective.variables()
    for constr in constraints:
        vars_ += constr.variables()
    vars_ = unique_list(vars_)

    return vars_


def compute_objective_values(
    objectives: List[Union[cp.Minimize, cp.Maximize]],
) -> np.ndarray:
    """Compute objective values of a problem.

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.

    Important! The associated variables must have some values.

    Returns
    -------
    np.ndarray
        The objective values at a given point
    """
    return np.array([objective.value for objective in objectives])


def number_of_variables(
    objectives: List[Union[cp.Minimize, cp.Maximize]], constraints: List[cp.Constraint]
) -> int:
    """Returns the number of variables of a problem.

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.

    constraints : list
        The constraints on the problem variables (can be empty).

    Returns
    -------
    int
        The number of variables of a multiobjective problem
    """
    vars_ = extract_variables_from_problem(objectives, constraints)
    return sum(var.size for var in vars_)


def number_of_constraints(constraints: List[cp.Constraint]) -> int:
    """Returns the number of constraints of a set of constraints.

    NB: The number of constraints returned by this function takes
    into account the dimensions of the problem.

    Arguments
    ---------
    constraints : list
        The constraints on the problem variables (can be empty).

    Returns
    -------
    int
        The number of constraints of a set of constraints.
    """
    if len(constraints) == 0:
        return 0

    return sum(constraint.size for constraint in constraints)
