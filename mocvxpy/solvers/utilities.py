import cvxpy as cp
import numpy as np

from cvxpy.utilities.deterministic import unique_list
from mocvxpy.expressions.order_cone import OrderCone
from typing import List, Optional, Tuple, Union


def compute_extreme_objective_vectors(
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: Optional[List[cp.Constraint]],
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Compute extreme solutions of a multiobjective problem.

    Solve min fi(x) for i = 1, 2, ..., m
          s.t. x in X
    where:
    - m is the number of objectives.
    - X is the set of constraints

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.

    constraints : list
        The constraints on the problem variables.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray]
        The status of this step ("solved", "unbounded", "unfeasible"),
        the set of solutions associated and their respective objective values.
    """
    nobj = len(objectives)
    if nobj <= 1:
        raise ValueError("The number of objectives must be superior to 1", nobj)
    constraints_ = [] if constraints is None else constraints

    vars_ = extract_variables_from_problem(objectives, constraints_)

    nvars = sum(var.size for var in vars_)

    solutions = np.ndarray((0, nvars))
    objective_values = np.ndarray((0, nobj))

    status = "not_defined"
    # TODO: filter solutions in case
    for objective in objectives:
        single_obj_pb = cp.Problem(objective, constraints_)
        single_obj_pb.solve(solver=cp.MOSEK)

        # Collect solution
        if single_obj_pb.status not in ["infeasible", "unbounded"]:
            opt_values: List[float] = []
            for var in vars_:
                opt_values += var.value.tolist()
            solutions = np.vstack([solutions, opt_values])

            opt_fvalues = compute_objective_values(objectives)
            objective_values = np.vstack([objective_values, opt_fvalues])
            continue

        status = single_obj_pb.status
        return status, solutions, objective_values

    status = "solved"
    return status, solutions, objective_values

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

def compute_objective_values(objectives: List[Union[cp.Minimize, cp.Maximize]]) -> np.ndarray:
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
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: List[cp.Constraint]) -> int:
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

def compute_extreme_points_hyperplane(extreme_pts: np.ndarray) -> Optional[np.ndarray]:
    """Return the equation of the hyperplane passing by all extreme
    points in the objective space.

    Arguments
    ---------
    extreme_pts: np.ndarray

    Returns
    -------
    Optional[np.ndarray]
        The equation of the hyperplane passing by all extreme points.
    """
    nobj = extreme_pts.shape[1]
    nextreme_pts = extreme_pts.shape[0]
    if nextreme_pts != nobj:
        return

    # We want to find the hyperplane passing by each extreme inner vertex z1, z2, ... zm
    # where m is the number of objectives of the problem. It must satisfy
    # a1 z1[1] + a2 z1[2] + ... + am z1[m] = 1
    # a1 z2[1] + a2 z2[2] + ... + am z2[m] = 1
    # ...
    # a1 zm[1] + a2 zm[2] + ... + am z[m] = 1
    # which is equivalent to:
    # Z * a = 1
    hyp_eq: np.ndarray = np.linalg.solve(extreme_pts, np.ones(nobj))
    return hyp_eq
