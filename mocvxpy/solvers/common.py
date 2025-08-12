import cvxpy as cp
import numpy as np

from mocvxpy.subproblems.one_objective import OneObjectiveSubproblem
from typing import Dict, List, Optional, Tuple, Union


def compute_extreme_objective_vectors(
    objectives: List[Union[cp.Minimize, cp.Maximize]],
    constraints: Optional[List[cp.Constraint]],
    solver_options: Optional[Dict],
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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

    solver_options: optional, dict
        The options to pass to the solver computing the extreme points.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray, np.ndarray, optional[np.ndarray]
        The status of this step ("solved", "unbounded", "unfeasible"),
        the set of solutions associated, their respective objective values,
        their respective dual objective values and their respective dual constraint values,
        if they exist.
    """
    nobj = len(objectives)
    if nobj <= 1:
        raise ValueError("The number of objectives must be superior to 1", nobj)
    solutions = []
    objective_values = []
    dual_objective_values = []
    dual_constraint_values = []

    # NB: we do not filter solutions, even if they can be the same
    # since their dual objective values are nonetheless different
    single_obj_pb = OneObjectiveSubproblem(objectives, constraints)
    for obj in range(nobj):
        single_obj_pb.parameters = obj
        if solver_options is None:
            single_obj_status = single_obj_pb.solve()
        else:
            single_obj_status = single_obj_pb.solve(**solver_options)

        # Collect solution
        if single_obj_status not in ["infeasible", "unbounded"]:
            solutions.append(single_obj_pb.solution())
            objective_values.append(single_obj_pb.objective_values())
            dual_objective_values.append(
                np.array([1.0 if i == obj else 0.0 for i in range(nobj)])
            )
            dual_values = single_obj_pb.dual_constraint_values()
            if dual_values is not None:
                dual_constraint_values.append(dual_values)
            continue

        status = single_obj_status
        return (
            status,
            np.array([]) if len(solutions) == 0 else np.vstack(solutions),
            np.array([]) if len(objective_values) == 0 else np.vstack(objective_values),
            (
                np.array([])
                if len(dual_objective_values) == 0
                else np.vstack(dual_objective_values)
            ),
            (
                None
                if len(dual_constraint_values) == 0
                else np.vstack(dual_constraint_values)
            ),
        )

    status = "solved"
    return (
        status,
        np.vstack(solutions),
        np.vstack(objective_values),
        np.vstack(dual_objective_values),
        None if len(dual_constraint_values) == 0 else np.vstack(dual_constraint_values),
    )


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
        return None

    # We want to find the hyperplane passing by each extreme inner vertex z1, z2, ... zm
    # where m is the number of objectives of the problem. It must satisfy
    # a1 z1[1] + a2 z1[2] + ... + am z1[m] = 1
    # a1 z2[1] + a2 z2[2] + ... + am z2[m] = 1
    # ...
    # a1 zm[1] + a2 zm[2] + ... + am z[m] = 1
    # which is equivalent to:
    # Z * a = 1
    try:
        hyp_eq: np.ndarray = np.linalg.solve(extreme_pts, np.ones(nobj))
        return hyp_eq
    except np.linalg.LinAlgError:
        return None
