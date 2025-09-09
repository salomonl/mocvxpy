import cvxpy as cp
import numpy as np
import time

from mocvxpy.constants import (
    MIN_DIST_OBJ_VECS,
    MONMO_MAX_ITER,
    MONMO_MAX_PB_SOLVED,
    MONMO_MIN_STOPPING_TOL,
)
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.problems.utilities import number_of_constraints, number_of_variables
from mocvxpy.solvers.common import (
    compute_extreme_objective_vectors,
    compute_extreme_points_hyperplane,
)
from mocvxpy.solvers.solution import OuterApproximation, Solution
from mocvxpy.subproblems.norm_min import NormMinSubproblem
from mocvxpy.subproblems.weighted_sum import WeightedSumSubproblem
from typing import Dict, List, Optional, Tuple, Union


class MONMOSolver:
    """The MONMO (MultiObjective Optimization by Norm Minimization Optimization) solver.

    Described in
    Ararat, Ç., Tekgül, S., & Ulus, F. (2023).
    "Geometric duality results and approximation algorithms for convex vector optimization problems".
    SIAM Journal on Optimization, 33(1), p. 116-146.
    https://doi.org/10.1137/21M1458788

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.
    constraints : list
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
        nobj = len(objectives)
        if nobj <= 1:
            raise ValueError("The number of objectives must be superior to 1", nobj)
        self._objectives = objectives

        if constraints is None:
            self._constraints = []
        else:
            self._constraints = constraints

        self._order_cone = order_cone

    def solve(
        self,
        verbose: bool = True,
        stopping_tol: float = 5 * 1e-4,
        max_iter: int = 100,
        max_pb_solved: int = 350,
        scalarization_solver_options: Optional[Dict] = None,
    ) -> Tuple[str, Solution]:
        """Solve the problem.

        Arguments
        ---------
        verbose: bool
           Display the output of the algorithm.

        stopping_tol: float
           The stopping tolerance. When the Hausdorff distance between the outer
           approximation and the inner approximation is below stopping_tol * scale_factor,
           the algorithm stops. Is always above or equal to MONMO_MIN_STOPPING_TOL = 1e-6.

        max_iter: int
           The maximum number of iterations allowed.

        max_pb_solved: int
           The maximum number of problems to be solved allowed. Do not account for the initial
           problems.

        scalarization_solver_options: optional[dict]
           The options of the solver used to solve the scalarization subproblem.
           Must be given under a dict whose keys are a pair of (str, any). Each key must follow
           the conventional way of giving options to a solver in cvxpy.

        Returns
        -------
        Tuple[str, mocvxpy.Solution]
           The status of the resolution and a corresponding solution set.
        """
        nobj = len(self._objectives)
        nvars = number_of_variables(self._objectives, self._constraints)
        sol = Solution(nvars, nobj)
        Z = None if self._order_cone is None else self._order_cone.inequalities

        if verbose:
            print("MONMO algorithm:")
            print("Number of objectives:", nobj)
            print("Number of variables:", nvars)
            print("Number of constraint expressions:", len(self._constraints))
            print("Number of constraints:", number_of_constraints(self._constraints))
            print(
                "Dominance cone:",
                (
                    "Pareto dominance cone"
                    if Z is None
                    else f"composed of {Z.shape[0]} rows"
                ),
            )

        initial_step_status, sol = self._initial_step(sol, scalarization_solver_options)
        if initial_step_status != "solved":
            if verbose:
                print(
                    "MONMO initialization failure: the algorithm cannot obtain extreme solutions;",
                    "initialisation phase status returns:",
                    initial_step_status,
                )
            return "no_extreme_solutions", sol

        # Initialize outer approximation defined by
        # TODO: take into consideration min and max objectives
        outer_approximation = OuterApproximation(nobj)
        if Z is None:
            # The ordering cone is the nobj dimensional positive orthant
            # Outer_init =  U {y[ob] >= f[obj](x*)}
            for obj, fvalues in enumerate(sol.objective_values):
                outer_approximation.insert_halfspace(
                    np.asarray(
                        [-fvalues[obj]]
                        + [1.0 if i == obj else 0.0 for i in range(nobj)]
                    )
                )
        else:
            # Outer_init = U {z^T y[ob] >= z^T f[obj](x*)}
            # where z are the rows from the matrix Z that define the order cone.
            for weights, fvalues in zip(Z, sol.objective_values):
                outer_approximation.insert_halfspace(
                    np.asarray([np.dot(-fvalues, weights)] + weights.tolist())
                )

        outer_vertices: np.ndarray = outer_approximation.compute_vertices()

        # Compute scaled stopping tolerance
        # 1- Approximately compute a hyperplane passing by all extreme points
        extreme_pts_hyp_params = compute_extreme_points_hyperplane(
            sol.extreme_objective_vectors()
        )
        if extreme_pts_hyp_params is None:
            scaled_stopping_tol = max(stopping_tol, MONMO_MIN_STOPPING_TOL)
        else:
            # 2- Compute distance from the ideal point to the hyperplane
            #    Given a hyperplane a^T x - 1 = 0 and a point x0, the distance is
            #    given by: d = |a^T x0 - 1| / ||a||
            ideal_to_extreme_pts_hyp_dist = np.absolute(
                np.dot(extreme_pts_hyp_params, sol.ideal_objective_vector()) - 1.0
            ) / np.linalg.norm(extreme_pts_hyp_params)
            scaled_stopping_tol = max(stopping_tol, MONMO_MIN_STOPPING_TOL) * max(
                1.0, ideal_to_extreme_pts_hyp_dist
            )

        # Set other stopping criteria
        max_iter = min(max_iter, MONMO_MAX_ITER)
        max_pb_solved = min(max_pb_solved, MONMO_MAX_PB_SOLVED)

        if verbose:
            print(f"Stopping tolerance: {scaled_stopping_tol:.5E}")

        hausdorff_dist = np.inf
        if verbose:
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|vert(Ok)|":>10} {"|halfspaces(Ok)|":>16} ",
                f"{"nb_pbs_solved_per_iter":>22} {"|unknown_vertices|":>18} {"Hausdorff_dist":>14} ",
                f"{"time_NM_pb_solve (s)":>20} {"avg_time_NM_pb_solve (s)":>24}",
            )
            print()
            print(
                f"{0:5d} {len(sol.objective_values):10d} ",
                f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                f"{"-":>20} {"-":>19} {hausdorff_dist:21e} ",
                f"{"-":>12} {"-":>20}",
            )

        explored_outer_vertices_information = []
        optimal_outer_vertices = []

        # Initialize subproblem
        norm_min_pb = NormMinSubproblem(
            self._objectives, self._constraints, self._order_cone
        )

        status = "max_iter_reached"
        start_optimization = time.perf_counter()
        elapsed_total_subproblems = 0.0
        total_nm_pbs_solved = 0
        total_nm_pbs_failed = 0
        for iter in range(max_iter):
            # Remove vertices that do not belong to the set of outer vertices
            for index in reversed(range(len(explored_outer_vertices_information))):
                remove_vertex = True
                vertex = explored_outer_vertices_information[index][0]
                for outer_vertex in outer_vertices:
                    if np.linalg.norm(vertex - outer_vertex) <= MIN_DIST_OBJ_VECS:
                        remove_vertex = False
                        break
                if remove_vertex:
                    del explored_outer_vertices_information[index]

            # Compute the set of unknown outer vertices
            unknown_outer_vertices = []
            for vertex in outer_vertices:
                is_an_optimal_vertex = False
                for voptimal in optimal_outer_vertices:
                    if np.linalg.norm(vertex - voptimal) <= MIN_DIST_OBJ_VECS:
                        is_an_optimal_vertex = True
                        break
                if is_an_optimal_vertex:
                    continue

                # Check if this vertex has not already been explored
                explore_vertex = False
                for vexplored, _, _, _ in explored_outer_vertices_information:
                    if np.linalg.norm(vertex - vexplored) <= MIN_DIST_OBJ_VECS:
                        explore_vertex = True
                        break
                if not explore_vertex:
                    unknown_outer_vertices.append(vertex)

            hausdorff_dist = -np.inf
            nb_subproblems_solved_per_iter = 0
            nb_subproblems_failed_per_iter = 0
            elapsed_subproblems_per_iter = 0.0
            for vertex in unknown_outer_vertices:
                v = np.asarray(vertex)
                norm_min_pb.parameters = v
                start_nm_pb_solved = time.perf_counter()
                if scalarization_solver_options is None:
                    status_norm_min = norm_min_pb.solve()
                else:
                    status_norm_min = norm_min_pb.solve(**scalarization_solver_options)
                end_nm_pb_solved = time.perf_counter()

                # If there is a failure, we ignore it and try to compute
                # the norm min subproblem for another vertex
                if status_norm_min != "solved":
                    optimal_outer_vertices.append(v)
                    nb_subproblems_failed_per_iter += 1
                    total_nm_pbs_failed += 1
                    continue

                elapsed_subproblems_per_iter += end_nm_pb_solved - start_nm_pb_solved

                nb_subproblems_solved_per_iter += 1
                total_nm_pbs_solved += 1

                # Update solution
                opt_obj_values = norm_min_pb.objective_values()
                w_opt = norm_min_pb.dual_objective_values()
                sol.insert_solution(
                    norm_min_pb.solution(),
                    opt_obj_values,
                    w_opt if Z is None else Z.T @ w_opt,
                    norm_min_pb.dual_constraint_values(),
                )

                opt_val = norm_min_pb.value()
                if opt_val <= scaled_stopping_tol:
                    # Update set of optimal outer vertices
                    optimal_outer_vertices.append(v)
                    hausdorff_dist = max(hausdorff_dist, opt_val)
                else:
                    # Update information of explored vertices
                    explored_outer_vertices_information.append(
                        (v, opt_obj_values, w_opt, opt_val)
                    )

                if total_nm_pbs_solved >= max_pb_solved:
                    break

            # Find v in argmax{||z^v|| | ||z^v|| > tol, v in outer_vertices}
            best_opt_val = -np.inf
            opt_index = -1
            for index, (_, _, _, opt_val) in enumerate(
                explored_outer_vertices_information
            ):
                hausdorff_dist = max(hausdorff_dist, opt_val)
                if opt_val > scaled_stopping_tol and opt_val > best_opt_val:
                    best_opt_val = opt_val
                    opt_index = index

            if opt_index >= 0:
                # Update outer approximation
                opt_obj_values = explored_outer_vertices_information[opt_index][1]
                w_opt = explored_outer_vertices_information[opt_index][2]
                if Z is None:
                    outer_approximation.insert_halfspace(
                        np.asarray([-np.dot(opt_obj_values, w_opt)] + w_opt.tolist())
                    )
                else:
                    outer_approximation.insert_halfspace(
                        np.asarray(
                            [-np.dot(opt_obj_values, Z.T @ w_opt)]
                            + (Z.T @ w_opt).tolist()
                        )
                    )
                # Move the corresponding vertex into the set of optimal vertices
                v = explored_outer_vertices_information[opt_index][0]
                optimal_outer_vertices.append(v)

                # Remove it from the set of explored vertices
                del explored_outer_vertices_information[opt_index]

            elapsed_total_subproblems += elapsed_subproblems_per_iter
            outer_vertices: np.ndarray = outer_approximation.compute_vertices()
            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.xvalues):10d} ",
                    f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                    f"{nb_subproblems_solved_per_iter:20d} {len(unknown_outer_vertices):19d} {hausdorff_dist:21e}",
                    f"{elapsed_subproblems_per_iter:19e} {elapsed_total_subproblems / total_nm_pbs_solved:19e}",
                )

            # NB: when there does not exist outer vertices (i.e., unknown_outer_vertices = []) to explore during this
            # iteration, the algorithm can still try to improve the outer approximation by adding new hyperplanes for
            # vertices already explored during some previous iterations.
            if unknown_outer_vertices and nb_subproblems_failed_per_iter == len(
                unknown_outer_vertices
            ):
                status = "norm_min_subproblem_failure"
                break

            if total_nm_pbs_solved >= max_pb_solved:
                status = "max_pbs_solved_reached"
                break

            if hausdorff_dist <= scaled_stopping_tol:
                status = "solved"
                break

        end_optimization = time.perf_counter()
        if verbose:
            print("\nStopping reason:", status)
            print("Resolution time (s):", end_optimization - start_optimization)
            print("Number of solutions found:", len(sol.xvalues))
            print(
                "Number of outer approximation's halfspaces found:",
                len(outer_approximation.halfspaces),
            )
            print(
                "Number of outer approximation's vertices found:", len(outer_vertices)
            )
            print("Number of nm problems solved:", total_nm_pbs_solved)
            print("Number of nm problems failed:", total_nm_pbs_failed)
            print(
                "Average time on nm problems solved:",
                elapsed_total_subproblems / total_nm_pbs_solved,
            )
            print()

        return status, sol

    def _initial_step(
        self, sol: Solution, scalarization_solver_options: Optional[Dict]
    ) -> Tuple[str, Solution]:
        """The initial step.

        Compute extreme solutions and their objective values according to the ordering cone.

        Arguments
        ---------
        sol: Solution
            The solution (initialized).

        scalarization_solver_options: optional[dict]
           The options of the solver used to solve the extreme solution subproblems.
           Must be given under a dict whose keys are a pair of (str, any). Each key must follow
           the conventional way of giving options to a solver in cvxpy.

        Returns
        -------
        str:
            The status of the initial step.

        Solution:
            The set of all solutions found during the initial step.
        """
        # Compute extreme solutions.
        if self._order_cone is None:
            init_phase_result = compute_extreme_objective_vectors(
                self._objectives,
                self._constraints,
                solver_options=scalarization_solver_options,
            )
            for ind, opt_values in enumerate(
                zip(init_phase_result[1], init_phase_result[2], init_phase_result[3])
            ):
                if init_phase_result[4] is None:
                    sol.insert_solution(opt_values[0], opt_values[1], opt_values[2])
                else:
                    sol.insert_solution(
                        opt_values[0],
                        opt_values[1],
                        opt_values[2],
                        init_phase_result[4][ind],
                    )

            return init_phase_result[0], sol

        status = "solved"
        weighted_sum_pb = WeightedSumSubproblem(self._objectives, self._constraints)
        for weights in self._order_cone.inequalities:
            weighted_sum_pb.parameters = weights
            if scalarization_solver_options is None:
                weighted_sum_status = weighted_sum_pb.solve()
            else:
                weighted_sum_status = weighted_sum_pb.solve(
                    **scalarization_solver_options
                )
            if weighted_sum_status == "solved":
                sol.insert_solution(
                    weighted_sum_pb.solution(),
                    weighted_sum_pb.objective_values(),
                    weights,
                    weighted_sum_pb.dual_constraint_values(),
                )
                continue

            if weighted_sum_status == "infeasible":
                status = "infeasible"
                break

            # The solver has not found an optimal solution.
            # We continue to try to compute as many initial solutions as possible
            status = "no_extreme_solutions"

        return status, sol
