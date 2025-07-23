import cvxpy as cp
import dask
import numpy as np
import time

from dask.distributed import Client
from mocvxpy.constants import MIN_DIST_OBJ_VECS, MONMO_MAX_ITER, MONMO_MIN_STOPPING_TOL
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.problems.utilities import number_of_variables
from mocvxpy.solvers.common import compute_extreme_points_hyperplane
from mocvxpy.solvers.solution import OuterApproximation, Solution
from mocvxpy.subproblems.norm_min import solve_norm_min_subproblem
from mocvxpy.subproblems.one_objective import solve_one_objective_subproblem
from mocvxpy.subproblems.weighted_sum import solve_weighted_sum_subproblem
from typing import Dict, List, Optional, Tuple, Union


class MONMOParSolver:
    """The MONMO parallel (MultiObjective Optimization by Norm Minimization Optimization) solver.

    Adapted from
    Ararat, Ç., Tekgül, S., & Ulus, F. (2023).
    "Geometric duality results and approximation algorithms for convex vector optimization problems".
    SIAM Journal on Optimization, 33(1), p. 116-146.
    https://doi.org/10.1137/21M1458788

    Arguments
    ---------
    client: Client
        The dask client, that deals with distributing tasks.
    objectives : list[Minimize or Maximize]
        The problem's objectives.
    constraints : list
        The constraints on the problem variables.
    order_cone: Optional[OrderCone]
        The order cone of the problem
    """

    def __init__(
        self,
        client: Client,
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
        self._client = client

    def solve(
        self,
        verbose: bool = True,
        stopping_tol: float = 5 * 1e-4,
        max_iter: int = 10,
        max_pb_solved: int = 350,
        scalarization_solver_options: Optional[Dict] = None,
    ) -> Tuple[str, Solution]:
        """Solve the problem.

        Arguments
        ---------
        verbose: bool
           Display the output of the algorithm.

        stopping_tol: float
           The stopping tolerance. When the haussdorf distance between the outer
           approximation and the inner approximation is below stopping_tol * scale_factor,
           the algorithm stops. Is always above or equal to MONMO_MIN_STOPPING_TOL = 1e-6.

        max_iter: int
           The maximum number of iterations allowed.

        max_pb_solved: int
           The maximum number of problems to be solved allowed. Do not account for the initial
           problems.

        scalarization_solver_options: optional[dict]
           The options of the solver used to solve the scalarization subproblem.
           Must be given under a dict whose keys are pair of (str, any). Each key must follow
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
            print("MONMO Parallel algorithm:")
            print("Number of objectives:", nobj)
            print("Number of variables:", nvars)
            print(
                "Dominance cone:",
                (
                    "Pareto dominance cone"
                    if Z is None
                    else f"composed of {Z.shape[0]} rows"
                ),
            )
            print("Number of processes:", len(self._client.ncores()))
            print("Number of threads", sum(self._client.ncores().values()))

        initial_step_status, sol = self._initial_step(sol, scalarization_solver_options)
        if initial_step_status != "solved":
            if verbose:
                print(
                    "MONMO Parallel initialization failure: the algorithm cannot obtain extreme solutions;",
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
        extreme_pts_hyp_params: np.ndarray = compute_extreme_points_hyperplane(
            sol.extreme_objective_vectors()
        )
        # 2- Compute distance from the ideal point to the hyperplane
        #    Given a hyperplane a^T x - 1 = 0 and a point x0, the distance is
        #    given by: d = |a^T x0 - 1| / ||a||
        ideal_to_extreme_pts_hyp_dist = np.absolute(
            np.dot(extreme_pts_hyp_params, sol.ideal_objective_vector()) - 1.0
        ) / np.linalg.norm(extreme_pts_hyp_params)
        scaled_stopping_tol = max(stopping_tol, MONMO_MIN_STOPPING_TOL) * max(
            1.0, ideal_to_extreme_pts_hyp_dist
        )

        if verbose:
            print(f"Stopping tolerance: {scaled_stopping_tol:.5E}")

        haussdorf_dist = np.inf
        if verbose:
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|vert(Ok)|":>10} {"|halfspaces(Ok)|":>16} ",
                f"{"nb_pbs_solved_per_iter":>22} {"|unknown_vertices|":>18} {"Haussdorf_dist":>14} ",
                f"{"time_NM_pb_solve (s)":>20}",
            )
            print()
            print(
                f"{0:5d} {len(sol.objective_values):10d} ",
                f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                f"{"-":>20} {"-":>19} {haussdorf_dist:21e} ",
                f"{"-":>12}",
            )

        # Set options
        max_iter = min(max_iter, MONMO_MAX_ITER)

        explored_outer_vertices_information = []
        optimal_outer_vertices = []

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

            haussdorf_dist = -np.inf
            nb_subproblems_solved_per_iter = 0
            nb_subproblems_failed_per_iter = 0

            # Run problems in parallel
            # TODO: add a way to limit the maximum number of problems to solve
            max_pb_to_solve_per_iter = min(
                max_pb_solved - total_nm_pbs_solved, len(unknown_outer_vertices)
            )
            tasks = [
                dask.delayed(solve_norm_min_subproblem)(
                    np.asarray(v),
                    self._objectives,
                    self._constraints,
                    self._order_cone,
                    **(
                        scalarization_solver_options
                        if scalarization_solver_options is not None
                        else {}
                    ),
                )
                for v in unknown_outer_vertices[:max_pb_to_solve_per_iter]
            ]
            start_nm_pb_solved = time.perf_counter()
            run_tasks = self._client.compute(tasks)
            optimization_results = self._client.gather(run_tasks)
            end_nm_pb_solved = time.perf_counter()
            elapsed_subproblems_per_iter = end_nm_pb_solved - start_nm_pb_solved

            # Collect solutions
            for optimization_logs, vertex in zip(
                optimization_results, unknown_outer_vertices[:max_pb_to_solve_per_iter]
            ):
                v = np.asarray(vertex)
                status_norm_min = optimization_logs[0]

                # If there is a failure, we ignore it and try to compute
                # the norm min subproblem for another vertex
                if status_norm_min != "solved":
                    optimal_outer_vertices.append(v)
                    nb_subproblems_failed_per_iter += 1
                    total_nm_pbs_failed += 1
                    continue

                nb_subproblems_solved_per_iter += 1
                total_nm_pbs_solved += 1

                # Update solution
                opt_obj_values = optimization_logs[2]
                w_opt = optimization_logs[3]
                sol.insert_solution(
                    optimization_logs[1],
                    optimization_logs[2],
                    w_opt if Z is None else Z.T @ w_opt,
                    optimization_logs[5],
                )

                opt_val = optimization_logs[4]
                if opt_val <= scaled_stopping_tol:
                    # Update set of optimal outer vertices
                    optimal_outer_vertices.append(v)
                    haussdorf_dist = max(haussdorf_dist, opt_val)
                else:
                    # Update information of explored vertices
                    explored_outer_vertices_information.append(
                        (v, opt_obj_values, w_opt, opt_val)
                    )

                if total_nm_pbs_solved >= max_pb_solved:
                    break

            # Find v in arg{||z^v|| | ||z^v|| > tol, v in outer_vertices}
            selected_vertex_indexes = []
            for index, (_, _, _, opt_val) in enumerate(
                explored_outer_vertices_information
            ):
                haussdorf_dist = max(haussdorf_dist, opt_val)
                if opt_val > scaled_stopping_tol:
                    selected_vertex_indexes.append(index)

            for index in reversed(selected_vertex_indexes):
                opt_obj_values = explored_outer_vertices_information[index][1]
                w_opt = explored_outer_vertices_information[index][2]

                # Update outer approximation
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
                v = explored_outer_vertices_information[index][0]
                optimal_outer_vertices.append(v)

                # Remove it from the set of explored vertices
                del explored_outer_vertices_information[index]

            elapsed_total_subproblems += elapsed_subproblems_per_iter
            outer_vertices: np.ndarray = outer_approximation.compute_vertices()
            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.xvalues):10d} ",
                    f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                    f"{nb_subproblems_solved_per_iter:20d} {max_pb_to_solve_per_iter:19d} {haussdorf_dist:21e}",
                    f"{elapsed_subproblems_per_iter:19e}",
                )

            if nb_subproblems_failed_per_iter == len(unknown_outer_vertices):
                status = "norm_min_subproblem_failure"
                break

            if total_nm_pbs_solved >= max_pb_solved:
                status = "max_pbs_solved_reached"
                break

            if haussdorf_dist <= scaled_stopping_tol:
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
            print()

        return status, sol

    def _initial_step(
        self, sol: Solution, scalarization_solver_options: Optional[Dict]
    ) -> Tuple[str, Solution]:
        """The initial step.

        Compute in parallel extreme solutions and their objective values according to the ordering cone.

        Arguments
        ---------
        sol: Solution
            The solution (initialized).

        scalarization_solver_options: optional[dict]
            The options of the solver used to solve the extreme solution subproblems.
            Must be given under a dict whose keys are pair of (str, any). Each key must follow
            the conventional way of giving options to a solver in cvxpy.

        Returns
        -------
        str:
            The status of the initial step.

        Solution:
            The set of all solutions found during the initial step.
        """
        # Solve all problems in parallel
        nobj = len(self._objectives)
        if self._order_cone is None:
            tasks = [
                dask.delayed(solve_one_objective_subproblem)(
                    obj,
                    self._objectives,
                    self._constraints,
                    **(
                        scalarization_solver_options
                        if scalarization_solver_options is not None
                        else {}
                    ),
                )
                for obj in range(nobj)
            ]
        else:
            tasks = [
                dask.delayed(solve_weighted_sum_subproblem)(
                    weights,
                    self._objectives,
                    self._constraints,
                    **(
                        scalarization_solver_options
                        if scalarization_solver_options is not None
                        else {}
                    ),
                )
                for weights in self._order_cone.inequalities
            ]
        run_tasks = self._client.compute(tasks)
        optimization_results = self._client.gather(run_tasks)

        # Collect solutions
        status = "solved"
        for optimization_logs in optimization_results:
            single_obj_status = optimization_logs[0]

            if single_obj_status == "solved":
                sol.insert_solution(
                    optimization_logs[1],
                    optimization_logs[2],
                    optimization_logs[3],
                    optimization_logs[4],
                )
                continue

            if single_obj_status == "infeasible":
                status = "infeasible"
                break

            # The solver has not found an optimal solution.
            # We continue to try to compute as many initial solutions as possible
            status = "no_extreme_solutions"

        return status, sol
