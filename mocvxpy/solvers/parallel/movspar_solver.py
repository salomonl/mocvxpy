import cvxpy as cp
import dask
import numpy as np
import time

from dask.distributed import Client
from itertools import batched
from mocvxpy.constants import MIN_DIST_OBJ_VECS, MOVS_MAX_ITER, MOVS_MIN_STOPPING_TOL
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.problems.utilities import number_of_constraints, number_of_variables
from mocvxpy.solvers.common import compute_extreme_points_hyperplane
from mocvxpy.solvers.solution import OuterApproximation, Solution
from mocvxpy.subproblems.one_objective import (
    OneObjectiveSubproblem,
    solve_one_objective_subproblem,
)
from mocvxpy.subproblems.pascoletti_serafini import (
    PascolettiSerafiniSubproblem,
    solve_pascoletti_serafini_subproblem,
)
from mocvxpy.subproblems.weighted_sum import (
    WeightedSumSubproblem,
    solve_weighted_sum_subproblem,
)
from typing import Dict, List, Optional, Tuple, Union


class MOVSParSolver:
    """The MOVS parallel (MultiObjective Optimization by Vertex Selection) solver.

    Described in
    Dörfler, D., Löhne, A., Schneider, C., & Weißing, B. (2021).
    "A Benson-type algorithm for bounded convex vector optimization problems with vertex selection".
    Optimization Methods and Software, 37(3), p. 1006–1026.
    https://doi.org/10.1080/10556788.2021.1880579

    Arguments
    ---------
    client: Client
        The dask client, that deals with distributing tasks.
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
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
        max_iter: int = 7,
        scalarization_solver_options: Optional[Dict] = None,
        vertex_selection_solver_options: Optional[Dict] = None,
    ) -> Tuple[str, Solution]:
        """Solve the problem.

        Arguments
        ---------
        verbose: bool
           Display the output of the algorithm.

        stopping_tol: float
           The stopping tolerance. When the Hausdorff distance between the outer
           approximation and the inner approximation is below stopping_tol * scale_factor,
           the algorithm stops. Is always above or equal to MOVS_MIN_STOPPING_TOL = 1e-6.

        max_iter: int
           The maximum number of iterations allowed.

        scalarization_solver_options: optional[dict]
           The options of the solver used to solve the scalarization subproblem.
           Must be given under a dict whose keys are a pair of (str, any). Each key must follow
           the conventional way of giving options to a solver in cvxpy.

        vertex_selection_solver_options: optional[dict]
           The options of the solver used to solve the vertex selection subproblem. Must be able
           to solve a quadratic problem.
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

        ntotal_cores = sum(self._client.ncores().values())

        if verbose:
            print("MOVS Parallel algorithm:")
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
            print("Number of processes:", len(self._client.ncores()))
            print("Number of threads", ntotal_cores)

        initial_step_status, sol = self._initial_step(sol, scalarization_solver_options)
        if initial_step_status != "solved":
            if verbose:
                print(
                    "MOVS Parallel initialization failure: the algorithm cannot obtain extreme solutions;",
                    "initialisation phase status returns:",
                    initial_step_status,
                )
            return "no_extreme_solutions", sol

        # Initialize outer approximation
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
            scaled_stopping_tol = max(stopping_tol, MOVS_MIN_STOPPING_TOL)
        else:
            # 2- Compute distance from the ideal point to the hyperplane
            #    Given a hyperplane a^T x - 1 = 0 and a point x0, the distance is
            #    given by: d = |a^T x0 - 1| / ||a||
            ideal_to_extreme_pts_hyp_dist = np.absolute(
                np.dot(extreme_pts_hyp_params, sol.ideal_objective_vector()) - 1.0
            ) / np.linalg.norm(extreme_pts_hyp_params)
            scaled_stopping_tol = max(stopping_tol, MOVS_MIN_STOPPING_TOL) * max(
                1.0, ideal_to_extreme_pts_hyp_dist
            )

        if verbose:
            print(f"Stopping tolerance: {scaled_stopping_tol:.5E}")

        hausdorff_dist = np.inf
        if verbose:
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|vert(Ok)|":>10} {"|halfspaces(Ok)|":>16} ",
                f"{"|QP(s, Ik)|":>11} {"Hausdorff_dist":>14} ",
                f"{"nb_ps_solved":>12} {"nb_ps_failed":>12} ",
                f"{"time_QP_solve (s)":>17} {"time_PS_pb_solve (s)":>20}",
            )
            print()
            print(
                f"{0:5d} {len(sol.objective_values):10d} ",
                f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                f"{"-":>13}",
                f"{hausdorff_dist:14e}",
            )

        # Set options
        max_iter = min(MOVS_MAX_ITER, max_iter)

        # Allocate a set of subproblems that will be reused during the computation
        ps_subproblems_poll = [
            dask.delayed(PascolettiSerafiniSubproblem)(
                self._objectives, self._constraints, self._order_cone
            )
            for _ in range(ntotal_cores)
        ]
        ps_subproblems_poll = [
            self._client.persist(ps_pb_task) for ps_pb_task in ps_subproblems_poll
        ]

        status = "maxiter_reached"
        vertex_selection_solutions = np.array([]).reshape((0, 2 * nobj))
        visited_outer_vertices = np.array([]).reshape((0, nobj))
        current_inner_vertex_indexes = []
        nb_subproblems_failed_per_iter = 0
        nb_subproblems_solved_per_iter = 0
        elapsed_ps_pb = 0.0
        start_optimization = time.perf_counter()
        for iter in range(max_iter):
            start_vertex_selection_pb = time.perf_counter()
            vertex_selection_solutions, nb_qp_solved = self._solve_vertex_selection_pb(
                outer_vertices,
                sol.objective_values,
                vertex_selection_solutions,
                current_inner_vertex_indexes,
                vertex_selection_solver_options,
            )
            end_vertex_selection_pb = time.perf_counter()
            elapsed_vertex_selection_pb = (
                end_vertex_selection_pb - start_vertex_selection_pb
            )

            # If the optimization of all vertex selection subproblems fails,
            # stop the procedure
            if len(vertex_selection_solutions) == 0:
                status = "vertex_selection_failure"
                break

            hausdorff_dist = -np.inf
            for vertex_pair in vertex_selection_solutions:
                hausdorff_dist = max(
                    hausdorff_dist,
                    np.linalg.norm(vertex_pair[:nobj] - vertex_pair[nobj:]),
                )

            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.objective_values):10d} ",
                    f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                    f"{nb_qp_solved:14d} {hausdorff_dist:17e}",
                    f"{nb_subproblems_solved_per_iter:11d} {nb_subproblems_failed_per_iter:12d} ",
                    f"{elapsed_vertex_selection_pb:16e} {elapsed_ps_pb:16e}",
                )

            if hausdorff_dist <= scaled_stopping_tol:
                status = "solved"
                break

            # Select vertices that are above the stopping threshold and that have not been already visited
            vertex_selection_candidates = []
            for sp_pair in vertex_selection_solutions:
                is_visited = False
                for v in visited_outer_vertices:
                    if np.linalg.norm(sp_pair[:nobj] - v) <= MIN_DIST_OBJ_VECS:
                        is_visited = True
                        break
                if (
                    not is_visited
                    and np.linalg.norm(sp_pair[:nobj] - sp_pair[nobj:])
                    > scaled_stopping_tol
                ):
                    vertex_selection_candidates.append(sp_pair)

            if not vertex_selection_candidates:
                status = "vertex_selection_failure"
                break

            # Solve problems in parallel per batch to prevent potential memory
            # issues due to the allocation of large problems
            optimization_results = []
            for vertex_pair_batch in batched(vertex_selection_candidates, ntotal_cores):
                nvertices_pair = len(vertex_pair_batch)
                tasks = []
                for sp_pair, ps_pb in zip(
                    vertex_pair_batch, ps_subproblems_poll[:nvertices_pair]
                ):
                    tasks.append(
                        dask.delayed(solve_pascoletti_serafini_subproblem)(
                            sp_pair[:nobj],
                            sp_pair[nobj:] - sp_pair[:nobj],
                            ps_pb,
                            **(
                                scalarization_solver_options
                                if scalarization_solver_options is not None
                                else {}
                            ),
                        )
                    )

                # Solve problems
                start_ps_pb = time.perf_counter()
                run_tasks = self._client.compute(tasks)
                optimization_batch_results = self._client.gather(run_tasks)
                end_ps_pb = time.perf_counter()
                elapsed_ps_pb = end_ps_pb - start_ps_pb
                optimization_results += optimization_batch_results
                del tasks  # Try to decrease the memory allocated for the problems

            # Collect solutions
            nb_subproblems_failed_per_iter = 0
            nb_subproblems_solved_per_iter = 0
            current_inner_vertex_indexes = []
            for optimization_logs, sp_pair in zip(
                optimization_results, vertex_selection_candidates
            ):
                status_ps = optimization_logs[0]

                # If there is a failure, we ignore it and move to the next one
                if status_ps != "solved":
                    nb_subproblems_failed_per_iter += 1
                    continue

                nb_subproblems_solved_per_iter += 1

                # Update solution
                w_opt = optimization_logs[3]
                sol.insert_solution(
                    optimization_logs[1],
                    optimization_logs[2],
                    w_opt if Z is None else Z.T @ w_opt,
                    optimization_logs[5],
                )
                current_inner_vertex_indexes.append(len(sol.objective_values) - 1)

                outer_v = sp_pair[:nobj]
                visited_outer_vertices = np.vstack([visited_outer_vertices, outer_v])

                # Update outer approximation
                z_opt = optimization_logs[4]
                if Z is None:
                    outer_approximation.insert_halfspace(
                        np.asarray([-z_opt - np.dot(outer_v, w_opt)] + w_opt.tolist())
                    )
                else:
                    outer_approximation.insert_halfspace(
                        np.asarray(
                            [-z_opt - np.dot(outer_v, Z.T @ w_opt)]
                            + (Z.T @ w_opt).tolist()
                        )
                    )

            if nb_subproblems_failed_per_iter == len(optimization_results):
                status = "ps_subproblem_failure"
                if verbose:
                    print(
                        f"{iter+2:5d} {len(sol.objective_values):10d} ",
                        f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                        f"{nb_qp_solved:14d} {hausdorff_dist:17e}",
                        f"{nb_subproblems_solved_per_iter:11d} {nb_subproblems_failed_per_iter:12d} ",
                        f"{elapsed_vertex_selection_pb:16e} {elapsed_ps_pb:16e}",
                    )
                break

            outer_vertices = outer_approximation.compute_vertices()

        end_optimization = time.perf_counter()
        if verbose:
            print("\nStopping reason:", status)
            print("Resolution time (s):", end_optimization - start_optimization)
            print(
                "Hausdorff distance between outer and inner approximation of the solution set:",
                f"{hausdorff_dist:.5E}",
            )
            print("Number of solutions found:", len(sol.xvalues))
            print(
                "Number of outer approximation's halfspaces found:",
                len(outer_approximation.halfspaces),
            )
            print(
                "Number of outer approximation's vertices found:", len(outer_vertices)
            )
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
            Must be given under a dict whose keys are a pair of (str, any). Each key must follow
            the conventional way of giving options to a solver in cvxpy.

        Returns
        -------
        str:
            The status of the initial step.

        Solution:
            The set of all solutions found during the initial step.
        """
        nobj = len(self._objectives)
        # Create separate subproblem instances; bypass cvxpy threading
        # issues that may result in some subproblems being unsolved
        # in sequential mode
        if self._order_cone is None:
            initial_subproblems_poll = [
                dask.delayed(OneObjectiveSubproblem)(
                    self._objectives, self._constraints
                )
                for _ in range(nobj)
            ]
        else:
            initial_subproblems_poll = [
                dask.delayed(WeightedSumSubproblem)(self._objectives, self._constraints)
                for _ in self._order_cone.inequalities
            ]

        # Solve all problems in parallel
        if self._order_cone is None:
            tasks = [
                dask.delayed(solve_one_objective_subproblem)(
                    obj,
                    init_pb,
                    **(
                        scalarization_solver_options
                        if scalarization_solver_options is not None
                        else {}
                    ),
                )
                for (obj, init_pb) in enumerate(initial_subproblems_poll)
            ]
        else:
            tasks = [
                dask.delayed(solve_weighted_sum_subproblem)(
                    weights,
                    init_pb,
                    **(
                        scalarization_solver_options
                        if scalarization_solver_options is not None
                        else {}
                    ),
                )
                for (weights, init_pb) in zip(
                    self._order_cone.inequalities, initial_subproblems_poll
                )
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

    def _solve_vertex_selection_pb(
        self,
        outer_vertices: np.ndarray,
        inner_vertices: np.ndarray,
        previous_vertex_selection_solutions: np.ndarray,
        current_inner_vertex_indexes: List[int],
        solver_options: Optional[Dict],
    ) -> Tuple[np.ndarray, int]:
        """Solve the vertex solution problem in parallel.

        Solve max: QP(s, Ik): min || p - s ||^2
                               p
                              s.t. p in Ik = conv(F(xj)) + C
        for s vertex of the outer approximation.

        C is the ordering polyhedral cone.

        Arguments
        ---------
        outer_vertices: np.ndarray
            The list of outer vertices of the outer approximation.

        inner_vertices: np.ndarray
            The current list of objective vectors found.

        previous_vertex_selection_solutions:
            The pairs of solutions found during the previous resolution
            of the vertex selection problem.
            For each row, the first m columns contain the coordinates of
            the outer vertex and the last m columns the coordinates of
            the corresponding inner vertex where m is the number of objectives.

        current_inner_vertex_indexes:
            The indexes of the inner vertices found at the previous iteration.

        solver_options: optional[dict]
            The options of the solver used to solve the vertex selection
            problem.

        Returns
        -------
        np.ndarray:
            The new pairs of solutions found during the resolution of the
            vertex selection problem.

        int:
            The number of subproblems solved.
        """

        # Define task
        def solve_qp_subproblem(
            nobj: int,
            inner_vertices: np.ndarray,
            Y: Optional[np.ndarray],
            outer_vertex: np.ndarray,
            previous_vertex_selection_solutions: np.ndarray,
            current_inner_vertex_indexes: List[int],
            solver_options: Optional[Dict],
        ):
            # Check if it is an outer vertex from the previous outer approximation.
            previous_inner_vertex = None
            for vertex_pair in previous_vertex_selection_solutions:
                if (
                    np.linalg.norm(outer_vertex - vertex_pair[:nobj])
                    <= MIN_DIST_OBJ_VECS
                ):
                    previous_inner_vertex = vertex_pair[nobj:]
                    break

            # This is an outer vertex from the previous outer approximation
            # If it satisfies the optimality condition, no need to recompute it
            if previous_inner_vertex is not None:
                satisfy_optimality_conditions = True
                for current_inner_vertex_ind in current_inner_vertex_indexes:
                    if (
                        np.dot(
                            previous_inner_vertex - outer_vertex,
                            inner_vertices[current_inner_vertex_ind]
                            - previous_inner_vertex,
                        )
                        < 0
                    ):
                        satisfy_optimality_conditions = False
                        break
                if satisfy_optimality_conditions:
                    return np.hstack((outer_vertex, previous_inner_vertex)), False

            # Ik is defined by
            # Ik = {F(Xk) theta + Y mu: mu >= 0 and sum lambda_i = 1}
            # where F(Xk) = [F(x1) F(x2) ... F(x^|Xk|)]
            # and Y is the matrix of rays of the ordering cone.
            #
            # QP(s, Ik) is rewritten as
            #    min    || F(Xk) theta + Z mu - s ||^2
            # theta, mu
            # s.t. sum theta_i = 1
            #      mu >= 0
            #
            # We allocate the problem for each task to avoid concurrency issues when solving these
            # problems in parallel.
            ninner_vertices = inner_vertices.shape[0]
            theta = cp.Variable(ninner_vertices)
            mu = cp.Variable(nobj) if Y is None else cp.Variable(Y.shape[1])
            vertex_selection_constraints = [cp.sum(theta) == 1, mu >= 0, theta >= 0]
            vertex_selection_obj = cp.Minimize(
                cp.sum_squares(inner_vertices.T @ theta + mu - outer_vertex)
                if Y is None
                else cp.sum_squares(inner_vertices.T @ theta + Y @ mu - outer_vertex)
            )
            qp_pb = cp.Problem(vertex_selection_obj, vertex_selection_constraints)

            # Solve it
            try:
                if solver_options is None:
                    qp_pb.solve()
                else:
                    qp_pb.solve(**solver_options)
            except cp.SolverError:
                return np.array([]), False

            if qp_pb.status not in ["infeasible", "unbounded"]:
                p_values = (
                    inner_vertices.T @ theta.value + mu.value
                    if Y is None
                    else inner_vertices.T @ theta.value + Y @ mu.value
                )
                return np.hstack((outer_vertex, [val for val in p_values])), True

            return np.array([]), True

        nobj = len(self._objectives)
        Y = None if self._order_cone is None else self._order_cone.rays.T

        # Solve all problems in parallel
        tasks = [
            dask.delayed(
                solve_qp_subproblem(
                    nobj,
                    inner_vertices,
                    Y,
                    outer_vertex,
                    previous_vertex_selection_solutions,
                    current_inner_vertex_indexes,
                    solver_options,
                )
            )
            for outer_vertex in outer_vertices
        ]
        run_tasks = self._client.compute(tasks)
        optimization_results = self._client.gather(run_tasks)
        del tasks

        # Collect solutions and number of qp solved
        new_vertex_selection_solutions = []
        nb_qp_solved = 0
        for optimization_logs in optimization_results:
            if np.size(optimization_logs[0]) != 0:
                new_vertex_selection_solutions.append(optimization_logs[0])
            if optimization_logs[1]:
                nb_qp_solved += 1

        new_vertex_selection_solutions = np.asarray(new_vertex_selection_solutions)

        return new_vertex_selection_solutions, nb_qp_solved
