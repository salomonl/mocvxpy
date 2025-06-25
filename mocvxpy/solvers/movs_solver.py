import cvxpy as cp
import numpy as np
import time

from mocvxpy.constants import MIN_DIST_OBJ_VECS
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.solvers.solution import OuterApproximation, Solution
from mocvxpy.solvers.utilities import (
    compute_extreme_objective_vectors,
    compute_extreme_points_hyperplane,
    number_of_variables,
    solve_weighted_sum_problem,
)
from mocvxpy.subproblems.pascoletti_serafini import PascolettiSerafiniSubproblem
from typing import List, Optional, Tuple, Union


class MOSVSolver:
    """The MOSV (MultiObjective Optimization by Vertex Selection) solver.

    Described in
    Dörfler, D., Löhne, A., Schneider, C., & Weißing, B. (2021).
    "A Benson-type algorithm for bounded convex vector optimization problems with vertex selection".
    Optimization Methods and Software, 37(3), p. 1006–1026.
    https://doi.org/10.1080/10556788.2021.1880579

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
        self, stopping_tol: float = 5 * 1e-4, maxiter: int = 100, verbose: bool = True
    ) -> Tuple[str, Solution]:
        """Solve the problem.

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
            print("MOVS algorithm:")
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

        initial_step_status, sol = self._initial_step(sol)
        if initial_step_status != "solved":
            if verbose:
                print(
                    "MOVS initialization failure: the algorithm cannot obtain extreme solutions;",
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
        extreme_pts_hyp_params: np.ndarray = compute_extreme_points_hyperplane(
            sol.extreme_objective_vectors()
        )
        # 2- Compute distance from the ideal point to the hyperplane
        #    Given a hyperplane a^T x - 1 = 0 and a point x0, the distance is
        #    given by: d = |a^T x0 - 1| / ||a||
        ideal_to_extreme_pts_hyp_dist = np.absolute(
            np.dot(extreme_pts_hyp_params, sol.ideal_objective_vector()) - 1.0
        ) / np.linalg.norm(extreme_pts_hyp_params)
        scaled_stopping_tol = stopping_tol * max(1.0, ideal_to_extreme_pts_hyp_dist)

        if verbose:
            print(f"Stopping tolerance: {scaled_stopping_tol:.5E}")

        haussdorf_dist = np.inf
        if verbose:
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|vert(Ok)|":>10} {"|halfspaces(Ok)|":>16} ",
                f"{"|QP(s, Ik)|":>11} {"Haussdorf_dist":>14} ",
                f"{"time_QP_solve (s)":>17} {"time_PS_pb_solve (s)":>20}",
            )
            print()
            print(
                f"{0:5d} {len(sol.objective_values):10d} ",
                f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                f"{"-":>13}",
                f"{haussdorf_dist:14e}",
            )

        # Initialize subproblem
        ps_pb = PascolettiSerafiniSubproblem(
            self._objectives, self._constraints, self._order_cone
        )

        status = "maxiter_reached"
        vertex_selection_solutions = np.array([]).reshape((0, 2 * nobj))
        visited_outer_vertices = np.array([]).reshape((0, nobj))
        current_inner_vertex_ind = -1
        elapsed_ps_pb = 0.0
        start_optimization = time.perf_counter()
        for iter in range(maxiter):
            start_vertex_selection_pb = time.perf_counter()
            vertex_selection_solutions, opt_pair_ind, nb_qp_solved = (
                self._solve_vertex_selection_pb(
                    outer_vertices,
                    sol.objective_values,
                    vertex_selection_solutions,
                    current_inner_vertex_ind,
                )
            )
            end_vertex_selection_pb = time.perf_counter()
            elapsed_vertex_selection_pb = (
                end_vertex_selection_pb - start_vertex_selection_pb
            )

            s = vertex_selection_solutions[opt_pair_ind][:nobj]
            p = vertex_selection_solutions[opt_pair_ind][nobj:]
            haussdorf_dist = np.linalg.norm(p - s)

            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.objective_values):10d} ",
                    f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                    f"{nb_qp_solved:14d} {haussdorf_dist:17e}",
                    f"{elapsed_vertex_selection_pb:16e} {elapsed_ps_pb:16e}",
                )

            if haussdorf_dist <= scaled_stopping_tol:
                status = "solved"
                break

            if nb_qp_solved == 0:
                # No new qp has been recomputed
                # If there exists some outer vertices that have not been visited yet, take the corresponding
                # vertex solution pair with the maximum distance
                vertex_selection_candidates = []
                for sp_pair in vertex_selection_solutions:
                    is_visited = False
                    for v in visited_outer_vertices:
                        if np.linalg.norm(sp_pair[:nobj] - v) <= MIN_DIST_OBJ_VECS:
                            is_visited = True
                            break
                    if not is_visited:
                        vertex_selection_candidates.append(sp_pair)

                if not vertex_selection_candidates:
                    status = "vertex_selection_failure"
                    break

                sorted_vertex_solutions_indexes = np.argsort(
                    [
                        np.linalg.norm(v[:nobj] - v[nobj:])
                        for v in vertex_selection_candidates
                    ]
                )
                s = vertex_selection_solutions[sorted_vertex_solutions_indexes[-1]][
                    :nobj
                ]
                p = vertex_selection_solutions[sorted_vertex_solutions_indexes[-1]][
                    nobj:
                ]

            v = s
            c = p - s
            ps_pb.parameters = np.concatenate((v, c))

            start_ps_pb = time.perf_counter()
            status_ps = ps_pb.solve()
            end_ps_pb = time.perf_counter()
            elapsed_ps_pb = end_ps_pb - start_ps_pb

            if status_ps != "solved":
                status = "ps_subproblem_failure"
                break

            # Update solution
            sol.insert_solution(ps_pb.solution(), ps_pb.objective_values())
            current_inner_vertex_ind = len(sol.objective_values) - 1

            z_opt = ps_pb.value()
            w_opt = ps_pb.dual_objective_values()
            if Z is None:
                outer_approximation.insert_halfspace(
                    np.asarray([-z_opt - np.dot(v, w_opt)] + w_opt.tolist())
                )
            else:
                outer_approximation.insert_halfspace(
                    np.asarray(
                        [-z_opt - np.dot(v, Z.T @ w_opt)] + (Z.T @ w_opt).tolist()
                    )
                )
            outer_vertices = outer_approximation.compute_vertices()

            visited_outer_vertices = np.vstack([visited_outer_vertices, s])

        end_optimization = time.perf_counter()
        if verbose:
            print("\nStopping reason:", status)
            print("Resolution time (s):", end_optimization - start_optimization)
            print(
                "Hausdorff distance between outer and inner approximation of the solution set:",
                f"{haussdorf_dist:.5E}",
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

    def _initial_step(self, sol: Solution) -> Tuple[str, Solution]:
        """The initial step.

        Compute extreme solutions and their objective values according to the ordering cone.

        Arguments
        ---------
        sol: Solution
            The solution (initialized).

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
                self._objectives, self._constraints
            )
            for opt_values, obj_values in zip(
                init_phase_result[1], init_phase_result[2]
            ):
                sol.insert_solution(opt_values, obj_values)
            return init_phase_result[0], sol

        status = "solved"
        for weights in self._order_cone.inequalities:
            weighted_sum_result = solve_weighted_sum_problem(
                self._objectives, self._constraints, weights
            )
            if weighted_sum_result[0] == "solved":
                sol.insert_solution(weighted_sum_result[1], weighted_sum_result[2])
                continue

            if weighted_sum_result[0] == "infeasible":
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
        current_inner_vertex_ind: int,
    ) -> Tuple[np.ndarray, int, int]:
        """Solve the vertex solution problem.

        Solve max: QP(s, Ik): min || p - s ||^2
                               p
                              s.t. p in Ik = conv(F(xj)) + C
        for s vertex of the outer approximation.

        C is the ordering polyhedral cone.

        Arguments
        ---------
        outer_vertices: np.ndarray
            The list of outer vertices of the outer approximation

        inner_vertices: np.ndarray
            The current list of objective vectors found

        previous_vertex_selection_solutions:
            The pairs of solutions found during the previous resolution
            of the vertex selection problem.
            For each row, the first nobj columns contain the coordinates of
            the outer vertex and the last nobj columns the coordinates of
            the corresponding inner vertex.

        Returns
        -------
        np.ndarray:
            The new pairs of solutions found during the resolution of the
            vertex selection problem.

        int:
            The index of the optimal solution pair.

        int:
            The number of subproblems solved.
        """
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
        nobj = len(self._objectives)
        Y = None if self._order_cone is None else self._order_cone.rays.T
        ninner_vertices = inner_vertices.shape[0]
        theta = cp.Variable(ninner_vertices)
        mu = cp.Variable(nobj) if Y is None else cp.Variable(Y.shape[1])
        vertex_selection_constraints = [cp.sum(theta) == 1, mu >= 0, theta >= 0]

        # Initialize QP(s, Ik) problem
        outer_v = cp.Parameter(nobj)
        vertex_selection_obj = cp.Minimize(
            cp.sum_squares(inner_vertices.T @ theta + mu - outer_v)
            if Y is None
            else cp.sum_squares(inner_vertices.T @ theta + Y @ mu - outer_v)
        )
        qp_pb = cp.Problem(vertex_selection_obj, vertex_selection_constraints)

        nb_qp_solved = 0
        dist_p2s = -np.inf

        new_vertex_selection_solutions = []
        opt_pair_ind = -1
        cur_ind = 0
        for outer_vertex in outer_vertices:

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
            if (
                previous_inner_vertex is not None
                and np.dot(
                    previous_inner_vertex - outer_vertex,
                    inner_vertices[current_inner_vertex_ind] - previous_inner_vertex,
                )
                >= 0
            ):
                new_vertex_selection_solutions.append(
                    [val for val in outer_vertex]
                    + [val for val in previous_inner_vertex]
                )
                # Update the best current solution if required
                cur_dist_p2s = np.sum((previous_inner_vertex - outer_vertex) ** 2)
                if cur_dist_p2s > dist_p2s:
                    dist_p2s = cur_dist_p2s
                    opt_pair_ind = cur_ind

                cur_ind += 1
                continue

            # Otherwise, we need to solve the qp problem for vertex selection
            outer_v.value = outer_vertex
            try:
                qp_pb.solve(solver=cp.GUROBI)
            except cp.SolverError:
                continue

            if qp_pb.status not in ["infeasible", "unbounded"]:
                p_values = (
                    inner_vertices.T @ theta.value + mu.value
                    if Y is None
                    else inner_vertices.T @ theta.value + Y @ mu.value
                )
                # Store the pair (s, p*)
                new_vertex_selection_solutions.append(
                    [val for val in outer_vertex] + [val for val in p_values]
                )

                # Update the best current solution if required
                cur_dist_p2s = np.sum((p_values - outer_vertex) ** 2)
                if cur_dist_p2s > dist_p2s:
                    dist_p2s = cur_dist_p2s
                    opt_pair_ind = cur_ind

                cur_ind += 1

            nb_qp_solved += 1

        new_vertex_selection_solutions = np.asarray(new_vertex_selection_solutions)

        return new_vertex_selection_solutions, opt_pair_ind, nb_qp_solved
