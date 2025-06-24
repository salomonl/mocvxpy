import cvxpy as cp
import numpy as np
import time

from mocvxpy.constants import MIN_DIST_OBJ_VECS
from mocvxpy.expressions.order_cone import OrderCone
from mocvxpy.solvers.solution import OuterApproximation, Solution
from typing import List, Optional, Tuple, Union

from mocvxpy.solvers.utilities import (
    compute_extreme_objective_vectors,
    compute_extreme_points_hyperplane,
    extract_variables_from_problem,
    number_of_variables,
    solve_weighted_sum_problem
)


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
        order_cone: Optional[OrderCone] = None
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
            self, stopping_tol: float = 5 * 1e-4, maxiter: int = 10, max_pb_solved: int = 200,
            verbose: bool = True
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
            print("MONMO algorithm:")
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
                    "MONMO initialization failure: the algorithm cannot obtain extreme solutions;",
                    "initialisation phase status returns:", initial_step_status
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
        scaled_stopping_tol = stopping_tol * max(1.0, ideal_to_extreme_pts_hyp_dist)

        if verbose:
            print(f"Stopping tolerance: {scaled_stopping_tol:.5E}")

        haussdorf_dist = np.inf
        if verbose:
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|vert(Ok)|":>10} {"|halfspaces(Ok)|":>16} ",
                f"{"nb_pbs_solved_per_iter":>22} {"|unknown_vertices|":>18} {"Haussdorf_dist":>14} ",
                f"{"time_NM_pb_solve (s)":>20} {"avg_time_NM_pb_solve (s)":>24}"
            )
            print()
            print(
                f"{0:5d} {len(sol.objective_values):10d} ",
                f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                f"{"-":>20} {"-":>19} {haussdorf_dist:21e} ",
                f"{"-":>12} {"-":>20}"
            )

        known_outer_vertices = []

        status = "max_iter_reached"
        start_optimization = time.perf_counter()
        elapsed_total_subproblems = 0.
        total_nm_pbs_solved = 0
        total_nm_pbs_failed = 0
        for iter in range(maxiter):
            # Compute the set of unknown outer vertices
            unknown_outer_vertices = []
            for vertex in outer_vertices:
                is_a_known_vertex = False
                for vknown in known_outer_vertices:
                    if np.linalg.norm(vertex - vknown) <= MIN_DIST_OBJ_VECS:
                        is_a_known_vertex = True
                        break
                if not is_a_known_vertex:
                    unknown_outer_vertices.append(vertex)

            haussdorf_dist = -np.inf
            nb_subproblems_solved_per_iter = 0
            nb_subproblems_failed_per_iter = 0
            update_outer_approximation = False
            elapsed_subproblems_per_iter = 0.
            for vertex in unknown_outer_vertices:
                v = np.asarray(vertex)
                start_nm_pb_solved = time.perf_counter()
                opt_values, opt_val, w_opt, status_norm_min = (
                    self._solve_norm_min_pb(v)
                )
                end_nm_pb_solved = time.perf_counter()

                # If there is a failure, we ignore it and try to compute
                # the norm inf problem for another vertex
                if status_norm_min != "solved":
                    nb_subproblems_failed_per_iter += 1
                    total_nm_pbs_failed += 1
                    continue

                elapsed_subproblems_per_iter += end_nm_pb_solved - start_nm_pb_solved

                nb_subproblems_solved_per_iter += 1
                total_nm_pbs_solved += 1

                known_outer_vertices.append(v)

                # Update solution
                sol.insert_solution(opt_values[:nvars], opt_values[nvars:])
                haussdorf_dist = max(haussdorf_dist, opt_val)
                if opt_val > scaled_stopping_tol:
                    update_outer_approximation = True
                    if Z is None:
                        outer_approximation.insert_halfspace(
                            np.asarray([-np.dot(opt_values[nvars:], w_opt)] + w_opt.tolist())
                        )
                    else:
                        outer_approximation.insert_halfspace(
                            np.asarray(
                                [-np.dot(opt_values[nvars:], Z.T @ w_opt)] + (Z.T @ w_opt).tolist()
                            )
                        )
                    # outer_approximation.insert_halfspace(
                    #     np.asarray([-np.dot(opt_values[nvars:], w_opt)] + w_opt.tolist())
                    # )


                if total_nm_pbs_solved >= max_pb_solved:
                    break

            elapsed_total_subproblems += elapsed_subproblems_per_iter
            outer_vertices: np.ndarray = outer_approximation.compute_vertices()
            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.xvalues):10d} ",
                    f"{len(outer_vertices):13d} {len(outer_approximation.halfspaces):12d} ",
                    f"{nb_subproblems_solved_per_iter:20d} {len(unknown_outer_vertices):19d} {haussdorf_dist:21e}",
                    f"{elapsed_subproblems_per_iter:19e} {elapsed_total_subproblems / total_nm_pbs_solved:19e}"
                )

            if nb_subproblems_failed_per_iter == len(unknown_outer_vertices):
                status = "norm_min_subproblem_failure"
                break

            if total_nm_pbs_solved >= max_pb_solved:
                status = "max_pbs_solved_reached"
                break

            if update_outer_approximation:
                continue

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
            print("Average time on nm problems solved:", elapsed_total_subproblems / total_nm_pbs_solved)
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


    def _solve_norm_min_pb(
        self, vref: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, str]:
        """Solve the norm minimization subproblem.

        Solve min || z ||
              s.t. Z f(x) <= Z (z + vref)
              x in Omega
        where Z defines the ordering cone
        C = {y : Z y >= 0}
        of the multiobjective optimization problem.

        and its dual.

        Arguments
        ---------
        vref: np.ndarray
            The outer vertex target.

        Returns
        -------
        np.ndarray:
            The optimal solution of the problem.
            Contains the optimal decision values and their associated objective values.

        float:
            The optimal value of the norm min subproblem.

        np.ndarray:
            The optimal dual values of the problem,
            that correspond to the dual optimal values of the constraints associated
            to the objective functions.

        str:
            The status of the resolution.
        """
        nobj = len(self._objectives)
        vars_ = extract_variables_from_problem(self._objectives, self._constraints)
        Z = None if self._order_cone is None else self._order_cone.inequalities

        z = cp.Variable(nobj)
        norm_min_constraints = [cstr for cstr in self._constraints]

        # Add constraints: Z f(x) <= Z (v + z)
        if Z is None:
            # Use the Pareto dominance cone
            for obj, objective in enumerate(self._objectives):
                norm_min_constraints.append(objective.expr <= vref[obj] + z[obj])
        else:
            for zrow in Z:
                norm_min_constraints.append(
                    sum(
                        zrow[obj] * (objective.expr - vref[obj] - z[obj])
                        for obj, objective in enumerate(self._objectives)
                    )
                    <= 0
                )

        norm_min_pb = cp.Problem(cp.Minimize(cp.norm(z)), norm_min_constraints)
        try:
            norm_min_pb.solve(solver=cp.MOSEK)
        except:
            return np.zeros([]), 0.0, np.zeros([]), "unsolved"

        if norm_min_pb.status not in ["infeasible", "unbounded"]:
            # Collect dual values associated to the rays of the order cone
            dual_obj_constraints_vals = (
                np.zeros(nobj) if Z is None else np.zeros(Z.shape[0])
            )
            if Z is None:
                for obj in range(nobj):
                    dual_obj_constraints_vals[obj] = norm_min_constraints[
                        -nobj + obj
                    ].dual_value
            else:
                for ind in range(Z.shape[0]):
                    dual_obj_constraints_vals[ind] = norm_min_constraints[
                        -Z.shape[0] + ind
                    ].dual_value

            # Collect optimal decision and objective values
            opt_values = []
            for var in vars_:
                opt_values += [val for val in var.value]
            opt_values += [objective.expr.value for objective in self._objectives]
            return (
                np.asarray(opt_values),
                norm_min_pb.value,
                dual_obj_constraints_vals,
                "solved",
            )

        return np.zeros([]), 0.0, np.zeros([]), "unsolved"
