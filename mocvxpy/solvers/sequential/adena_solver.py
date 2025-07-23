import cvxpy as cp
import numpy as np
import time

from mocvxpy.problems.utilities import number_of_variables
from mocvxpy.solvers.common import compute_extreme_objective_vectors
from mocvxpy.constants import (
    ADENA_BOX_EXTENSION_TOL,
    ADENA_MAX_ITER,
    ADENA_MAX_PB_SOLVED,
    ADENA_MIN_STOPPING_TOL,
)
from mocvxpy.solvers.solution import (
    Solution,
    update_local_lower_bounds,
    update_local_upper_bounds,
)
from mocvxpy.subproblems.pascoletti_serafini import PascolettiSerafiniSubproblem
from typing import Dict, List, Optional, Tuple, Union


class ADENASolver:
    """The ADENA (ADvanced ENclosure Algorithm) solver.

    Described in
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    Arguments
    ---------
    objectives: list[Minimize or Maximize]
        The problem's objectives.
    constraints: list
        The constraints on the problem variables.
    """

    def __init__(
        self,
        objectives: List[Union[cp.Minimize, cp.Maximize]],
        constraints: Optional[List[cp.Constraint]] = None,
    ) -> None:
        nobj = len(objectives)
        if nobj <= 1:
            raise ValueError("The number of objectives must be superior to 1", nobj)
        self._objectives = objectives

        if constraints is None:
            self._constraints = []
        else:
            self._constraints = constraints

    def solve(
        self,
        verbose: bool = True,
        stopping_tol: float = 0.005,
        max_iter: int = 50,
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
           the algorithm stops. Is always above or equal to ADENA_MIN_STOPPING_TOL = 1e-6.

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

        if verbose:
            print("ADENA algorithm:")
            print("Number of objectives: ", nobj)
            print("Number of variables: ", nvars)

        initial_step_status, sol = self._initial_step(sol, scalarization_solver_options)
        if initial_step_status != "solved":
            if verbose:
                print(
                    "ADENA initialization failure: the algorithm cannot obtain extreme solutions;",
                    "initialisation phase status returns:",
                    initial_step_status,
                )
            return "no_extreme_solutions", sol

        # Initialize initial lower and upper bound sets
        # TODO: nothing could prevent some solutions from being inside the enclosure
        # (from above: what to do in this case ?)
        lower_bounds = np.min(sol.objective_values, axis=0) - ADENA_BOX_EXTENSION_TOL
        lower_bounds = np.reshape(lower_bounds, (1, nobj))
        upper_bounds = np.max(sol.objective_values, axis=0) + ADENA_BOX_EXTENSION_TOL
        upper_bounds = np.reshape(upper_bounds, (1, nobj))

        # Compute scaled stopping tolerance
        min_enclosure_width = max(
            1,
            min(upper_bounds[0][obj] - lower_bounds[0][obj] for obj in range(nobj)),
        ) * max(stopping_tol, ADENA_MIN_STOPPING_TOL)
        if verbose:
            print(f"Stopping tolerance: {min_enclosure_width:.5E}")
            print()
            print(
                f"{"iter":>7} {"nb_solutions":>13} ",
                f"{"|lower_bounds|":>12} {"|upper_bounds|":>12} {"|SUP(l, u)|":>12}",
                f"{"enclosure width":>8}",
                f"{"time_SUP_pb_solve (s)":>21} {"avg_time_SUP_pb_solve (s)":>25} {"update bounds (s)":>17}",
            )
            print()

        # Initialize subproblem
        sup_pb = PascolettiSerafiniSubproblem(self._objectives, self._constraints, None)

        # Set options
        max_iter = min(max_iter, ADENA_MAX_ITER)
        max_pb_solved = min(max_pb_solved, ADENA_MAX_PB_SOLVED)

        nb_subproblems_solved_per_iter = 0
        total_pb_solved = 0
        elapsed_subproblems_per_iter = 0.0
        elapsed_update_bounds = 0.0
        status = "max_iter_reached"
        start_optimization = time.perf_counter()
        for iter in range(max_iter):
            # Compute enclosure width
            enclosure_width = -np.inf
            for lb in lower_bounds:
                for ub in upper_bounds:
                    if (lb <= ub).all():
                        enclosure_width = max(
                            min(ub[obj] - lb[obj] for obj in range(nobj)),
                            enclosure_width,
                        )

            if verbose:
                print(
                    f"{iter+1:5d} {len(sol.objective_values):10d} ",
                    f"{len(lower_bounds):13d} {len(upper_bounds):14d}",
                    f"{nb_subproblems_solved_per_iter:15d} {enclosure_width:17e}",
                    (
                        f"{"-":>14}"
                        if total_pb_solved == 0
                        else f"{elapsed_subproblems_per_iter:19e}"
                    ),
                    (
                        f"{"-":>20}"
                        if total_pb_solved == 0
                        else f"{elapsed_subproblems_per_iter/nb_subproblems_solved_per_iter:21e}"
                    ),
                    (
                        f"{"-":>20}"
                        if total_pb_solved == 0
                        else f"{elapsed_update_bounds:20e}"
                    ),
                )

            if enclosure_width <= min_enclosure_width:
                status = "optimal"
                break

            if status == "max_pbs_solved_reached":
                break

            nb_subproblems_solved_per_iter = 0
            elapsed_subproblems_per_iter = 0.0
            elapsed_update_bounds = 0.0
            for lb in lower_bounds:
                # Identify the upper bound element for a given lower bound element with maximum shortest edge length
                shortest_edge_length = -np.inf
                ub = None
                for u in upper_bounds:
                    tmp_shortest_edge_length = min(
                        u[obj] - (lb[obj] + min_enclosure_width) for obj in range(nobj)
                    )
                    if shortest_edge_length < tmp_shortest_edge_length:
                        shortest_edge_length = tmp_shortest_edge_length
                        ub = u

                if shortest_edge_length <= 0:
                    continue

                # Solve sup subproblem, that is a Pascoletti-Serafini subproblem
                # with vref = lb and direction = ub - lb
                sup_pb.parameters = np.concatenate((lb, ub - lb))
                start_sup_pb_solved = time.perf_counter()
                if scalarization_solver_options is None:
                    status_sup_pb = sup_pb.solve()
                else:
                    status_sup_pb = sup_pb.solve(**scalarization_solver_options)
                end_sup_pb_solved = time.perf_counter()

                if status_sup_pb != "solved":
                    continue

                elapsed_subproblems_per_iter += end_sup_pb_solved - start_sup_pb_solved
                nb_subproblems_solved_per_iter += 1
                total_pb_solved += 1

                # Update solution
                opt_obj_values = sup_pb.objective_values()
                sol.insert_solution(
                    sup_pb.solution(),
                    opt_obj_values,
                    sup_pb.dual_objective_values(),
                    sup_pb.dual_constraint_values(),
                )

                # Update lower and upper bound sets
                t_opt = sup_pb.value()
                y_lower = lb + t_opt * (ub - lb)
                y_upper = opt_obj_values
                start_update_bounds = time.perf_counter()
                lower_bounds = update_local_lower_bounds(lower_bounds, y_lower, nobj)
                upper_bounds = update_local_upper_bounds(upper_bounds, y_upper, nobj)
                end_update_bounds = time.perf_counter()
                elapsed_update_bounds += end_update_bounds - start_update_bounds

                if total_pb_solved == max_pb_solved:
                    status = "max_pbs_solved_reached"
                    break

        end_optimization = time.perf_counter()
        if verbose:
            print("\nStopping reason:", status)
            print("Resolution time (s):", end_optimization - start_optimization)
            print(
                "enclosure width value:",
                f"{enclosure_width:.5E}",
            )
            print("Number of solutions found:", len(sol.xvalues))
            print(
                "Number of local lower bounds found:",
                len(lower_bounds),
            )
            print("Number of local upper bounds found:", len(upper_bounds))
            print()

        return status, sol

    def _initial_step(
        self, sol: Solution, scalarization_solver_options: Optional[Dict]
    ) -> Tuple[str, Solution]:
        """The initial step.

        Compute extreme solutions and their objective values.

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
