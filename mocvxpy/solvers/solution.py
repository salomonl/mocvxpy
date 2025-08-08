import numpy as np

from mocvxpy.constants import MIN_DIST_OBJ_VECS
from mocvxpy.utilities.polyhedron import Polyhedron
from typing import Optional


class Solution:
    """Contains the solutions of a multiobjective problem.

    Attributes:
    ----------
    nvars : int
        The number of decision variables of the problem
    nobj : int
        The number of objective of the problem
    """

    def __init__(self, nvars: int, nobj: int) -> None:
        if nvars <= 0:
            raise ValueError(
                "The number of variables of a problem must be positive", nvars
            )
        if nobj <= 1:
            raise ValueError(
                "The number of objectives of a problem must be superior or equal to 2",
                nobj,
            )

        self._nvars = nvars
        self._nobj = nobj
        self._xvalues = np.array([], dtype=np.float64).reshape((0, nvars))
        self._objective_values = np.array([], dtype=np.float64).reshape((0, nobj))
        self._dual_objective_values = np.array([], dtype=np.float64).reshape((0, nobj))
        self._dual_constraint_values = None

    @property
    def xvalues(self) -> np.ndarray:
        """Accessor method for decision variables values.

        Returns
        -------
        np.ndarray
           A matrix of decision variables values of dimensions |X| x nvars
        """
        return self._xvalues

    @property
    def objective_values(self) -> np.ndarray:
        """Accessor method for objective values.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions |X| x nobj
        """
        return self._objective_values

    @property
    def dual_objective_values(self) -> np.ndarray:
        """Accessor method for dual objective values.

        Returns
        -------
        np.ndarray
           A matrix of dual objective values (weights) of dimensions |X| x nobj
        """
        return self._dual_objective_values

    @property
    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Accessor method for dual objective values.

        Returns
        -------
        Optional[np.ndarray]
           A matrix of dual constraint values (weights) of dimensions |X| x cdims
        """
        return self._dual_constraint_values

    def insert_solution(
        self,
        x: np.ndarray,
        fvalues: np.ndarray,
        dual_fvalues: np.ndarray,
        dual_values: Optional[np.ndarray] = None,
    ) -> None:
        """Insert solution into the Solution set.

        Arguments
        -------
        x: np.ndarray
           A decision vector: it is the responsability from the user to check
           it is an efficient solution

        fvalues: np.ndarray
           A objective vector: it is the responsability from the user to check
           it is non dominated

        dual_fvalues: np.ndarray
           The corresponding dual objective values.

        dual_values: Optional[np.ndarray]
           The dual constraint values.
        """
        if x.shape != (self._nvars,):
            raise ValueError(
                "The dimensions of x are not compatible with the other solutions",
                x.shape,
            )
        if fvalues.shape != (self._nobj,):
            raise ValueError(
                "The dimensions of fvalues are not compatible with the other objective vectors",
                fvalues.shape,
            )
        if dual_fvalues.shape != (self._nobj,):
            raise ValueError(
                "The dimensions of dual_fvalues are not compatible with the other dual objective vectors",
                dual_fvalues.shape,
            )
        if dual_values is None:
            if self._dual_constraint_values is not None:
                raise ValueError("Dual values for constraints must be provided")
        else:
            if self._dual_constraint_values is not None:
                ncons = self._dual_constraint_values.shape[1]
                if dual_values.shape != (ncons,):
                    raise ValueError(
                        "The dimensions of dual_values are not compatible with the other dual vectors",
                        dual_values.shape,
                    )

        self._xvalues = np.vstack([self._xvalues, x])
        self._objective_values = np.vstack([self._objective_values, fvalues])
        self._dual_objective_values = np.vstack(
            [self._dual_objective_values, dual_fvalues]
        )
        if self._dual_constraint_values is None:
            self._dual_constraint_values = dual_values
            if self._dual_constraint_values is not None:
                self._dual_constraint_values = np.reshape(
                    self._dual_constraint_values, (1, self._dual_constraint_values.size)
                )
        else:
            self._dual_constraint_values = np.vstack(
                [self._dual_constraint_values, dual_values]
            )

    def extreme_objective_vectors(self) -> np.ndarray:
        """Compute the set of extreme objective vectors.

        An extreme objective vector is a point in the objective space
        who reaches a minimum f value for one of the solutions.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions nobj x nobj:
           row i corresponds to objective i.
        """
        extreme_pts_indexes = np.argmin(self.objective_values, axis=0)
        return self.objective_values[extreme_pts_indexes, :]

    def ideal_objective_vector(self) -> np.ndarray:
        """Compute the set of extreme objective vectors.

        An extreme objective vector is a point in the objective space
        who reaches a minimum f value for one of the solutions.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions nobj x nobj:
           row i corresponds to objective i.
        """
        return np.min(self.objective_values, axis=0)


class OuterApproximation:
    """Defines an outer approximation of a solution set
    for a multiobjective problem.

    Attributes:
    ----------
    nobj : int
        The number of objectives of the problem
    """

    def __init__(self, nobj: int) -> None:
        if nobj <= 1:
            raise ValueError(
                "The number of objectives of a problem must be superior or equal to 2",
                nobj,
            )
        self._halfspaces = np.array([], dtype=np.float64).reshape((0, nobj + 1))
        self._dim = nobj

    @property
    def halfspaces(self) -> np.ndarray:
        """Returns the halfspaces that define the outer approximation."""
        return self._halfspaces

    def insert_halfspace(self, halfspace: np.ndarray) -> None:
        """Insert a new halfspace into the current outer approximation

        Attributes:
        ----------
        halfspace : np.ndarray
           A vector [beta, a1, a2, ..., am] that characterizes the hyperplane
           0 <= beta + a^T
        """
        if halfspace.size != self._dim + 1:
            return

        self._halfspaces = np.vstack([self._halfspaces, halfspace])

    def compute_vertices(self) -> Optional[np.ndarray]:
        """Compute the vertices of the outer approximation."""
        if self.halfspaces.size == 0:
            return None

        poly = Polyhedron(A=-self.halfspaces[:, 1:], b=self.halfspaces[:, 0])

        # Reload the set of hyperplanes to try removing redundant hyperplanes
        self._halfspaces = poly.halfspaces

        # Generate outer vertices
        outer_vertices = poly.V

        # In the case where no vertex is present, it means that the polyhedron does not possess any vertex or
        # has only (0, ..., 0) as a vertex, which will supposed to be the case for the moment
        if outer_vertices.size == 0:
            outer_vertices = np.zeros((1, self._dim))

        # Remove all vertices that are too close to each other
        # Taken from D. Dorfler original implementation of this algorithm
        indices = []
        for ind_i in range(len(outer_vertices)):
            for ind_j in range(ind_i + 1, len(outer_vertices)):
                if (
                    np.linalg.norm(outer_vertices[ind_j] - outer_vertices[ind_i])
                    <= MIN_DIST_OBJ_VECS
                ):
                    indices.append(ind_i)
                    break
        mask = np.ones(len(outer_vertices), dtype=bool)
        mask[indices] = False

        return outer_vertices[mask, ...]


def update_local_lower_bounds(
    lower_bounds: np.ndarray, y: np.ndarray, nobj: int
) -> np.ndarray:
    """Update a local lower bounds set.

    Given an objective vector, update a set of local lower bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 2 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsability of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    lower_bounds: np.ndarray
        The set of local lower bounds. Has dimensions |L| x nobj.
    y: np.ndarray
        An update point of dimension nobj.

    Returns
    -------
    list[np.ndarray]
       The set of local lower bounds updated.
    """
    # Compute A = {l in lower_bounds | y > l}: Search zones that contain y
    L = lower_bounds
    A_indexes = np.all(np.greater(y, L), axis=1)

    # Compute Bi = {l in lower_bounds | yi = li and y(_i) > l(_i)} for i = 1, 2, ..., nobj
    # where y(_i) = (y1, ..., yi-1, y(i+1), ..., ym)
    B = [None] * nobj
    for i in range(nobj):
        B[i] = np.all(
            np.vstack(
                [
                    np.equal(y[i], L[:, i]),
                    np.greater(y[0:i], L[:, 0:i]).T,
                    np.greater(y[i + 1 :], L[:, i + 1 :]).T,
                ]
            ),
            axis=0,
        )

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local lower bounds of A
    P = [L[A_indexes] for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {l in Pi | l not >= l' for all l' in Pi cup Bi, l' != l}
    # for i = 1, 2, ..., m
    nlower_bounds = L.shape[0]
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        # Remove all dominated points from Pi by the ones of Pi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominated
                if comparison_status == 1:
                    P_indexes[i][ind1] = False
                    break
                # Point pind1 dominates or is equal to point pind2
                elif comparison_status == 2 or comparison_status == 3:
                    P_indexes[i][ind2] = False

            # no need to keep on iterating, the point pind1 is dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1 + 1, P[i].shape[0]):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominated
                if comparison_status == 1:
                    P_indexes[i][ind1] = False
                    break
                # Point pind1 dominates or is equal to point pind2
                elif comparison_status == 2 or comparison_status == 3:
                    P_indexes[i][ind2] = False

        # Remove points of Pi dominated by points of Bi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(nlower_bounds):
                if not B[i][ind2]:
                    continue

                comparison_status = compare_objective_vectors(P[i][ind1], L[ind2])
                # Point at index ind1 is dominated by another point of Bi or equal
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    new_lower_bounds = np.vstack([L[~A_indexes, :], np.vstack(P)])

    return new_lower_bounds


def update_local_upper_bounds(
    upper_bounds: np.ndarray, y: np.ndarray, nobj: int
) -> np.ndarray:
    """Update a local upper bounds set.

    Given an objective vector, update a set of local upper bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 1 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsability of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    upper_bounds: np.ndarray
        The set of local upper bounds. Has dimensions (|U|, nobj).
    y: np.ndarray
        An update point of dimension nobj.

    Returns
    -------
    list[np.ndarray]
       The set of local upper bounds updated.
    """
    # Compute A = {u in upper_bounds | y < u}: Search zones that contain y
    # U = np.array(upper_bounds)
    U = upper_bounds
    A_indexes = np.all(np.less(y, U), axis=1)

    # Compute Bi = {u in upper_bounds | yi = ui and y(_i) < u(_i)} for i = 1, 2, ..., nobj
    # where u(_i) = (y1, ..., yi-1, u(i+1), ..., ym)
    B = [None] * nobj
    for i in range(nobj):
        B[i] = np.all(
            np.vstack(
                [
                    np.equal(y[i], U[:, i]),
                    np.less(y[0:i], U[:, 0:i]).T,
                    np.less(y[i + 1 :], U[:, i + 1 :]).T,
                ]
            ),
            axis=0,
        )

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local upper bounds of A
    P = [U[A_indexes] for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {u in Pi | u not <= u' for all u' in Pi cup Bi, u' != u}
    # for i = 1, 2, ..., m
    nupper_bounds = U.shape[0]
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        for ind1 in range(P[i].shape[0]):
            # The point has been already processed
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominating
                if comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break
                # Point pind2 dominates or equals point pind1
                elif comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind2] = False

            # no need to continue, the point pind1 is dominating
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1 + 1, P[i].shape[0]):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominating
                if comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break
                # Point pind2 dominates or equals point pind1
                elif comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind2] = False

        # Filter out redundant points of Pi by Bi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(nupper_bounds):
                if not B[i][ind2]:
                    continue

                comparison_status = compare_objective_vectors(P[i][ind1], U[ind2])
                # Point pind1 dominates or equals a point in B
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    # new_upper_bounds = (upper_bounds \ A) cup P
    new_upper_bounds = np.vstack([U[~A_indexes, :], np.vstack(P)])

    return new_upper_bounds


def compare_objective_vectors(y1: np.ndarray, y2: np.ndarray) -> int:
    """Compare two objective vectors according to Pareto dominance.

    Arguments
    ---------
    y1: np.ndarray
        The first objective vector.
    y2: np.ndarray
        The second objective vector.
    nobj: int
        The number of objectives.

    Returns
    -------
    int:
       0 if y1 is non-dominated by y2;
       1 if y1 is dominated by y2;
       2 if y1 is dominating y2;
       3 if y1 and y2 are equal.
    """
    is_better = np.any(np.less(y1, y2))
    is_worse = np.any(np.less(y2, y1))
    if is_worse:
        if is_better:
            # non-dominated
            return 0
        else:
            # dominated
            return 1
    else:
        if is_better:
            # dominating
            return 2
        else:
            # equal
            return 3
